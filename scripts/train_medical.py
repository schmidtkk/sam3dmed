"""
Medical fine-tuning training script for SAM3D.

This script trains the SLatMeshDecoder with LoRA adapters for medical imaging.
Supports mesh-only training with SDF + Chamfer + mesh regularization losses.

Usage:
    python scripts/train_medical.py \
        --data_root /path/to/preprocessed \
        --checkpoint_dir ./checkpoints/medical \
        --batch_size 4 \
        --epochs 50 \
        --lr 1e-3
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class MedicalTrainingLosses(nn.Module):
    """
    Combined loss functions for medical mesh training.

    Supports:
    - SDF loss (L1/MSE on predicted vs GT SDF)
    - Chamfer distance loss on mesh vertices
    - Mesh regularization (Laplacian smoothing, edge length)
    - Occupancy/mask loss
    """

    def __init__(
        self,
        sdf_weight: float = 1.0,
        chamfer_weight: float = 0.5,
        mesh_reg_weight: float = 0.1,
        occupancy_weight: float = 0.1,
        sdf_loss_type: str = "l1",  # "l1" or "mse"
    ):
        super().__init__()
        self.sdf_weight = sdf_weight
        self.chamfer_weight = chamfer_weight
        self.mesh_reg_weight = mesh_reg_weight
        self.occupancy_weight = occupancy_weight
        self.sdf_loss_type = sdf_loss_type

        # Import metrics lazily to avoid import errors
        self._chamfer_fn = None

    @property
    def chamfer_fn(self):
        """Lazy load chamfer function."""
        if self._chamfer_fn is None:
            from sam3d_objects.utils.metrics import compute_chamfer

            self._chamfer_fn = compute_chamfer
        return self._chamfer_fn

    def compute_sdf_loss(
        self,
        pred_sdf: torch.Tensor,
        gt_sdf: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute SDF reconstruction loss.

        Args:
            pred_sdf: Predicted SDF values (B, D, H, W) or (B, C, D, H, W)
            gt_sdf: Ground truth SDF values
            mask: Optional mask for valid regions
        """
        # If pred and gt sizes mismatch, crop to the minimum overlapping region to avoid
        # broadcasting/shape mismatch errors. This is helpful for datasets with variable
        # slice sizes during development sanity runs.
        if pred_sdf.shape != gt_sdf.shape:
            # Align dimensions: support (B, C, H, W) and (B, C, D, H, W)
            pred_shape = pred_sdf.shape
            gt_shape = gt_sdf.shape
            # Work with last two dims (H, W) primarily; for volumetric GT, index the last two
            h_pred, w_pred = pred_shape[-2], pred_shape[-1]
            h_gt, w_gt = gt_shape[-2], gt_shape[-1]
            h_min, w_min = min(h_pred, h_gt), min(w_pred, w_gt)
            # Crop both tensors to overlapping region
            if pred_sdf.ndim == 4:
                pred_sdf = pred_sdf[..., :h_min, :w_min]
            elif pred_sdf.ndim == 5:
                pred_sdf = pred_sdf[..., :h_min, :w_min]
            if gt_sdf.ndim == 4:
                gt_sdf = gt_sdf[..., :h_min, :w_min]
            elif gt_sdf.ndim == 5:
                gt_sdf = gt_sdf[..., :h_min, :w_min]

        if self.sdf_loss_type == "l1":
            loss = F.l1_loss(pred_sdf, gt_sdf, reduction="none")
        else:
            loss = F.mse_loss(pred_sdf, gt_sdf, reduction="none")

        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)
        return loss.mean()

    def compute_chamfer_loss(
        self,
        pred_vertices: torch.Tensor,
        gt_vertices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Chamfer distance between predicted and GT mesh vertices.

        Args:
            pred_vertices: Predicted vertex positions (N, 3)
            gt_vertices: Ground truth vertex positions (M, 3)
        """
        return self.chamfer_fn(pred_vertices, gt_vertices)

    def compute_mesh_regularization(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mesh regularization loss (Laplacian smoothing + edge length).

        Args:
            vertices: Mesh vertices (V, 3)
            faces: Mesh faces (F, 3)
        """
        # Edge length regularization
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        edge_lengths = torch.cat(
            [
                (v0 - v1).norm(dim=-1),
                (v1 - v2).norm(dim=-1),
                (v2 - v0).norm(dim=-1),
            ]
        )

        # Penalize variance in edge lengths (encourages uniform triangles)
        edge_reg = edge_lengths.var()

        return edge_reg

    def compute_occupancy_loss(
        self,
        pred_occupancy: torch.Tensor,
        gt_occupancy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute binary occupancy loss.

        Args:
            pred_occupancy: Predicted occupancy (logits or probabilities)
            gt_occupancy: Ground truth occupancy (binary)
        """
        return F.binary_cross_entropy_with_logits(pred_occupancy, gt_occupancy.float())

    def forward(
        self,
        pred_sdf: torch.Tensor | None = None,
        gt_sdf: torch.Tensor | None = None,
        pred_vertices: torch.Tensor | None = None,
        gt_vertices: torch.Tensor | None = None,
        pred_faces: torch.Tensor | None = None,
        pred_occupancy: torch.Tensor | None = None,
        gt_occupancy: torch.Tensor | None = None,
        sdf_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute all losses.

        Returns:
            Dict with individual losses and total loss
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=self._get_device(pred_sdf, pred_vertices))

        # SDF loss
        if pred_sdf is not None and gt_sdf is not None:
            sdf_loss = self.compute_sdf_loss(pred_sdf, gt_sdf, sdf_mask)
            losses["sdf"] = sdf_loss
            total_loss = total_loss + self.sdf_weight * sdf_loss

        # Chamfer loss
        if pred_vertices is not None and gt_vertices is not None:
            chamfer_loss = self.compute_chamfer_loss(pred_vertices, gt_vertices)
            losses["chamfer"] = chamfer_loss
            total_loss = total_loss + self.chamfer_weight * chamfer_loss

        # Mesh regularization
        if pred_vertices is not None and pred_faces is not None:
            mesh_reg = self.compute_mesh_regularization(pred_vertices, pred_faces)
            losses["mesh_reg"] = mesh_reg
            total_loss = total_loss + self.mesh_reg_weight * mesh_reg

        # Occupancy loss
        if pred_occupancy is not None and gt_occupancy is not None:
            occ_loss = self.compute_occupancy_loss(pred_occupancy, gt_occupancy)
            losses["occupancy"] = occ_loss
            total_loss = total_loss + self.occupancy_weight * occ_loss

        losses["total"] = total_loss
        return losses

    def _get_device(self, *tensors) -> torch.device:
        """Get device from first non-None tensor."""
        for t in tensors:
            if t is not None:
                return t.device
        return torch.device("cpu")


class MedicalTrainer:
    """
    Trainer for medical fine-tuning of SAM3D.

    Handles:
    - LoRA injection and parameter freezing
    - Training loop with gradient accumulation
    - Validation and checkpointing
    - Logging and metrics tracking
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        checkpoint_dir: str = "./checkpoints",
        grad_accum_steps: int = 1,
        mixed_precision: bool = True,
        device: str = "cuda",
        loss_config: dict | None = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.checkpoint_dir = Path(checkpoint_dir)
        self.grad_accum_steps = grad_accum_steps
        self.mixed_precision = mixed_precision

        # Setup checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup losses
        loss_config = loss_config or {}
        self.losses = MedicalTrainingLosses(**loss_config)

        # Setup LoRA
        self._setup_lora()

        # Setup optimizer (only for trainable params)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

        # Setup scheduler (will be configured per training run)
        self.scheduler = None

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler("cuda") if mixed_precision else None

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.history = {"train": [], "val": []}

    def _setup_lora(self) -> None:
        """Inject LoRA and freeze base parameters."""
        from sam3d_objects.model.lora import setup_lora_for_medical_finetuning

        param_counts = setup_lora_for_medical_finetuning(
            self.model,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=0.0,
            unfreeze_output_layers=True,
        )

        logger.info("LoRA setup complete:")
        logger.info(f"  Total params: {param_counts['total']:,}")
        logger.info(f"  Trainable params: {param_counts['trainable']:,}")
        logger.info(f"  LoRA params: {param_counts['lora']:,}")

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._to_device(batch)

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                outputs = self._forward_step(batch)
                # Debug: detect non-finite tensors early to find NaN sources
                self._detect_nonfinite_tensors(batch, outputs, batch_idx)
                losses = self._compute_losses(outputs, batch)
                loss = losses["total"] / self.grad_accum_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

            # Record losses
            epoch_losses.append({k: v.item() for k, v in losses.items()})

            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch {self.current_epoch} | "
                    f"Step {batch_idx + 1}/{len(self.train_loader)} | "
                    f"Loss: {losses['total'].item():.4f}"
                )

        # Aggregate epoch losses
        avg_losses = self._aggregate_losses(epoch_losses)
        self.history["train"].append(avg_losses)

        return avg_losses

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_losses = []

        for batch in self.val_loader:
            batch = self._to_device(batch)

            with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                outputs = self._forward_step(batch)
                losses = self._compute_losses(outputs, batch)

            val_losses.append({k: v.item() for k, v in losses.items()})

        avg_losses = self._aggregate_losses(val_losses)
        self.history["val"].append(avg_losses)

        return avg_losses

    def train(
        self,
        epochs: int,
        save_every: int = 5,
        validate_every: int = 1,
    ) -> None:
        """
        Full training loop.

        Args:
            epochs: Number of epochs
            save_every: Save checkpoint every N epochs
            validate_every: Validate every N epochs
        """
        # Setup scheduler
        total_steps = epochs * len(self.train_loader) // self.grad_accum_steps
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
        )

        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Total steps: {total_steps}")

        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Epoch {self.current_epoch}/{epochs}")
            logger.info(f"{'=' * 50}")

            # Train
            train_losses = self.train_epoch()
            logger.info(f"Train Loss: {train_losses['total']:.4f}")

            # Validate
            if self.val_loader is not None and epoch % validate_every == 0:
                val_losses = self.validate()
                logger.info(f"Val Loss: {val_losses['total']:.4f}")

                # Save best model
                if val_losses["total"] < self.best_val_loss:
                    self.best_val_loss = val_losses["total"]
                    self.save_checkpoint("best.pt")
                    logger.info("New best model saved!")

            # Periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

        # Final save
        self.save_checkpoint("final.pt")
        self._save_history()
        logger.info("Training complete!")

    def _forward_step(self, batch: dict) -> dict:
        """
        Forward pass through model.

        Override this method for custom model architectures.
        """
        # Extract inputs
        image = batch.get("image")
        pointmap = batch.get("pointmap")

        # Forward pass (model-specific)
        # This is a placeholder - actual implementation depends on model
        outputs = self.model(image, pointmap)

        return outputs

    def _compute_losses(
        self,
        outputs: dict,
        batch: dict,
    ) -> dict[str, torch.Tensor]:
        """
        Compute losses from model outputs and batch.

        Override this method for custom loss computation.
        """
        # Extract predictions
        pred_sdf = outputs.get("sdf")
        pred_vertices = outputs.get("vertices")
        pred_faces = outputs.get("faces")
        pred_occupancy = outputs.get("occupancy")

        # Extract ground truth - handle both 'sdf' and 'mask_sdf' keys
        gt_sdf = batch.get("sdf") or batch.get("mask_sdf")
        gt_vertices = batch.get("vertices")
        gt_occupancy = batch.get("mask") or batch.get("segmentation")

        return self.losses(
            pred_sdf=pred_sdf,
            gt_sdf=gt_sdf,
            pred_vertices=pred_vertices,
            gt_vertices=gt_vertices,
            pred_faces=pred_faces,
            pred_occupancy=pred_occupancy,
            gt_occupancy=gt_occupancy,
        )

    def _to_device(self, batch: dict) -> dict:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

    def _detect_nonfinite_tensors(self, batch: dict, outputs: dict, batch_idx: int) -> None:
        """Log any non-finite tensors found in the batch or outputs for debugging.

        This helper is enabled only when SANITY_DEBUG env var is set to '1'.
        """
        import os

        if os.environ.get("SANITY_DEBUG") != "1":
            return

        def _check_tensor(name, t):
            if t is None:
                return
            if not isinstance(t, torch.Tensor):
                return
            if not t.isfinite().all():
                nans = (~t.isfinite()).sum().item()
                logger.warning(f"Non-finite detected in {name} at batch {batch_idx}: count={nans}, shape={t.shape}")
            else:
                # Optionally log min/max for numeric issues
                try:
                    t_min = t.min().item()
                    t_max = t.max().item()
                except Exception:
                    t_min, t_max = None, None
                logger.debug(f"{name} finite: shape={t.shape}, min={t_min}, max={t_max}")

        # Check some expected keys
        for k in ("image", "pointmap", "mask_sdf", "segmentation"):
            if k in batch:
                _check_tensor(f"batch.{k}", batch[k])
        for k, v in outputs.items():
            _check_tensor(f"outputs.{k}", v)

    def _aggregate_losses(self, losses_list: list[dict]) -> dict[str, float]:
        """Aggregate list of loss dicts into means."""
        aggregated = {}
        for key in losses_list[0].keys():
            aggregated[key] = sum(d[key] for d in losses_list) / len(losses_list)
        return aggregated

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        from sam3d_objects.model.lora import get_lora_state_dict

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "lora_state_dict": get_lora_state_dict(self.model),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "lr": self.lr,
            },
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        from sam3d_objects.model.lora import load_lora_state_dict

        checkpoint = torch.load(path, map_location=self.device)

        load_lora_state_dict(self.model, checkpoint["lora_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        logger.info(f"Checkpoint loaded from {path}")

    def _save_history(self) -> None:
        """Save training history to JSON."""
        path = self.checkpoint_dir / "history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"History saved: {path}")


def create_dummy_model():
    """Create a dummy model for testing."""

    class DummyMeshModel(nn.Module):
        def __init__(self, channels: int = 64):
            super().__init__()
            self.embed = nn.Linear(3, channels)
            self.to_qkv = nn.Linear(channels, channels * 3)
            self.to_out = nn.Linear(channels, channels)
            self.sdf_head = nn.Linear(channels, 1)
            self.out_layer = nn.Linear(channels, channels)

        def forward(self, image, pointmap):
            # Dummy forward
            # pointmap comes in as (B, 3, H, W) - convert to (B, H, W, 3) for embed layer
            if pointmap.ndim == 4 and pointmap.shape[1] == 3:
                pointmap = pointmap.permute(0, 2, 3, 1).contiguous()
            x = self.embed(pointmap)  # (B, H, W, channels)
            x = self.to_out(self.to_qkv(x)[..., : x.shape[-1]])
            x = self.out_layer(x)
            sdf = self.sdf_head(x)  # (B, H, W, 1)
            # Rearrange to (B, 1, H, W) for compatibility with SDF losses
            sdf = sdf.permute(0, 3, 1, 2).contiguous()
            return {
                "sdf": sdf,
                "vertices": None,
                "faces": None,
                "occupancy": None,
            }

    return DummyMeshModel()


def create_model(name: str = "dummy", params: dict | None = None) -> nn.Module:
    """Factory to create models by name.

    Currently supports:
    - "dummy": returns the existing dummy model (for unit testing only)
    - "slat_mesh": returns a SLatMeshDecoder instance (for mesh training)

    IMPORTANT: No silent fallbacks! If dependencies are missing, this will fail loudly.
    """
    params = params or {}
    if name == "dummy":
        return create_dummy_model()
    if name == "slat_mesh":
        # Import the real model - fail loudly if dependencies are missing
        from sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_mesh import (
            SLatMeshDecoder,
        )

        # Default small model params for quick tests; allow override via params
        defaults = {
            "resolution": int(params.get("resolution", 4)),
            "model_channels": int(params.get("model_channels", 32)),
            "latent_channels": int(params.get("latent_channels", 16)),
            "num_blocks": int(params.get("num_blocks", 2)),
            "num_heads": int(params.get("num_heads", 4)),
            "use_fp16": bool(params.get("use_fp16", False)),
            "use_checkpoint": bool(params.get("use_checkpoint", False)),
            "device": params.get("device", "cuda"),
            "representation_config": params.get("representation_config", {"use_color": False}),
        }
        return SLatMeshDecoder(
            resolution=defaults["resolution"],
            model_channels=defaults["model_channels"],
            latent_channels=defaults["latent_channels"],
            num_blocks=defaults["num_blocks"],
            num_heads=defaults["num_heads"],
            use_fp16=defaults["use_fp16"],
            use_checkpoint=defaults["use_checkpoint"],
            device=defaults["device"],
            representation_config=defaults["representation_config"],
        )
    raise ValueError(f"Unknown model name: {name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Medical fine-tuning for SAM3D")

    # Data
    parser.add_argument("--data_root", type=str, required=True, help="Path to preprocessed data")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")

    # Training
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=8.0, help="LoRA alpha")

    # Loss weights
    parser.add_argument("--sdf_weight", type=float, default=1.0, help="SDF loss weight")
    parser.add_argument("--chamfer_weight", type=float, default=0.5, help="Chamfer loss weight")
    parser.add_argument("--mesh_reg_weight", type=float, default=0.1, help="Mesh reg weight")

    # Checkpointing
    parser.add_argument(
        "--checkpoint_dir", type=str, default="./checkpoints/medical", help="Checkpoint directory"
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--model_name", type=str, default="slat_mesh", help="Model to use (dummy|slat_mesh)")
    parser.add_argument(
        "--model_params",
        type=str,
        default=None,
        help="JSON string of model params to pass to model factory",
    )
    parser.add_argument("--slice_cache_dir", type=str, default=None, help="Slice cache dir override (defaults to data_root/slice_cache)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--augment", action="store_true", default=False, help="Enable per-slice augmentations")
    parser.add_argument("--no_mixed_precision", action="store_true", help="Disable mixed precision")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logger.info("Medical Fine-Tuning Script")
    logger.info(f"Args: {args}")

    # Create data loader - require real dataset, no dummy fallback
    if args.data_root is None:
        raise ValueError("--data_root is required. Provide path to preprocessed medical data.")
    
    logger.info("Creating TS_SAM3D_Dataset loader from data_root: %s", args.data_root)
    from sam3d_objects.data.dataset.ts.ts_sam3d_dataloader import (
        TS_SAM3D_Dataset,
        data_collate,
    )
    slice_cache_dir = args.slice_cache_dir or (str(args.data_root) + "/slice_cache")
    train_ds = TS_SAM3D_Dataset(
        original_nifti_dir=args.data_root,
        cache_slices=True,
        slice_cache_dir=slice_cache_dir,
        classes=1,
        augment=args.augment,
        augment_mode="train",
        occupancy_threshold=0.01,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collate,
        num_workers=args.num_workers,
    )

    # Create model
    logger.info("Creating model: %s", args.model_name)
    model_params = None
    if args.model_params:
        import json as _json

        model_params = _json.loads(args.model_params)
    model = create_model(name=args.model_name, params=model_params)

    # Create trainer
    trainer = MedicalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        checkpoint_dir=args.checkpoint_dir,
        grad_accum_steps=args.grad_accum,
        mixed_precision=not args.no_mixed_precision,
        device=args.device,
        loss_config={
            "sdf_weight": args.sdf_weight,
            "chamfer_weight": args.chamfer_weight,
            "mesh_reg_weight": args.mesh_reg_weight,
        },
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(
        epochs=args.epochs,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()


def train_from_config(cfg: dict):
    """Run training using a configuration dictionary (e.g., from Hydra or OmegaConf).

    Expected config structured similarly to the CLI args used in `main()`.
    """
    class Args:
        pass

    args = Args()
    # Load common training params
    args.batch_size = int(cfg.get("training", {}).get("batch_size", 4))
    args.epochs = int(cfg.get("training", {}).get("epochs", 50))
    args.lr = float(cfg.get("training", {}).get("lr", 1e-3))
    args.weight_decay = float(cfg.get("training", {}).get("weight_decay", 0.01))
    args.grad_accum = int(cfg.get("training", {}).get("grad_accum", 1))
    args.num_workers = int(cfg.get("training", {}).get("num_workers", 4))
    args.no_mixed_precision = not cfg.get("training", {}).get("mixed_precision", True)

    # LoRA
    args.lora_rank = int(cfg.get("lora", {}).get("rank", 4))
    args.lora_alpha = float(cfg.get("lora", {}).get("alpha", 8.0))

    # Checkpointing
    args.checkpoint_dir = cfg.get("checkpoint", {}).get("dir", "./checkpoints/medical")
    args.resume = cfg.get("checkpoint", {}).get("resume", None)
    args.save_every = int(cfg.get("checkpoint", {}).get("save_every", 5))

    # Data - require real dataset
    args.data_root = cfg.get("data", {}).get("data_root")
    args.slice_cache_dir = cfg.get("data", {}).get("slice_cache_dir")
    args.augment = bool(cfg.get("data", {}).get("augment", False))

    # Device
    args.device = cfg.get("device", "cuda")

    # Build train_loader - require real dataset, no dummy fallback
    if args.data_root is None:
        raise ValueError("data.data_root is required. Provide path to preprocessed medical data.")
    
    logger.info("(Hydra) Creating TS_SAM3D_Dataset loader from data_root: %s", args.data_root)
    from sam3d_objects.data.dataset.ts.ts_sam3d_dataloader import (
        TS_SAM3D_Dataset,
        data_collate,
    )
    slice_cache_dir = args.slice_cache_dir or (str(args.data_root) + "/slice_cache")
    train_ds = TS_SAM3D_Dataset(
        original_nifti_dir=args.data_root,
        cache_slices=True,
        slice_cache_dir=slice_cache_dir,
        classes=1,
        augment=args.augment,
        augment_mode="train",
        occupancy_threshold=0.01,
    )
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collate,
        num_workers=args.num_workers,
    )

    # Create model & trainer
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "slat_mesh")
    model_params = model_cfg.get("params", {})
    model = create_model(name=model_name, params=model_params)
    trainer = MedicalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        checkpoint_dir=args.checkpoint_dir,
        grad_accum_steps=args.grad_accum,
        mixed_precision=not args.no_mixed_precision,
        device=args.device,
        loss_config={
            "sdf_weight": 1.0,
            "chamfer_weight": 0.5,
            "mesh_reg_weight": 0.1,
        },
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(epochs=args.epochs, save_every=args.save_every, validate_every=1)
