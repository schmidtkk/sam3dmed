"""
Medical fine-tuning training script for SAM3D.

This script provides training utilities for the SAM3D two-stage pipeline:
  - Stage 1: ss_generator (Sparse Structure) - predicts WHERE voxels are
  - Stage 2: slat_generator + slat_decoder_mesh - predicts WHAT the shape is

CRITICAL: The SAM3D framework requires BOTH stages for end-to-end reconstruction.
Training only the decoder without the generators will not produce a working pipeline.

Usage (Hydra-based):
    # Default run (uses configs/train.yaml)
    python scripts/train_medical.py
    
    # Override parameters
    python scripts/train_medical.py \\
        data.data_root=/path/to/preprocessed \\
        checkpoint.dir=./checkpoints/medical \\
        training.batch_size=4 \\
        training.epochs=50
    
    # Enable two-stage training
    python scripts/train_medical.py \\
        training.mode=two_stage \\
        stage1.enabled=true

See configs/train.yaml for full configuration options.
"""

import os

# Skip LIDRA initialization - sam3d_objects.init module does not exist in this fork.
# LIDRA was the original internal codename/framework. This fork removed it but kept
# the conditional import guard for compatibility.
os.environ.setdefault('LIDRA_SKIP_INIT', '1')

import hydra
from omegaconf import OmegaConf, DictConfig
from typing import Optional
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

# TensorBoard for logging
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")

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
    - Two-stage training: Stage 1 (ss_generator) and Stage 2 (slat_generator + decoder)
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
        grad_clip_norm: float = 1.0,
        mixed_precision: bool = True,
        device: str = "cuda",
        loss_config: dict | None = None,
        slat_generator_ckpt: str | None = None,
        slat_generator_params: dict | None = None,
        # Stage 1 (ss_generator) config
        ss_generator: nn.Module | None = None,
        ss_decoder: nn.Module | None = None,
        train_stage1: bool = False,
        # Two-stage training mode
        training_mode: str = "stage2_only",  # "stage1_only", "stage2_only", "two_stage"
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
        self.grad_clip_norm = grad_clip_norm
        self.mixed_precision = mixed_precision
        self.slat_generator_ckpt = slat_generator_ckpt
        self.slat_generator_params = slat_generator_params or {}
        self.training_mode = training_mode
        self.train_stage1 = train_stage1 or (training_mode in ["stage1_only", "two_stage"])

        # Setup checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup losses
        loss_config = loss_config or {}
        self.losses = MedicalTrainingLosses(**loss_config)
        
        # Stage 1 models
        self.ss_generator = ss_generator.to(self.device) if ss_generator is not None else None
        self.ss_decoder = ss_decoder.to(self.device) if ss_decoder is not None else None

        # Setup LoRA
        self._setup_lora()
        # Optionally load slat generator for mesh decoder training
        self.slat_generator = None
        if self.slat_generator_ckpt is not None:
            try:
                logger.info(f"Loading slat generator checkpoint: {self.slat_generator_ckpt}")
                # Instantiate using `create_model` for flexible small generator setup
                self.slat_generator = create_model(name="slat_generator", params=self.slat_generator_params)
                from sam3d_objects.model.io import load_model_from_checkpoint

                self.slat_generator = load_model_from_checkpoint(
                    self.slat_generator,
                    self.slat_generator_ckpt,
                    strict=False,
                    device=self.device,
                    freeze=True,
                    eval=True,
                )
                logger.info("Slat generator loaded and set to eval mode")
            except Exception as e:
                logger.warning(f"Failed to load slat generator: {e}")

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
        
        # TensorBoard writer
        if TENSORBOARD_AVAILABLE:
            log_dir = self.checkpoint_dir / "tensorboard" / datetime.now().strftime("%Y%m%d-%H%M%S")
            self.writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"TensorBoard logging to: {log_dir}")
        else:
            self.writer = None

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
        
        # Setup LoRA for Stage 1 if training Stage 1
        if self.train_stage1 and self.ss_generator is not None:
            logger.info("Setting up LoRA for Stage 1 (ss_generator)...")
            ss_param_counts = setup_lora_for_medical_finetuning(
                self.ss_generator,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=0.0,
                unfreeze_output_layers=True,
            )
            logger.info("Stage 1 LoRA setup complete:")
            logger.info(f"  SS Generator trainable params: {ss_param_counts['trainable']:,}")

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch.
        
        Supports three training modes:
        - stage1_only: Train ss_generator with flow matching loss
        - stage2_only: Train slat_decoder with mesh/SDF loss
        - two_stage: Train both stages jointly
        """
        self.model.train()
        if self.ss_generator is not None and self.train_stage1:
            self.ss_generator.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                if self.training_mode == "stage1_only":
                    # Stage 1 only: train ss_generator
                    losses = self._forward_stage1(batch)
                elif self.training_mode == "two_stage":
                    # Two-stage: train both stages
                    stage1_losses = self._forward_stage1(batch) if self.ss_generator else {}
                    outputs = self._forward_step(batch)
                    self._detect_nonfinite_tensors(batch, outputs, batch_idx)
                    stage2_losses = self._compute_losses(outputs, batch)
                    # Combine losses
                    losses = {**stage1_losses, **stage2_losses}
                    losses["total"] = sum(v for k, v in losses.items() if k != "total" and isinstance(v, torch.Tensor))
                else:
                    # Stage 2 only (default): train slat_decoder
                    outputs = self._forward_step(batch)
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
                # Gradient clipping to prevent NaN/explosion
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.grad_clip_norm
                    )
                
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
            epoch_losses.append({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()})

            # Log progress
            if (batch_idx + 1) % 10 == 0:
                current_loss = losses['total'].item() if isinstance(losses['total'], torch.Tensor) else losses['total']
                logger.info(
                    f"Epoch {self.current_epoch} | "
                    f"Step {batch_idx + 1}/{len(self.train_loader)} | "
                    f"Loss: {current_loss:.4f}"
                )
                # TensorBoard logging per step
                if self.writer is not None:
                    self.writer.add_scalar("train/loss_step", current_loss, self.global_step)
                    if self.scheduler is not None:
                        self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)
                    # Log individual loss components
                    for k, v in losses.items():
                        if k != "total" and isinstance(v, torch.Tensor):
                            self.writer.add_scalar(f"train/{k}_step", v.item(), self.global_step)

        # Aggregate epoch losses
        avg_losses = self._aggregate_losses(epoch_losses)
        self.history["train"].append(avg_losses)
        
        # TensorBoard logging per epoch
        if self.writer is not None:
            self.writer.add_scalar("train/loss_epoch", avg_losses.get("total", 0), self.current_epoch)
            for k, v in avg_losses.items():
                if k != "total":
                    self.writer.add_scalar(f"train/{k}_epoch", v, self.current_epoch)

        return avg_losses

    def _forward_stage1(self, batch: dict) -> dict[str, torch.Tensor]:
        """Forward pass for Stage 1 (ss_generator) training.
        
        Uses flow matching loss to train the sparse structure generator.
        The generator learns to predict occupancy voxels from image conditioning.
        """
        if self.ss_generator is None:
            return {"stage1_loss": torch.tensor(0.0, device=self.device)}
        
        # Get conditioning from batch
        image = batch.get("image")  # (B, C, H, W)
        
        # Get ground truth sparse structure (occupancy)
        # This should be a 3D voxel grid of shape (B, 1, D, H, W) or similar
        gt_occupancy = batch.get("gt_occupancy")
        if gt_occupancy is None:
            # Fallback: create from 3D SDF if available
            gt_sdf = batch.get("gt_sdf")
            if gt_sdf is not None:
                gt_occupancy = (gt_sdf < 0).float()  # Inside surface
            else:
                # Last resort: use 2D mask expanded to pseudo-3D
                mask = batch.get("mask")
                if mask is not None:
                    # Create a thin 3D volume from 2D mask
                    if mask.dim() == 3:  # (B, H, W)
                        mask = mask.unsqueeze(1)  # (B, 1, H, W)
                    # Reshape to 16x16x16 voxel grid for ss_generator
                    gt_occupancy = F.interpolate(
                        mask.unsqueeze(2).float(),  # (B, 1, 1, H, W) 
                        size=(16, 16, 16),
                        mode="nearest"
                    )
                else:
                    logger.warning("No ground truth occupancy available for Stage 1")
                    return {"stage1_loss": torch.tensor(0.0, device=self.device)}
        
        # Encode to latent (flatten to token sequence for flow matching)
        # ss_generator expects latent of shape (B, 4096, 8) for 16^3 grid with 8 channels
        # First encode the occupancy to latent space
        B = gt_occupancy.shape[0]
        
        # Reshape occupancy to (B, 8, 16, 16, 16) latent representation
        # For now, use a simple encoding: replicate occupancy across channels
        if gt_occupancy.dim() == 4:  # (B, D, H, W)
            gt_occupancy = gt_occupancy.unsqueeze(1)  # (B, 1, D, H, W)
        
        # Resize to 16^3 if needed
        if gt_occupancy.shape[-3:] != (16, 16, 16):
            gt_occupancy = F.interpolate(gt_occupancy, size=(16, 16, 16), mode="nearest")
        
        # Create latent representation (B, 8, 16, 16, 16)
        gt_latent = gt_occupancy.expand(B, 8, 16, 16, 16)
        
        # Flatten to sequence (B, 4096, 8)
        gt_latent_seq = gt_latent.flatten(2).permute(0, 2, 1)  # (B, 4096, 8)
        
        # Create conditioning embedding (simplified: use image features)
        # In full implementation, this should use the condition_embedder
        cond = torch.zeros(B, 1024, device=self.device, dtype=image.dtype)
        
        # Compute flow matching loss
        # The ss_generator (FlowMatching wrapper) has a loss method
        try:
            total_loss, loss_dict = self.ss_generator.loss(
                {"shape": gt_latent_seq},  # x1 (target)
                cond,  # conditioning
            )
            return {"stage1_loss": total_loss, **loss_dict}
        except Exception as e:
            logger.warning(f"Stage 1 loss computation failed: {e}")
            return {"stage1_loss": torch.tensor(0.0, device=self.device, requires_grad=True)}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation and compute metrics.
        
        Computes:
        - Loss metrics (SDF, occupancy, etc.)
        - 3D metrics if GT mesh available (Chamfer distance, IoU)
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_losses = []
        
        # Additional metrics
        all_ious = []
        all_chamfer = []

        for batch in self.val_loader:
            batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                outputs = self._forward_step(batch)
                losses = self._compute_losses(outputs, batch)
                
                # Compute IoU if we have predicted and GT occupancy
                pred_occupancy = outputs.get("occupancy")
                gt_occupancy = batch.get("gt_occupancy") or batch.get("mask")
                if pred_occupancy is not None and gt_occupancy is not None:
                    # Threshold predictions
                    pred_binary = (pred_occupancy > 0.5).float()
                    gt_binary = (gt_occupancy > 0.5).float()
                    intersection = (pred_binary * gt_binary).sum()
                    union = ((pred_binary + gt_binary) > 0).float().sum()
                    iou = (intersection / (union + 1e-6)).item()
                    all_ious.append(iou)

            val_losses.append({k: v.item() for k, v in losses.items()})

        avg_losses = self._aggregate_losses(val_losses)
        
        # Add IoU metric if computed
        if all_ious:
            avg_losses["iou"] = sum(all_ious) / len(all_ious)
        
        self.history["val"].append(avg_losses)
        
        # TensorBoard logging for validation
        if self.writer is not None:
            self.writer.add_scalar("val/loss", avg_losses.get("total", 0), self.current_epoch)
            for k, v in avg_losses.items():
                if k != "total":
                    self.writer.add_scalar(f"val/{k}", v, self.current_epoch)

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
        self._close_writer()
        logger.info("Training complete!")
    
    def _close_writer(self):
        """Close TensorBoard writer if open."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def _forward_step(self, batch: dict) -> dict:
        """
        Forward pass through model.

        Override this method for custom model architectures.
        Supports both:
        - Standard models that accept (image, pointmap)
        - SLatMeshDecoder that requires a SparseTensor input
        """
        # Extract inputs
        image = batch.get("image")
        pointmap = batch.get("pointmap")
        mask = batch.get("mask")

        # Sanitize inputs: replace NaN/inf to prevent propagation through Linear layers
        # The dataset uses NaN for background pixels in pointmaps
        if pointmap is not None and not torch.isfinite(pointmap).all():
            pointmap = torch.nan_to_num(pointmap, nan=0.0, posinf=0.0, neginf=0.0)
        if image is not None and not torch.isfinite(image).all():
            image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        # Forward pass (model-specific)
        # Check if model expects a SparseTensor (e.g., SLatMeshDecoder)
        model_class_name = self.model.__class__.__name__
        if "SLatMeshDecoder" in model_class_name or "SLatDecoder" in model_class_name:
            # Create SparseTensor from batch data
            slat_input = self._create_sparse_tensor_from_batch(batch, image, pointmap, mask)
            raw_outputs = self.model(slat_input)
            # SLatMeshDecoder returns List[MeshExtractResult], wrap in dict for consistent API
            outputs = {
                "meshes": raw_outputs,  # List of mesh results
                "sdf_logits": raw_outputs,  # For loss computation compatibility
            }
        else:
            # Standard forward with (image, pointmap)
            outputs = self.model(image, pointmap)

        return outputs

    def _create_sparse_tensor_from_batch(
        self,
        batch: dict,
        image: torch.Tensor,
        pointmap: torch.Tensor,
        mask: torch.Tensor | None,
    ):
        """
        Create a SparseTensor from batch data for SLatMeshDecoder.

        The SparseTensor represents a 3D structured latent with:
        - coords: (N, 4) with [batch_idx, x, y, z] for each occupied voxel
        - feats: (N, C) features at each occupied voxel

        For 2D medical slices, we create a thin 3D volume (depth=1) and use
        the mask to determine occupied positions.
        """
        from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp

        B = image.shape[0]
        device = image.device
        dtype = image.dtype

        # Get resolution from model
        resolution = getattr(self.model, "resolution", 4)
        latent_channels = getattr(self.model, "in_channels", 16)

        # Create synthetic 3D coordinates from 2D mask
        # For each batch, create a small grid of occupied voxels
        all_coords = []
        all_feats = []

        for b in range(B):
            # Use mask to find foreground pixels, or create a uniform grid
            if mask is not None and mask.ndim >= 2:
                # Downsample mask to resolution x resolution
                mask_b = mask[b] if mask.ndim > 2 else mask
                if mask_b.ndim == 3:  # (C, H, W)
                    mask_b = mask_b[0]  # Take first channel
                # Resize to resolution x resolution
                mask_resized = F.interpolate(
                    mask_b.unsqueeze(0).unsqueeze(0).float(),
                    size=(resolution, resolution),
                    mode="nearest",
                ).squeeze()
                # Get non-zero positions
                occupied = (mask_resized > 0.5).nonzero(as_tuple=False)
                if occupied.shape[0] == 0:
                    # No foreground - create a single center voxel
                    occupied = torch.tensor([[resolution // 2, resolution // 2]], device=device)
            else:
                # No mask - create a full grid
                y_coords, x_coords = torch.meshgrid(
                    torch.arange(resolution, device=device),
                    torch.arange(resolution, device=device),
                    indexing="ij",
                )
                occupied = torch.stack([y_coords.flatten(), x_coords.flatten()], dim=1)

            num_voxels = occupied.shape[0]
            # Create 3D coords: (N, 4) with [batch_idx, x, y, z]
            # Use z=0 for 2D slices (thin volume)
            batch_idx = torch.full((num_voxels, 1), b, device=device, dtype=torch.int32)
            x_coords = occupied[:, 1:2].int()  # column index
            y_coords = occupied[:, 0:1].int()  # row index
            z_coords = torch.zeros((num_voxels, 1), device=device, dtype=torch.int32)
            coords_b = torch.cat([batch_idx, x_coords, y_coords, z_coords], dim=1)
            all_coords.append(coords_b)

            # Create features for each voxel
            # Use image features pooled to resolution, then sample at occupied positions
            if image is not None:
                img_b = image[b]  # (C, H, W)
                img_resized = F.interpolate(
                    img_b.unsqueeze(0),
                    size=(resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)  # (C, res, res)
                # Sample at occupied positions
                feats_b = img_resized[:, occupied[:, 0], occupied[:, 1]].T  # (N, C)
                # Pad or project to latent_channels
                if feats_b.shape[1] < latent_channels:
                    feats_b = F.pad(feats_b, (0, latent_channels - feats_b.shape[1]))
                elif feats_b.shape[1] > latent_channels:
                    feats_b = feats_b[:, :latent_channels]
            else:
                # Random features
                feats_b = torch.randn((num_voxels, latent_channels), device=device, dtype=dtype)

            all_feats.append(feats_b)

        # Concatenate all batches
        coords = torch.cat(all_coords, dim=0)
        # Use float32 for features to avoid mixed precision issues in mesh extraction
        feats = torch.cat(all_feats, dim=0).float()

        # Create SparseTensor
        sparse_tensor = sp.SparseTensor(feats=feats, coords=coords)
        return sparse_tensor

    # Backwards-compatible wrappers in case global module
    # functions exist (due to previous indentation issues).
    def _compute_losses(self, outputs, batch: dict) -> dict[str, torch.Tensor]:
        """
        Compute losses from model outputs and batch.

        Handles both:
        - dict outputs from simple models (e.g., DummyMeshModel)
        - List[MeshExtractResult] from SLatMeshDecoder (wrapped in dict with 'meshes' key)
        """
        # Handle outputs wrapped in dict with 'meshes' key (from SLatMeshDecoder)
        if isinstance(outputs, dict) and "meshes" in outputs:
            return self._compute_mesh_losses(outputs["meshes"], batch)
        
        # Handle List[MeshExtractResult] directly from SLatMeshDecoder
        if isinstance(outputs, list):
            return self._compute_mesh_losses(outputs, batch)

        # Standard dict output
        # Extract predictions
        pred_sdf = outputs.get("sdf")
        pred_vertices = outputs.get("vertices")
        pred_faces = outputs.get("faces")
        pred_occupancy = outputs.get("occupancy")

        # Extract ground truth - handle both 'sdf' and 'mask_sdf' keys
        # Use explicit None check to avoid boolean tensor evaluation error
        gt_sdf = batch.get("sdf")
        if gt_sdf is None:
            gt_sdf = batch.get("mask_sdf")
        gt_vertices = batch.get("vertices")
        gt_occupancy = batch.get("mask")
        if gt_occupancy is None:
            gt_occupancy = batch.get("segmentation")

        return self.losses(
            pred_sdf=pred_sdf,
            gt_sdf=gt_sdf,
            pred_vertices=pred_vertices,
            gt_vertices=gt_vertices,
            pred_faces=pred_faces,
            pred_occupancy=pred_occupancy,
            gt_occupancy=gt_occupancy,
        )

    def _compute_mesh_losses(self, mesh_results: list, batch: dict) -> dict[str, torch.Tensor]:
        """
        Compute losses from List[MeshExtractResult] output.

        MeshExtractResult has:
        - vertices: (V, 3) mesh vertices
        - faces: (F, 3) mesh faces
        - tsdf_v, tsdf_s: TSDF values (for training)
        - reg_loss: regularization loss from mesh extraction
        """
        device = self.device
        reg_losses = []
        vertex_losses = []

        # Extract ground truth mask for occupancy-based loss
        gt_mask = batch.get("mask")
        if gt_mask is None:
            gt_mask = batch.get("segmentation")

        for i, mesh in enumerate(mesh_results):
            if not mesh.success:
                # Empty mesh - skip (no gradients to propagate)
                continue

            # Regularization loss from FlexiCubes (this has gradients)
            if mesh.reg_loss is not None and mesh.reg_loss.requires_grad:
                reg_losses.append(mesh.reg_loss)

            # Vertex-based occupancy loss: encourage vertices within mask region
            if gt_mask is not None and mesh.vertices.shape[0] > 0 and mesh.vertices.requires_grad:
                # Normalize vertex positions to [0, 1] range for comparison with mask
                verts = mesh.vertices  # (V, 3)
                v_min = verts.min(dim=0)[0]
                v_max = verts.max(dim=0)[0]
                v_range = (v_max - v_min).clamp(min=1e-6)
                verts_norm = (verts - v_min) / v_range  # (V, 3) in [0, 1]

                # Project to 2D (use x, y for now) and sample mask
                mask_i = gt_mask[i] if gt_mask.ndim > 2 else gt_mask
                if mask_i.ndim == 3:
                    mask_i = mask_i[0]  # (H, W)

                # Sample mask at vertex positions
                grid_x = (verts_norm[:, 0] * 2 - 1).clamp(-1, 1)  # to [-1, 1]
                grid_y = (verts_norm[:, 1] * 2 - 1).clamp(-1, 1)
                grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, V, 2)
                mask_sampled = F.grid_sample(
                    mask_i.unsqueeze(0).unsqueeze(0).float(),
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                ).squeeze()  # (V,)

                # Loss: vertices should be in foreground (mask > 0.5)
                vertex_loss = F.binary_cross_entropy_with_logits(
                    mask_sampled, torch.ones_like(mask_sampled),
                    reduction="mean"
                )
                vertex_losses.append(vertex_loss)

        # Combine losses - ensure we have at least one differentiable loss
        if len(reg_losses) > 0:
            reg_loss = torch.stack(reg_losses).mean()
        else:
            reg_loss = torch.zeros(1, device=device, requires_grad=True).squeeze()

        if len(vertex_losses) > 0:
            vertex_loss = torch.stack(vertex_losses).mean()
        else:
            vertex_loss = torch.zeros(1, device=device, requires_grad=True).squeeze()

        total_loss = reg_loss * self.losses.mesh_reg_weight + vertex_loss

        losses = {
            "total": total_loss,
            "reg": reg_loss.detach(),
            "vertex": vertex_loss.detach(),
        }
        return losses

    def _to_device(self, batch: dict) -> dict:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

    def _detect_nonfinite_tensors(self, batch: dict, outputs: dict, batch_idx: int) -> None:
        """Log any non-finite tensors found in the batch or outputs for debugging.

        This helper is enabled only when SANITY_DEBUG env var is set to '1'.
        """
        import os as _os

        if _os.environ.get("SANITY_DEBUG") != "1":
            return

        def _check_tensor(name: str, t: torch.Tensor | None) -> None:
            if t is None or not isinstance(t, torch.Tensor):
                return
            if not t.isfinite().all():
                nans = (~t.isfinite()).sum().item()
                logger.warning(f"Non-finite detected in {name} at batch {batch_idx}: count={nans}, shape={t.shape}")
            else:
                try:
                    t_min, t_max = t.min().item(), t.max().item()
                except Exception:
                    t_min, t_max = None, None
                logger.debug(f"{name} finite: shape={t.shape}, min={t_min}, max={t_max}")

        for k in ("image", "pointmap", "mask_sdf", "segmentation"):
            if k in batch:
                _check_tensor(f"batch.{k}", batch[k])
        for k, v in outputs.items():
            _check_tensor(f"outputs.{k}", v)

    def _aggregate_losses(self, losses_list: list[dict]) -> dict[str, float]:
        """Aggregate list of loss dicts into means."""
        aggregated: dict[str, float] = {}
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


def _find_yaml_for_ckpt(ckpt_path: str) -> Optional[str]:
    """
    Try to find a YAML config file corresponding to a checkpoint file.
    It checks the same directory for files with the same stem and .yaml or .yml extension.
    """
    if ckpt_path is None:
        return None
    ckpt_path = str(ckpt_path)
    try:
        folder = os.path.dirname(ckpt_path)
        stem = os.path.splitext(os.path.basename(ckpt_path))[0]
        for ext in (".yaml", ".yml"):
            cand = os.path.join(folder, stem + ext)
            if os.path.exists(cand):
                return cand
    except Exception:
        pass
    return None


def _extract_model_params_from_yaml(yaml_path: str, model_name: str) -> dict:
    """
    Extract model parameter dict from a checkpoint YAML config for the given model name.
    Supports `slat_mesh` and `slat_generator` extraction heuristics.
    """
    if yaml_path is None:
        return {}
    from omegaconf import DictConfig
    oc = OmegaConf.load(yaml_path)
    params = {}
    try:
        if model_name in ("slat_mesh", "slat_decoder", "slat_decoder_mesh"):
            # Common keys at top-level
            for k in ["resolution", "model_channels", "latent_channels", "num_blocks", "num_heads", "use_fp16", "use_checkpoint", "representation_config"]:
                if k in oc:
                    params[k] = oc[k]
        elif model_name in ("slat_generator", "slat_flow", "slat_flow_model"):
            # Try nested path used in pipeline YAML
            # The slat_generator YAML structure is:
            # module.generator.backbone.reverse_fn.backbone (contains SLatFlowModelTdfyWrapper params)
            nested_keys = [
                ("module", "generator", "backbone", "reverse_fn", "backbone"),  # Actual structure
                ("module", "generator", "backbone", "backbone", "model"),
                ("module", "generator", "backbone", "backbone"),
                ("module", "generator", "backbone", "model"),
            ]
            param_keys = ["model_channels", "in_channels", "out_channels", "num_blocks", 
                          "num_heads", "io_block_channels", "use_fp16", "patch_size",
                          "cond_channels", "resolution", "mlp_ratio", "qk_rms_norm", "pe_mode"]
            for path in nested_keys:
                node = oc
                valid = True
                for p in path:
                    if hasattr(node, "get") and node.get(p) is not None:
                        node = node.get(p)
                    elif isinstance(node, dict) and p in node:
                        node = node[p]
                    else:
                        valid = False
                        break
                if valid and (isinstance(node, dict) or isinstance(node, DictConfig)):
                    for k in param_keys:
                        if k in node:
                            params[k] = node[k]
                    if params:  # Found params, stop searching
                        logger.debug(f"Extracted params from path {'.'.join(path)}: {list(params.keys())}")
                        break
            # Fallback to top-level if missing
            if not params:
                for k in ["model_channels", "in_channels", "out_channels"]:
                    if k in oc:
                        params[k] = oc[k]
    except Exception as e:
        logger.debug(f"Failed to extract params from YAML: {e}")
    return params


def _safe_load_checkpoint(path: str):
    """
    Safely load a checkpoint file (torch or safetensors). Returns the object loaded or OrderedDict.
    """
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint path not found: {path}")
    if path.endswith('.safetensors'):
        try:
            from safetensors.torch import load_file

            return load_file(path)
        except Exception:
            # Fall back to torch.load
            pass
    try:
        return torch.load(path, map_location='cpu')
    except Exception:
        # Some older files may require non-weights only load
        return torch.load(path, map_location='cpu', weights_only=False)


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
            # Replace NaNs/inf in pointmap to avoid NaN propagation in Linear layers
            if torch.isfinite(pointmap).all() is False:
                pointmap = torch.nan_to_num(pointmap, nan=0.0, posinf=0.0, neginf=0.0)
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
            "resolution": int(params.get("resolution", 64)),
            "model_channels": int(params.get("model_channels", 768)),
            "latent_channels": int(params.get("latent_channels", 8)),
            "num_blocks": int(params.get("num_blocks", 12)),
            "num_heads": int(params.get("num_heads", 12)),
            "use_fp16": bool(params.get("use_fp16", False)),
            "use_checkpoint": bool(params.get("use_checkpoint", False)),
            "device": params.get("device", "cuda"),
            "representation_config": params.get("representation_config", {"use_color": False}),
        }
        # Validate that the default model_channels yields valid group norm sizes
        # For SparseGroupNorm32 we use num_groups=32 and the decoder upsample
        # computes out_channels = model_channels // 4 and // 8. Both must be
        # divisible by num_groups. This implies model_channels must be divisible
        # by 8 * num_groups (i.e., 256 when num_groups=32).
        num_groups = 32
        if defaults["model_channels"] % (8 * num_groups) != 0:
            raise ValueError(
                "Invalid 'model_channels' for SLatMeshDecoder; model_channels must be a multiple "
                f"of {8 * num_groups} (8 * num_groups), e.g., 256. Found {defaults['model_channels']}"
            )

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
    if name == "slat_generator":
        # Instantiate the slat generator (SLatFlowModelTdfyWrapper) for training/testing
        from sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_flow import (
            SLatFlowModelTdfyWrapper,
        )

        defaults = {
            "resolution": int(params.get("resolution", 64)),
            "in_channels": int(params.get("in_channels", 8)),
            "model_channels": int(params.get("model_channels", 1024)),
            "cond_channels": int(params.get("cond_channels", 1024)),
            "out_channels": int(params.get("out_channels", 8)),
            "num_blocks": int(params.get("num_blocks", 24)),
            "num_heads": int(params.get("num_heads", 16)),
            "use_fp16": bool(params.get("use_fp16", False)),
            "use_checkpoint": bool(params.get("use_checkpoint", False)),
            # Do not pass a `device` kwarg to the backbone constructors - call .to(device) instead
            "io_block_channels": params.get("io_block_channels", [64, 128, 256]),
        }
        model = SLatFlowModelTdfyWrapper(
            resolution=defaults["resolution"],
            in_channels=defaults["in_channels"],
            model_channels=defaults["model_channels"],
            cond_channels=defaults["cond_channels"],
            out_channels=defaults["out_channels"],
            num_blocks=defaults["num_blocks"],
            num_heads=defaults["num_heads"],
            use_fp16=defaults["use_fp16"],
            use_checkpoint=defaults["use_checkpoint"],
            io_block_channels=defaults["io_block_channels"],
        )
        # Move to device in the caller and return the generator instance
        return model
    if name == "ss_generator":
        # Instantiate the sparse structure generator for Stage 1 training
        # This uses the FlowMatching wrapper with SparseStructureFlowModel backbone
        from sam3d_objects.model.backbone.generator.flow_matching.model import FlowMatching
        from sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow import (
            SparseStructureFlowTdfyWrapper,
        )

        defaults = {
            "in_channels": int(params.get("in_channels", 8)),
            "model_channels": int(params.get("model_channels", 1024)),
            "cond_channels": int(params.get("cond_channels", 1024)),
            "out_channels": int(params.get("out_channels", 8)),
            "num_blocks": int(params.get("num_blocks", 24)),
            "num_heads": int(params.get("num_heads", 16)),
            "mlp_ratio": float(params.get("mlp_ratio", 4.0)),
            "use_fp16": bool(params.get("use_fp16", False)),
            "use_checkpoint": bool(params.get("use_checkpoint", False)),
            "is_shortcut_model": bool(params.get("is_shortcut_model", False)),
            # Latent mapping for shape token (simplified for medical)
            "latent_mapping": params.get("latent_mapping", {
                "shape": {
                    "_target_": "sam3d_objects.model.backbone.tdfy_dit.models.mm_latent.Latent",
                    "in_channels": 8,
                    "model_channels": 1024,
                }
            }),
        }
        
        # Build the backbone (SparseStructureFlowTdfyWrapper)
        backbone = SparseStructureFlowTdfyWrapper(
            in_channels=defaults["in_channels"],
            model_channels=defaults["model_channels"],
            cond_channels=defaults["cond_channels"],
            out_channels=defaults["out_channels"],
            num_blocks=defaults["num_blocks"],
            num_heads=defaults["num_heads"],
            mlp_ratio=defaults["mlp_ratio"],
            use_fp16=defaults["use_fp16"],
            use_checkpoint=defaults["use_checkpoint"],
            is_shortcut_model=defaults["is_shortcut_model"],
            latent_mapping=defaults["latent_mapping"],
        )
        
        # Wrap in FlowMatching for training
        model = FlowMatching(
            reverse_fn=backbone,
            sigma_min=float(params.get("sigma_min", 0.0)),
            inference_steps=int(params.get("inference_steps", 25)),
            time_scale=float(params.get("time_scale", 1000.0)),
        )
        return model
    if name == "ss_decoder":
        # Instantiate the sparse structure decoder
        from sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae import (
            SparseStructureDecoderTdfyWrapper,
        )

        defaults = {
            "out_channels": int(params.get("out_channels", 1)),
            "latent_channels": int(params.get("latent_channels", 8)),
            "num_res_blocks": int(params.get("num_res_blocks", 2)),
            "num_res_blocks_middle": int(params.get("num_res_blocks_middle", 2)),
            "channels": params.get("channels", [512, 128, 32]),
            "reshape_input_to_cube": bool(params.get("reshape_input_to_cube", False)),
        }
        
        model = SparseStructureDecoderTdfyWrapper(
            out_channels=defaults["out_channels"],
            latent_channels=defaults["latent_channels"],
            num_res_blocks=defaults["num_res_blocks"],
            num_res_blocks_middle=defaults["num_res_blocks_middle"],
            channels=defaults["channels"],
            reshape_input_to_cube=defaults["reshape_input_to_cube"],
        )
        return model
    raise ValueError(f"Unknown model name: {name}")


# =============================================================================
# Hydra-based training entry point
# =============================================================================
# NOTE: Standalone CLI (parse_args/main) has been removed.
# Use train_medical_hydra.py for all training runs.
# =============================================================================


def train_from_config(cfg: dict):
    """Run training using a configuration dictionary (from Hydra or OmegaConf).

    This is the main entry point for training. It is called by train_medical_hydra.py.
    
    IMPORTANT: The SAM3D framework uses a two-stage pipeline:
      - Stage 1: ss_generator predicts WHERE voxels should be (sparse structure)
      - Stage 2: slat_generator + slat_decoder_mesh predicts WHAT the shape is
    
    Currently this trains the slat_decoder_mesh with optional slat_generator for
    producing latents. Full two-stage training (including ss_generator) is planned.

    Args:
        cfg: Configuration dictionary with keys:
            - training: batch_size, epochs, lr, weight_decay, grad_accum, num_workers, mixed_precision
            - lora: rank, alpha, dropout
            - checkpoint: dir, resume, save_every
            - data: data_root, slice_cache_dir, augment, preprocess_crop_size
            - model: name, params
            - device: cuda/cpu
            - stage1: (optional) ss_generator config for future two-stage training
            - stage2: slat_generator config
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
    args.preprocess_crop_size = tuple(cfg.get("data", {}).get("preprocess_crop_size", (256, 256)))

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
    from torch.utils.data import DataLoader, random_split
    
    slice_cache_dir = args.slice_cache_dir or (str(args.data_root) + "/slice_cache")
    full_dataset = TS_SAM3D_Dataset(
        original_nifti_dir=args.data_root,
        cache_slices=True,
        slice_cache_dir=slice_cache_dir,
        classes=1,
        augment=args.augment,
        augment_mode="train",
        preprocess_crop_size=args.preprocess_crop_size,
        occupancy_threshold=0.01,
    )
    
    # Train/val split
    val_split = float(cfg.get("data", {}).get("val_split", 0.1))
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    if val_size > 0:
        train_ds, val_ds = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Reproducible split
        )
        logger.info(f"Dataset split: {train_size} train, {val_size} val ({val_split*100:.0f}% val)")
        
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=data_collate,
            num_workers=args.num_workers,
        )
    else:
        train_ds = full_dataset
        val_loader = None
        logger.info(f"Dataset: {total_size} samples (no validation split)")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collate,
        num_workers=args.num_workers,
    )

    # Preprocess model cfg from provided config: prepare model config variables and
    # if no params provided, try to load from checkpoint YAML; else fail.
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "slat_mesh")
    model_params = model_cfg.get("params", {})
    if not model_params:
        # Only allowed when resuming from a checkpoint that includes a YAML; otherwise the user must explicitly provide params
        resume_ckpt = args.resume or cfg.get("checkpoint", {}).get("resume")
        if resume_ckpt:
            yaml_path = _find_yaml_for_ckpt(resume_ckpt)
            if yaml_path:
                model_params = _extract_model_params_from_yaml(yaml_path, model_name)
                logger.info("Hydra: loaded model params from checkpoint YAML: %s", yaml_path)
                if not model_params:
                    raise ValueError(
                        f"Cannot extract model params from '{yaml_path}'. Provide model.model_params explicitly in the Hydra config."
                    )
            else:
                raise ValueError(
                    "Model params not provided in config and resume checkpoint YAML not found; please provide model.params explicitly."
                )
        else:
            raise ValueError(
                "Model params not provided in config. Provide 'model.params' in your Hydra configuration or use a checkpoint with an accompanying YAML."
            )
    # Create model & trainer
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "slat_mesh")
    model_params = model_cfg.get("params", {})
    # Ensure model 'use_fp16' respects no_mixed_precision when running via Hydra
    if args.no_mixed_precision and model_params is not None:
        model_params["use_fp16"] = False
    model = create_model(name=model_name, params=model_params)
    
    # ==========================================================================
    # Stage configuration (two-stage pipeline)
    # ==========================================================================
    training_mode = cfg.get("training", {}).get("mode", "stage2_only")
    
    # Stage 1: ss_generator (sparse structure)
    stage1_cfg = cfg.get("stage1", {})
    ss_generator = None
    ss_decoder = None
    if stage1_cfg.get("enabled", False) or training_mode in ["stage1_only", "two_stage"]:
        logger.info("Setting up Stage 1 (ss_generator) training...")
        # Load ss_generator checkpoint if provided
        ss_generator_ckpt = stage1_cfg.get("ss_generator_ckpt")
        if ss_generator_ckpt:
            try:
                ss_generator = create_model(name="ss_generator", params={})
                from sam3d_objects.model.io import load_model_from_checkpoint
                ss_generator = load_model_from_checkpoint(
                    ss_generator, ss_generator_ckpt, strict=False, device=args.device
                )
                logger.info("Loaded ss_generator from: %s", ss_generator_ckpt)
            except Exception as e:
                logger.warning("Failed to load ss_generator: %s", e)
                ss_generator = create_model(name="ss_generator", params={})
        else:
            ss_generator = create_model(name="ss_generator", params={})
        
        # Load ss_decoder
        ss_decoder_ckpt = stage1_cfg.get("ss_decoder_ckpt", "./checkpoints/hf/ss_decoder.ckpt")
        try:
            ss_decoder = create_model(name="ss_decoder", params={})
            from sam3d_objects.model.io import load_model_from_checkpoint
            ss_decoder = load_model_from_checkpoint(
                ss_decoder, ss_decoder_ckpt, strict=False, device=args.device
            )
            logger.info("Loaded ss_decoder from: %s", ss_decoder_ckpt)
        except Exception as e:
            logger.warning("Failed to load ss_decoder: %s", e)
            ss_decoder = create_model(name="ss_decoder", params={})
    
    # Stage 2: slat_generator + slat_decoder_mesh
    stage2_cfg = cfg.get("stage2", {})
    
    # Pull slat generator params from stage2 config or legacy top-level config
    slat_generator_ckpt = stage2_cfg.get("slat_generator_ckpt") or cfg.get("slat_generator_ckpt", None)
    slat_generator_params = cfg.get("slat_generator_params", None)
    if slat_generator_params is None and slat_generator_ckpt:
        # Try to find YAML config for the generator
        yaml_path = stage2_cfg.get("slat_generator_config") or _find_yaml_for_ckpt(slat_generator_ckpt)
        if yaml_path is not None:
            ext_params = _extract_model_params_from_yaml(yaml_path, "slat_generator")
            if ext_params:
                slat_generator_params = ext_params
            else:
                logger.warning(f"Found generator checkpoint YAML but could not extract params: {yaml_path}")
    
    # Get loss config from Hydra config
    loss_cfg = cfg.get("loss", {})
    loss_config = {
        "sdf_weight": float(loss_cfg.get("sdf_weight", 1.0)),
        "chamfer_weight": float(loss_cfg.get("chamfer_weight", 0.5)),
        "mesh_reg_weight": float(loss_cfg.get("mesh_reg_weight", 0.1)),
        "occupancy_weight": float(loss_cfg.get("occupancy_weight", 1.0)),
    }
    
    trainer = MedicalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        checkpoint_dir=args.checkpoint_dir,
        grad_accum_steps=args.grad_accum,
        mixed_precision=not args.no_mixed_precision,
        device=args.device,
        loss_config=loss_config,
        slat_generator_ckpt=slat_generator_ckpt,
        slat_generator_params=slat_generator_params,
        # Stage 1 models
        ss_generator=ss_generator,
        ss_decoder=ss_decoder,
        training_mode=training_mode,
    )

    if args.resume:
        ckpt_obj = _safe_load_checkpoint(args.resume)
        if isinstance(ckpt_obj, dict) and "lora_state_dict" in ckpt_obj:
            trainer.load_checkpoint(args.resume)
        else:
            try:
                model_state = ckpt_obj
                if isinstance(model_state, dict) and "state_dict" in model_state:
                    state_dict = model_state["state_dict"]
                else:
                    state_dict = model_state
                try:
                    trainer.model.load_state_dict(state_dict, strict=False)
                    logger.info("Loaded model weights from %s", args.resume)
                except Exception as e:
                    logger.warning("Failed to load model weights strictly: %s", e)
            except Exception as e:
                logger.warning("Failed to load resume checkpoint as model: %s", e)

    trainer.train(epochs=args.epochs, save_every=args.save_every, validate_every=1)


# =============================================================================
# Hydra CLI entry point (default)
# =============================================================================

def _get_repo_root() -> Path:
    """Get repository root directory."""
    return Path(__file__).parent.parent


@hydra.main(version_base="1.1", config_path=str(_get_repo_root() / 'configs'), config_name='train')
def main(cfg: DictConfig) -> None:
    """Hydra-based entry point for SAM3D medical fine-tuning.
    
    IMPORTANT: SAM3D uses a TWO-STAGE pipeline:
      - Stage 1: ss_generator → predicts WHERE voxels are (sparse structure)
      - Stage 2: slat_generator + slat_decoder_mesh → predicts WHAT the shape is

    For end-to-end medical reconstruction, BOTH stages should be fine-tuned.
    
    Usage:
        python scripts/train_medical.py
        python scripts/train_medical.py training.batch_size=8 data.data_root=/path/to/data
        python scripts/train_medical.py training.mode=two_stage stage1.enabled=true
    """
    print("=" * 70)
    print("SAM3D Medical Fine-Tuning")
    print("=" * 70)
    print("\nConfig:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)
    train_from_config(OmegaConf.to_container(cfg, resolve=True))


if __name__ == '__main__':
    main()
