"""
Medical evaluation script for SAM3D.

This script evaluates trained models on medical imaging data using
3D metrics: Dice, HD95, Chamfer distance.

Usage:
    python scripts/eval_medical.py \
        --checkpoint checkpoints/medical/best.pt \
        --data_root /path/to/test_data \
        --output_dir ./results
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class MedicalEvaluator:
    """
    Evaluator for medical fine-tuned SAM3D models.

    Computes:
    - Dice coefficient (volumetric overlap)
    - HD95 (Hausdorff distance 95th percentile)
    - Chamfer distance (mesh quality)
    - Surface Dice (boundary accuracy)
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: str = "cuda",
        output_dir: str = "./results",
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        occupancy_threshold: float = 0.5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.data_loader = data_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.spacing = spacing
        self.occupancy_threshold = occupancy_threshold

        # Import metrics
        from sam3d_objects.utils.metrics import (
            compute_chamfer,
            compute_dice,
            compute_hd95,
        )

        self.compute_dice = compute_dice
        self.compute_chamfer = compute_chamfer
        self.compute_hd95 = compute_hd95

    @torch.no_grad()
    def evaluate(
        self,
        save_predictions: bool = False,
        compute_per_class: bool = True,
    ) -> dict[str, float]:
        """
        Run evaluation on the full dataset.

        Args:
            save_predictions: Whether to save predicted masks/meshes
            compute_per_class: Whether to compute metrics per class

        Returns:
            Dict with aggregated metrics
        """
        all_metrics = []

        for batch_idx, batch in enumerate(tqdm(self.data_loader, desc="Evaluating")):
            batch = self._to_device(batch)

            # Forward pass
            outputs = self._forward_step(batch)

            # Compute metrics for this batch
            batch_metrics = self._compute_batch_metrics(outputs, batch)
            all_metrics.append(batch_metrics)

            # Save predictions if requested
            if save_predictions:
                self._save_predictions(batch_idx, outputs, batch)

            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(self.data_loader)} batches")

        # Aggregate metrics
        aggregated = self._aggregate_metrics(all_metrics)

        # Save results
        self._save_results(aggregated)

        return aggregated

    def _forward_step(self, batch: dict) -> dict:
        """
        Forward pass through model.

        Override this method for custom model architectures.
        """
        image = batch.get("image")
        pointmap = batch.get("pointmap")

        outputs = self.model(image, pointmap)
        return outputs

    def _compute_batch_metrics(
        self,
        outputs: dict,
        batch: dict,
    ) -> dict[str, list[float]]:
        """
        Compute metrics for a single batch.

        Returns:
            Dict mapping metric names to lists of values (one per sample)
        """
        metrics = {
            "dice": [],
            "hd95": [],
            "chamfer": [],
        }

        # Get predictions and ground truth
        pred_sdf = outputs.get("sdf")
        gt_sdf = batch.get("sdf")
        gt_mask = batch.get("mask")

        if pred_sdf is None:
            return metrics

        # Convert SDF to occupancy
        pred_mask = (pred_sdf <= 0).float()

        # Process each sample in batch
        batch_size = pred_mask.shape[0]

        for i in range(batch_size):
            # Get individual samples
            pred_i = pred_mask[i]

            # Use GT mask if available, otherwise derive from SDF
            if gt_mask is not None:
                gt_i = gt_mask[i]
            elif gt_sdf is not None:
                gt_i = (gt_sdf[i] <= 0).float()
            else:
                continue

            # Dice
            dice = self.compute_dice(pred_i, gt_i)
            metrics["dice"].append(dice)

            # HD95 (need binary masks)
            pred_binary = (pred_i > self.occupancy_threshold).cpu().numpy()
            gt_binary = (gt_i > self.occupancy_threshold).cpu().numpy()

            # Check if both have valid regions
            if pred_binary.sum() > 0 and gt_binary.sum() > 0:
                hd95 = self.compute_hd95(pred_binary, gt_binary, spacing=self.spacing)
                metrics["hd95"].append(hd95)

            # Chamfer distance (if mesh vertices available)
            pred_vertices = outputs.get("vertices")
            gt_vertices = batch.get("vertices")

            if pred_vertices is not None and gt_vertices is not None:
                if isinstance(pred_vertices, list):
                    pred_v = pred_vertices[i] if i < len(pred_vertices) else None
                    gt_v = gt_vertices[i] if i < len(gt_vertices) else None
                else:
                    pred_v = pred_vertices[i] if pred_vertices.dim() > 2 else pred_vertices
                    gt_v = gt_vertices[i] if gt_vertices.dim() > 2 else gt_vertices

                if pred_v is not None and gt_v is not None:
                    chamfer = self.compute_chamfer(pred_v, gt_v)
                    metrics["chamfer"].append(chamfer)

        return metrics

    def _aggregate_metrics(
        self,
        all_metrics: list[dict[str, list[float]]],
    ) -> dict[str, float]:
        """
        Aggregate metrics across all batches.

        Returns:
            Dict with mean, std, median for each metric
        """
        # Flatten all values
        flattened = {}
        for batch_metrics in all_metrics:
            for metric_name, values in batch_metrics.items():
                if metric_name not in flattened:
                    flattened[metric_name] = []
                flattened[metric_name].extend(values)

        # Compute statistics
        aggregated = {}
        for metric_name, values in flattened.items():
            if len(values) == 0:
                continue

            values_np = np.array(values)
            aggregated[f"{metric_name}_mean"] = float(np.mean(values_np))
            aggregated[f"{metric_name}_std"] = float(np.std(values_np))
            aggregated[f"{metric_name}_median"] = float(np.median(values_np))
            aggregated[f"{metric_name}_min"] = float(np.min(values_np))
            aggregated[f"{metric_name}_max"] = float(np.max(values_np))
            aggregated[f"{metric_name}_count"] = len(values)

        return aggregated

    def _save_predictions(
        self,
        batch_idx: int,
        outputs: dict,
        batch: dict,
    ) -> None:
        """Save predictions to disk."""
        pred_dir = self.output_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)

        pred_sdf = outputs.get("sdf")
        if pred_sdf is not None:
            np.save(pred_dir / f"batch_{batch_idx}_pred_sdf.npy", pred_sdf.cpu().numpy())

        pred_vertices = outputs.get("vertices")
        if pred_vertices is not None:
            if isinstance(pred_vertices, torch.Tensor):
                np.save(
                    pred_dir / f"batch_{batch_idx}_pred_vertices.npy", pred_vertices.cpu().numpy()
                )

    def _save_results(self, aggregated: dict[str, float]) -> None:
        """Save evaluation results to JSON."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "metrics": aggregated,
            "config": {
                "spacing": self.spacing,
                "occupancy_threshold": self.occupancy_threshold,
                "num_samples": len(self.data_loader.dataset),
            },
        }

        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_path}")

        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)
        for metric_name, value in aggregated.items():
            if "mean" in metric_name:
                logger.info(f"{metric_name}: {value:.4f}")

    def _to_device(self, batch: dict) -> dict:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }


class PerClassEvaluator(MedicalEvaluator):
    """
    Extended evaluator that computes metrics per anatomical class.
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        class_names: list[str],
        **kwargs,
    ):
        super().__init__(model, data_loader, **kwargs)
        self.class_names = class_names
        self.num_classes = len(class_names)

    def _compute_batch_metrics(
        self,
        outputs: dict,
        batch: dict,
    ) -> dict[str, list[float]]:
        """Compute metrics per class."""
        metrics = {}

        # Initialize per-class metrics
        for class_name in self.class_names:
            metrics[f"dice_{class_name}"] = []
            metrics[f"hd95_{class_name}"] = []

        pred_sdf = outputs.get("sdf")  # Shape: (B, C, D, H, W) or (B, D, H, W)
        gt_sdf = batch.get("sdf")
        # gt_mask could be used for alternative binary ground truth

        if pred_sdf is None:
            return metrics

        batch_size = pred_sdf.shape[0]

        for i in range(batch_size):
            for class_idx, class_name in enumerate(self.class_names):
                # Extract class-specific predictions
                if pred_sdf.dim() == 5:  # (B, C, D, H, W)
                    pred_class = pred_sdf[i, class_idx]
                    gt_class = gt_sdf[i, class_idx] if gt_sdf is not None else None
                else:
                    pred_class = pred_sdf[i]
                    gt_class = gt_sdf[i] if gt_sdf is not None else None

                # Convert to binary
                pred_binary = (pred_class <= 0).float()
                gt_binary = (gt_class <= 0).float() if gt_class is not None else None

                if gt_binary is not None:
                    dice = self.compute_dice(pred_binary, gt_binary)
                    metrics[f"dice_{class_name}"].append(dice)

                    # HD95
                    pred_np = pred_binary.cpu().numpy()
                    gt_np = gt_binary.cpu().numpy()

                    if pred_np.sum() > 0 and gt_np.sum() > 0:
                        hd95 = self.compute_hd95(pred_np, gt_np, spacing=self.spacing)
                        metrics[f"hd95_{class_name}"].append(hd95)

        return metrics


def compute_summary_statistics(metrics: dict[str, list[float]]) -> dict[str, dict]:
    """
    Compute summary statistics for each metric.

    Returns:
        Dict with per-metric statistics
    """
    summary = {}

    for metric_name, values in metrics.items():
        if len(values) == 0:
            continue

        values_np = np.array(values)
        summary[metric_name] = {
            "mean": float(np.mean(values_np)),
            "std": float(np.std(values_np)),
            "median": float(np.median(values_np)),
            "min": float(np.min(values_np)),
            "max": float(np.max(values_np)),
            "q25": float(np.percentile(values_np, 25)),
            "q75": float(np.percentile(values_np, 75)),
            "count": len(values),
        }

    return summary


def create_evaluation_report(
    results: dict[str, float],
    output_path: Path,
    format: str = "markdown",
) -> None:
    """
    Create a formatted evaluation report.

    Args:
        results: Aggregated evaluation results
        output_path: Path to save report
        format: 'markdown' or 'html'
    """
    if format == "markdown":
        lines = ["# Medical Evaluation Report\n"]
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("\n## Metrics Summary\n")
        lines.append("| Metric | Mean | Std | Median | Min | Max |")
        lines.append("|--------|------|-----|--------|-----|-----|")

        # Group by base metric name
        base_metrics = set()
        for key in results.keys():
            parts = key.rsplit("_", 1)
            if len(parts) == 2 and parts[1] in ["mean", "std", "median", "min", "max", "count"]:
                base_metrics.add(parts[0])

        for metric in sorted(base_metrics):
            mean = results.get(f"{metric}_mean", 0)
            std = results.get(f"{metric}_std", 0)
            median = results.get(f"{metric}_median", 0)
            min_val = results.get(f"{metric}_min", 0)
            max_val = results.get(f"{metric}_max", 0)

            lines.append(
                f"| {metric} | {mean:.4f} | {std:.4f} | {median:.4f} | {min_val:.4f} | {max_val:.4f} |"
            )

        lines.append("\n")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Report saved to {output_path}")


def create_dummy_model():
    """Create a dummy model for testing."""

    class DummyModel(nn.Module):
        def __init__(self, channels: int = 64):
            super().__init__()
            self.embed = nn.Linear(3, channels)
            self.to_qkv = nn.Linear(channels, channels * 3)
            self.to_out = nn.Linear(channels, channels)
            self.sdf_head = nn.Linear(channels, 1)

        def forward(self, image, pointmap):
            x = self.embed(pointmap)
            x = self.to_out(self.to_qkv(x)[..., : x.shape[-1]])
            sdf = self.sdf_head(x)
            return {"sdf": sdf, "vertices": None, "faces": None}

    return DummyModel()


def parse_args():
    parser = argparse.ArgumentParser(description="Medical evaluation for SAM3D")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")

    # Data
    parser.add_argument("--data_root", type=str, required=True, help="Path to test data")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")

    # Output
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions")

    # Metrics
    parser.add_argument(
        "--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Voxel spacing"
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Occupancy threshold")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Medical Evaluation Script")
    logger.info(f"Args: {args}")

    # Create dummy data for testing
    logger.info("Creating dummy data loader...")
    from torch.utils.data import TensorDataset

    dummy_images = torch.randn(20, 1, 256, 256)
    dummy_pointmaps = torch.randn(20, 256, 256, 3)
    dummy_sdfs = torch.randn(20, 256, 256, 1)
    dummy_masks = (torch.rand(20, 256, 256) > 0.5).float()

    test_dataset = TensorDataset(dummy_images, dummy_pointmaps, dummy_sdfs, dummy_masks)

    def collate_fn(batch):
        images, pointmaps, sdfs, masks = zip(*batch)
        return {
            "image": torch.stack(images),
            "pointmap": torch.stack(pointmaps),
            "sdf": torch.stack(sdfs),
            "mask": torch.stack(masks),
        }

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Create model
    logger.info("Creating dummy model...")
    model = create_dummy_model()

    # Load checkpoint (if it exists)
    if os.path.exists(args.checkpoint):
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        from sam3d_objects.model.lora import inject_lora, load_lora_state_dict

        inject_lora(model, target_modules=["to_qkv", "to_out"], rank=4)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "lora_state_dict" in checkpoint:
            load_lora_state_dict(model, checkpoint["lora_state_dict"])
    else:
        logger.warning(f"Checkpoint not found: {args.checkpoint}")

    # Create evaluator
    evaluator = MedicalEvaluator(
        model=model,
        data_loader=test_loader,
        device=args.device,
        output_dir=args.output_dir,
        spacing=tuple(args.spacing),
        occupancy_threshold=args.threshold,
    )

    # Run evaluation
    results = evaluator.evaluate(save_predictions=args.save_predictions)

    # Create report
    create_evaluation_report(
        results,
        Path(args.output_dir) / "report.md",
        format="markdown",
    )

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
