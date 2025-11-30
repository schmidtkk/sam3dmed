"""Tests for medical evaluation script."""

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_medical import (
    MedicalEvaluator,
    PerClassEvaluator,
    compute_summary_statistics,
    create_dummy_model,
    create_evaluation_report,
)


class TestMedicalEvaluator:
    """Tests for MedicalEvaluator class."""

    @pytest.fixture
    def dummy_evaluator(self, tmp_path):
        """Create a dummy evaluator for testing."""
        model = create_dummy_model()

        # Create dummy dataloader
        dummy_images = torch.randn(8, 1, 256, 256)
        dummy_pointmaps = torch.randn(8, 256, 256, 3)
        dummy_sdfs = torch.randn(8, 256, 256, 1)
        dummy_masks = (torch.rand(8, 256, 256) > 0.5).float()

        dataset = TensorDataset(dummy_images, dummy_pointmaps, dummy_sdfs, dummy_masks)

        def collate_fn(batch):
            images, pointmaps, sdfs, masks = zip(*batch)
            return {
                "image": torch.stack(images),
                "pointmap": torch.stack(pointmaps),
                "sdf": torch.stack(sdfs),
                "mask": torch.stack(masks),
            }

        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        evaluator = MedicalEvaluator(
            model=model,
            data_loader=loader,
            device="cpu",
            output_dir=str(tmp_path),
            spacing=(1.0, 1.0, 1.0),
        )

        return evaluator

    def test_evaluator_init(self, dummy_evaluator):
        """Test evaluator initialization."""
        assert dummy_evaluator.model is not None
        assert dummy_evaluator.data_loader is not None
        assert dummy_evaluator.spacing == (1.0, 1.0, 1.0)

    def test_to_device(self, dummy_evaluator):
        """Test batch device transfer."""
        batch = {
            "image": torch.randn(2, 1, 256, 256),
            "pointmap": torch.randn(2, 256, 256, 3),
            "metadata": "string_value",
        }

        moved = dummy_evaluator._to_device(batch)

        assert moved["image"].device == dummy_evaluator.device
        assert moved["metadata"] == "string_value"

    def test_forward_step(self, dummy_evaluator):
        """Test forward pass."""
        batch = {
            "image": torch.randn(2, 1, 256, 256).to(dummy_evaluator.device),
            "pointmap": torch.randn(2, 256, 256, 3).to(dummy_evaluator.device),
        }

        outputs = dummy_evaluator._forward_step(batch)

        assert isinstance(outputs, dict)
        assert "sdf" in outputs

    def test_compute_batch_metrics(self, dummy_evaluator):
        """Test batch metric computation."""
        outputs = {
            "sdf": torch.randn(2, 256, 256, 1),  # Random SDF
        }
        batch = {
            "sdf": torch.randn(2, 256, 256, 1),
            "mask": (torch.rand(2, 256, 256) > 0.5).float(),
        }

        metrics = dummy_evaluator._compute_batch_metrics(outputs, batch)

        assert "dice" in metrics
        assert isinstance(metrics["dice"], list)

    def test_aggregate_metrics(self, dummy_evaluator):
        """Test metric aggregation."""
        all_metrics = [
            {"dice": [0.8, 0.85], "hd95": [5.0, 4.5]},
            {"dice": [0.9, 0.75], "hd95": [3.0, 6.0]},
        ]

        aggregated = dummy_evaluator._aggregate_metrics(all_metrics)

        assert "dice_mean" in aggregated
        assert "dice_std" in aggregated
        assert "dice_median" in aggregated
        assert "hd95_mean" in aggregated

        # Check values are reasonable
        assert 0 <= aggregated["dice_mean"] <= 1
        assert aggregated["dice_count"] == 4

    def test_evaluate(self, dummy_evaluator):
        """Test full evaluation."""
        results = dummy_evaluator.evaluate(save_predictions=False)

        assert isinstance(results, dict)
        # Should have at least dice metrics
        assert any("dice" in k for k in results.keys())

    def test_save_results(self, dummy_evaluator):
        """Test results saving."""
        aggregated = {
            "dice_mean": 0.85,
            "dice_std": 0.05,
            "hd95_mean": 4.5,
        }

        dummy_evaluator._save_results(aggregated)

        results_path = dummy_evaluator.output_dir / "evaluation_results.json"
        assert results_path.exists()

        with open(results_path) as f:
            saved = json.load(f)

        assert saved["metrics"]["dice_mean"] == 0.85


class TestPerClassEvaluator:
    """Tests for PerClassEvaluator class."""

    def test_per_class_metrics(self, tmp_path):
        """Test per-class metric computation."""
        model = create_dummy_model()

        # Create dummy data with 2 classes
        dummy_images = torch.randn(4, 1, 256, 256)
        dummy_pointmaps = torch.randn(4, 256, 256, 3)
        dummy_sdfs = torch.randn(4, 256, 256, 1)  # Single class for simplicity
        dummy_masks = (torch.rand(4, 256, 256) > 0.5).float()

        dataset = TensorDataset(dummy_images, dummy_pointmaps, dummy_sdfs, dummy_masks)

        def collate_fn(batch):
            images, pointmaps, sdfs, masks = zip(*batch)
            return {
                "image": torch.stack(images),
                "pointmap": torch.stack(pointmaps),
                "sdf": torch.stack(sdfs),
                "mask": torch.stack(masks),
            }

        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        evaluator = PerClassEvaluator(
            model=model,
            data_loader=loader,
            class_names=["liver", "spleen"],
            device="cpu",
            output_dir=str(tmp_path),
        )

        # Just test initialization
        assert evaluator.class_names == ["liver", "spleen"]
        assert evaluator.num_classes == 2


class TestSummaryStatistics:
    """Tests for summary statistics computation."""

    def test_compute_summary_statistics(self):
        """Test summary statistics."""
        metrics = {
            "dice": [0.8, 0.85, 0.9, 0.75, 0.88],
            "hd95": [5.0, 4.5, 3.0, 6.0, 4.0],
        }

        summary = compute_summary_statistics(metrics)

        assert "dice" in summary
        assert "hd95" in summary

        assert "mean" in summary["dice"]
        assert "std" in summary["dice"]
        assert "median" in summary["dice"]
        assert "q25" in summary["dice"]
        assert "q75" in summary["dice"]

        # Check dice mean is correct
        expected_mean = sum(metrics["dice"]) / len(metrics["dice"])
        assert abs(summary["dice"]["mean"] - expected_mean) < 1e-6

    def test_empty_metrics(self):
        """Test with empty metrics."""
        metrics = {"dice": []}
        summary = compute_summary_statistics(metrics)

        assert "dice" not in summary


class TestEvaluationReport:
    """Tests for evaluation report generation."""

    def test_create_markdown_report(self, tmp_path):
        """Test markdown report creation."""
        results = {
            "dice_mean": 0.85,
            "dice_std": 0.05,
            "dice_median": 0.86,
            "dice_min": 0.70,
            "dice_max": 0.95,
            "hd95_mean": 4.5,
            "hd95_std": 1.0,
            "hd95_median": 4.2,
            "hd95_min": 2.0,
            "hd95_max": 7.0,
        }

        output_path = tmp_path / "report.md"
        create_evaluation_report(results, output_path, format="markdown")

        assert output_path.exists()

        content = output_path.read_text()
        assert "# Medical Evaluation Report" in content
        assert "dice" in content
        assert "hd95" in content


class TestDummyModel:
    """Tests for dummy model."""

    def test_create_model(self):
        """Test model creation."""
        model = create_dummy_model()

        assert isinstance(model, nn.Module)
        assert hasattr(model, "to_qkv")

    def test_forward(self):
        """Test model forward pass."""
        model = create_dummy_model()

        image = torch.randn(2, 1, 256, 256)
        pointmap = torch.randn(2, 256, 256, 3)

        outputs = model(image, pointmap)

        assert "sdf" in outputs
        assert outputs["sdf"].shape == (2, 256, 256, 1)


class TestEvaluatorIntegration:
    """Integration tests for evaluation."""

    def test_full_evaluation_pipeline(self, tmp_path):
        """Test complete evaluation pipeline."""
        model = create_dummy_model()

        # Create dataset
        dummy_images = torch.randn(8, 1, 256, 256)
        dummy_pointmaps = torch.randn(8, 256, 256, 3)
        dummy_sdfs = torch.randn(8, 256, 256, 1)
        dummy_masks = (torch.rand(8, 256, 256) > 0.5).float()

        dataset = TensorDataset(dummy_images, dummy_pointmaps, dummy_sdfs, dummy_masks)

        def collate_fn(batch):
            images, pointmaps, sdfs, masks = zip(*batch)
            return {
                "image": torch.stack(images),
                "pointmap": torch.stack(pointmaps),
                "sdf": torch.stack(sdfs),
                "mask": torch.stack(masks),
            }

        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        evaluator = MedicalEvaluator(
            model=model,
            data_loader=loader,
            device="cpu",
            output_dir=str(tmp_path),
        )

        # Run evaluation
        evaluator.evaluate(save_predictions=False)

        # Check outputs
        assert (tmp_path / "evaluation_results.json").exists()

        # Load and verify results
        with open(tmp_path / "evaluation_results.json") as f:
            saved = json.load(f)

        assert "metrics" in saved
        assert "config" in saved
        assert "timestamp" in saved

    def test_visualization_outputs(self, tmp_path):
        """Test that evaluation visualization files are produced when requested."""
        model = create_dummy_model()

        # Create dataset
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        dummy_images = torch.randn(2, 1, 256, 256)
        dummy_pointmaps = torch.randn(2, 256, 256, 3)
        dummy_sdfs = torch.randn(2, 256, 256, 1)
        dummy_masks = (torch.rand(2, 256, 256) > 0.5).float()

        dataset = TensorDataset(dummy_images, dummy_pointmaps, dummy_sdfs, dummy_masks)

        def collate_fn(batch):
            images, pointmaps, sdfs, masks = zip(*batch)
            return {
                "image": torch.stack(images),
                "pointmap": torch.stack(pointmaps),
                "sdf": torch.stack(sdfs),
                "mask": torch.stack(masks),
            }

        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

        evaluator = MedicalEvaluator(
            model=model,
            data_loader=loader,
            device="cpu",
            output_dir=str(tmp_path),
        )

        evaluator.evaluate(save_predictions=False, visualize=True, visualize_every=1, visualize_format="html")

        viz_dir = tmp_path / "visualizations"
        assert viz_dir.exists()
        # Check presence of a sample folder with content
        subdirs = list(viz_dir.glob("batch_*"))
        assert len(subdirs) > 0
        # Check at least one file exists inside a sample directory
        files = list(subdirs[0].glob("**/*"))
        assert len(files) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
