"""Tests for medical training harness."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_medical import (
    MedicalTrainer,
    MedicalTrainingLosses,
    create_dummy_model,
)


class TestMedicalTrainingLosses:
    """Tests for MedicalTrainingLosses class."""

    def test_sdf_loss_l1(self):
        """Test L1 SDF loss."""
        losses = MedicalTrainingLosses(sdf_loss_type="l1")

        pred = torch.randn(2, 1, 8, 8, 8)
        gt = torch.randn(2, 1, 8, 8, 8)

        loss = losses.compute_sdf_loss(pred, gt)

        assert loss.shape == ()
        assert loss >= 0

    def test_sdf_loss_mse(self):
        """Test MSE SDF loss."""
        losses = MedicalTrainingLosses(sdf_loss_type="mse")

        pred = torch.randn(2, 1, 8, 8, 8)
        gt = torch.randn(2, 1, 8, 8, 8)

        loss = losses.compute_sdf_loss(pred, gt)

        assert loss.shape == ()
        assert loss >= 0

    def test_sdf_loss_with_mask(self):
        """Test SDF loss with mask."""
        losses = MedicalTrainingLosses()

        pred = torch.randn(2, 1, 8, 8, 8)
        gt = torch.randn(2, 1, 8, 8, 8)
        mask = torch.ones_like(pred)
        mask[:, :, :4, :, :] = 0  # Mask out half

        loss = losses.compute_sdf_loss(pred, gt, mask)

        assert loss.shape == ()

    def test_mesh_regularization(self):
        """Test mesh regularization loss."""
        losses = MedicalTrainingLosses()

        # Simple triangle mesh
        vertices = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ]
        )
        faces = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 3],
                [1, 2, 3],
                [0, 2, 3],
            ]
        )

        reg_loss = losses.compute_mesh_regularization(vertices, faces)

        assert reg_loss.shape == ()
        assert reg_loss >= 0

    def test_occupancy_loss(self):
        """Test occupancy loss."""
        losses = MedicalTrainingLosses()

        pred = torch.randn(2, 8, 8, 8)  # logits
        gt = (torch.rand(2, 8, 8, 8) > 0.5).float()

        occ_loss = losses.compute_occupancy_loss(pred, gt)

        assert occ_loss.shape == ()
        assert occ_loss >= 0

    def test_forward_all_losses(self):
        """Test forward with all losses."""
        losses = MedicalTrainingLosses(
            sdf_weight=1.0,
            chamfer_weight=0.0,  # Skip chamfer for simplicity
            mesh_reg_weight=0.0,  # Skip mesh reg
            occupancy_weight=0.5,
        )

        pred_sdf = torch.randn(2, 1, 8, 8, 8)
        gt_sdf = torch.randn(2, 1, 8, 8, 8)
        pred_occ = torch.randn(2, 8, 8, 8)
        gt_occ = torch.rand(2, 8, 8, 8) > 0.5

        result = losses(
            pred_sdf=pred_sdf,
            gt_sdf=gt_sdf,
            pred_occupancy=pred_occ,
            gt_occupancy=gt_occ,
        )

        assert "sdf" in result
        assert "occupancy" in result
        assert "total" in result
        assert result["total"] >= 0

    def test_forward_partial_losses(self):
        """Test forward with only some losses."""
        losses = MedicalTrainingLosses()

        # Only SDF loss
        pred_sdf = torch.randn(2, 1, 8, 8, 8)
        gt_sdf = torch.randn(2, 1, 8, 8, 8)

        result = losses(pred_sdf=pred_sdf, gt_sdf=gt_sdf)

        assert "sdf" in result
        assert "total" in result
        assert "chamfer" not in result


class TestCreateDummyModel:
    """Tests for dummy model creation."""

    def test_create_model(self):
        """Test model creation."""
        model = create_dummy_model()

        assert isinstance(model, nn.Module)
        assert hasattr(model, "to_qkv")
        assert hasattr(model, "to_out")

    def test_forward(self):
        """Test model forward pass."""
        model = create_dummy_model()

        image = torch.randn(2, 1, 256, 256)
        pointmap = torch.randn(2, 256, 256, 3)

        outputs = model(image, pointmap)

        assert isinstance(outputs, dict)
        assert "sdf" in outputs


class TestMedicalTrainer:
    """Tests for MedicalTrainer class."""

    @pytest.fixture
    def dummy_trainer(self, tmp_path):
        """Create a dummy trainer for testing."""
        model = create_dummy_model()

        # Create dummy dataloader - make SDF match model output shape
        dummy_images = torch.randn(8, 1, 256, 256)
        dummy_pointmaps = torch.randn(8, 256, 256, 3)
        # Match model output: (B, 256, 256, 1) for SDF
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

        trainer = MedicalTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            lr=1e-3,
            lora_rank=4,
            lora_alpha=8.0,
            checkpoint_dir=str(tmp_path),
            mixed_precision=False,  # CPU testing
            device="cpu",
        )

        return trainer

    def test_trainer_init(self, dummy_trainer):
        """Test trainer initialization."""
        assert dummy_trainer.lora_rank == 4
        assert dummy_trainer.lora_alpha == 8.0
        assert dummy_trainer.optimizer is not None

    def test_lora_setup(self, dummy_trainer):
        """Test that LoRA was set up correctly."""
        from sam3d_objects.model.lora import LoRALinear

        model = dummy_trainer.model

        # Check that to_qkv and to_out are LoRA layers
        assert isinstance(model.to_qkv, LoRALinear)
        assert isinstance(model.to_out, LoRALinear)

    def test_train_epoch(self, dummy_trainer):
        """Test single training epoch."""
        losses = dummy_trainer.train_epoch()

        assert "total" in losses
        assert losses["total"] >= 0

    def test_validate(self, dummy_trainer):
        """Test validation."""
        losses = dummy_trainer.validate()

        assert "total" in losses
        assert losses["total"] >= 0

    def test_save_and_load_checkpoint(self, dummy_trainer):
        """Test checkpoint save and load."""
        # Modify a LoRA parameter
        for name, param in dummy_trainer.model.named_parameters():
            if "lora_A" in name:
                param.data = torch.randn_like(param)
                break

        # Save
        dummy_trainer.save_checkpoint("test.pt")
        checkpoint_path = dummy_trainer.checkpoint_dir / "test.pt"
        assert checkpoint_path.exists()

        # Load into new trainer
        model2 = create_dummy_model()
        trainer2 = MedicalTrainer(
            model=model2,
            train_loader=dummy_trainer.train_loader,
            checkpoint_dir=str(dummy_trainer.checkpoint_dir),
            mixed_precision=False,
            device="cpu",
        )

        trainer2.load_checkpoint(str(checkpoint_path))

        # Compare LoRA weights
        for (n1, p1), (_n2, p2) in zip(
            dummy_trainer.model.named_parameters(), trainer2.model.named_parameters()
        ):
            if "lora_A" in n1 or "lora_B" in n1:
                torch.testing.assert_close(p1, p2)

    def test_to_device(self, dummy_trainer):
        """Test batch device transfer."""
        batch = {
            "image": torch.randn(2, 1, 256, 256),
            "pointmap": torch.randn(2, 256, 256, 3),
            "metadata": "not a tensor",
        }

        moved = dummy_trainer._to_device(batch)

        assert moved["image"].device == dummy_trainer.device
        assert moved["pointmap"].device == dummy_trainer.device
        assert moved["metadata"] == "not a tensor"

    def test_aggregate_losses(self, dummy_trainer):
        """Test loss aggregation."""
        losses_list = [
            {"total": 1.0, "sdf": 0.5},
            {"total": 2.0, "sdf": 1.0},
            {"total": 3.0, "sdf": 1.5},
        ]

        agg = dummy_trainer._aggregate_losses(losses_list)

        assert agg["total"] == 2.0
        assert agg["sdf"] == 1.0


class TestTrainingIntegration:
    """Integration tests for training."""

    def test_short_training_run(self, tmp_path):
        """Test a short training run completes without error."""
        model = create_dummy_model()

        # Create minimal dataset - match model output shape
        dummy_images = torch.randn(4, 1, 256, 256)
        dummy_pointmaps = torch.randn(4, 256, 256, 3)
        dummy_sdfs = torch.randn(4, 256, 256, 1)
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

        trainer = MedicalTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            lr=1e-3,
            checkpoint_dir=str(tmp_path),
            mixed_precision=False,
            device="cpu",
        )

        # Train for 2 epochs
        trainer.train(epochs=2, save_every=1, validate_every=1)

        # Check outputs exist
        assert (tmp_path / "final.pt").exists()
        assert (tmp_path / "history.json").exists()
        assert len(trainer.history["train"]) == 2
        assert len(trainer.history["val"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
