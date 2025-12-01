"""Tests for medical training harness.

IMPORTANT: These tests are STRICT about failures.
- No silent fallbacks
- All model loading errors should raise exceptions
- Tests should fail if critical components cannot be instantiated
"""

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
    create_model,
    _extract_model_params_from_yaml,
    _find_yaml_for_ckpt,
)


# =============================================================================
# Test Loss Functions
# =============================================================================

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


# =============================================================================
# Test Model Factory - STRICT, NO FALLBACKS
# =============================================================================

class TestModelFactory:
    """Tests for model factory functions - STRICT about failures."""

    def test_create_dummy_model(self):
        """Test dummy model creation."""
        model = create_dummy_model()

        assert isinstance(model, nn.Module)
        assert hasattr(model, "to_qkv")
        assert hasattr(model, "to_out")

    def test_dummy_model_forward(self):
        """Test dummy model forward pass."""
        model = create_dummy_model()

        image = torch.randn(2, 1, 256, 256)
        pointmap = torch.randn(2, 256, 256, 3)

        outputs = model(image, pointmap)

        assert isinstance(outputs, dict)
        assert "sdf" in outputs

    def test_create_slat_mesh_model(self):
        """Test SLatMeshDecoder instantiation - MUST NOT FAIL."""
        # model_channels must be divisible by 256 (8 * 32 for group norm)
        model = create_model(name="slat_mesh", params={
            "resolution": 4,
            "model_channels": 256,
            "latent_channels": 16,
            "num_blocks": 2,
            "num_heads": 4,
        })

        assert isinstance(model, nn.Module)
        
        # Must have attention layers for LoRA injection
        module_names = [name for name, _ in model.named_modules()]
        assert any("to_qkv" in name for name in module_names), "Model MUST have to_qkv layers"
        assert any("to_out" in name for name in module_names), "Model MUST have to_out layers"

    def test_create_slat_mesh_invalid_channels_raises(self):
        """Test that invalid model_channels raises ValueError."""
        with pytest.raises(ValueError, match="Invalid 'model_channels'"):
            create_model(name="slat_mesh", params={
                "resolution": 4,
                "model_channels": 64,  # Invalid: not divisible by 256
                "latent_channels": 16,
                "num_blocks": 2,
                "num_heads": 4,
            })

    def test_create_unknown_model_raises(self):
        """Test that unknown model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model name"):
            create_model(name="nonexistent_model")

    def test_create_slat_generator(self):
        """Test slat_generator model creation using Hydra instantiate pattern."""
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        from sam3d_objects.model.io import load_model_from_checkpoint, filter_and_remove_prefix_state_dict_fn
        
        workspace_dir = "/mnt/nas1/disk01/weidongguo/workspace/sam-3d-objects"
        config_path = f"{workspace_dir}/checkpoints/hf/slat_generator.yaml"
        ckpt_path = f"{workspace_dir}/checkpoints/hf/slat_generator.ckpt"
        
        # Load config and extract backbone config
        config = OmegaConf.load(config_path)["module"]["generator"]["backbone"]
        
        # Instantiate model from config
        model = instantiate(config)
        assert isinstance(model, nn.Module), "instantiate MUST return a Module"
        
        # Load weights
        state_dict_prefix_func = filter_and_remove_prefix_state_dict_fn("_base_models.generator.")
        model = load_model_from_checkpoint(
            model,
            ckpt_path,
            strict=True,
            device="cpu",
            freeze=True,
            eval=True,
            state_dict_key="state_dict",
            state_dict_fn=state_dict_prefix_func,
        )
        
        assert model is not None
        # Verify model has expected structure
        assert hasattr(model, 'forward')

    def test_create_ss_generator(self):
        """Test ss_generator (Stage 1) model creation using Hydra instantiate pattern."""
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        from sam3d_objects.model.io import load_model_from_checkpoint, filter_and_remove_prefix_state_dict_fn
        
        workspace_dir = "/mnt/nas1/disk01/weidongguo/workspace/sam-3d-objects"
        config_path = f"{workspace_dir}/checkpoints/hf/ss_generator.yaml"
        ckpt_path = f"{workspace_dir}/checkpoints/hf/ss_generator.ckpt"
        
        # Load config and extract backbone config
        config = OmegaConf.load(config_path)["module"]["generator"]["backbone"]
        
        # Instantiate model from config  
        model = instantiate(config)
        assert isinstance(model, nn.Module), "instantiate MUST return a Module"
        
        # Load weights
        state_dict_prefix_func = filter_and_remove_prefix_state_dict_fn("_base_models.generator.")
        model = load_model_from_checkpoint(
            model,
            ckpt_path,
            strict=True,
            device="cpu",
            freeze=True,
            eval=True,
            state_dict_key="state_dict",
            state_dict_fn=state_dict_prefix_func,
        )
        
        assert model is not None
        assert hasattr(model, 'forward')

    def test_create_ss_decoder(self):
        """Test ss_decoder (Stage 1) model creation - MUST NOT FAIL."""
        model = create_model(name="ss_decoder", params={
            "out_channels": 1,
            "latent_channels": 8,
        })
        
        assert isinstance(model, nn.Module)


# =============================================================================
# Test YAML Extraction - STRICT
# =============================================================================

class TestYamlExtraction:
    """Tests for YAML config extraction."""

    def test_extract_slat_generator_params(self, tmp_path):
        """Test extracting params from slat_generator YAML structure."""
        yaml_content = """
module:
  generator:
    backbone:
      reverse_fn:
        backbone:
          model_channels: 1024
          in_channels: 8
          out_channels: 8
          num_blocks: 24
          num_heads: 16
          resolution: 64
"""
        yaml_file = tmp_path / "slat_generator.yaml"
        yaml_file.write_text(yaml_content)
        
        params = _extract_model_params_from_yaml(str(yaml_file), "slat_generator")
        
        assert params["model_channels"] == 1024
        assert params["in_channels"] == 8
        assert params["out_channels"] == 8
        assert params["num_blocks"] == 24
        assert params["num_heads"] == 16

    def test_find_yaml_for_ckpt(self, tmp_path):
        """Test finding YAML config for checkpoint."""
        # Create checkpoint and yaml files
        (tmp_path / "model.ckpt").touch()
        (tmp_path / "model.yaml").write_text("test: true")
        
        yaml_path = _find_yaml_for_ckpt(str(tmp_path / "model.ckpt"))
        
        assert yaml_path == str(tmp_path / "model.yaml")

    def test_find_yaml_for_ckpt_returns_none_if_missing(self, tmp_path):
        """Test returns None if no YAML exists."""
        (tmp_path / "model.ckpt").touch()
        
        yaml_path = _find_yaml_for_ckpt(str(tmp_path / "model.ckpt"))
        
        assert yaml_path is None


# =============================================================================
# Test Trainer - STRICT INITIALIZATION
# =============================================================================

class TestMedicalTrainer:
    """Tests for MedicalTrainer class - STRICT about initialization."""

    @pytest.fixture
    def dummy_loader(self):
        """Create a dummy dataloader."""
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

        return DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    def test_trainer_init_with_dummy_model(self, dummy_loader, tmp_path):
        """Test trainer initialization with dummy model."""
        model = create_dummy_model()

        trainer = MedicalTrainer(
            model=model,
            train_loader=dummy_loader,
            val_loader=dummy_loader,
            lr=1e-3,
            lora_rank=4,
            lora_alpha=8.0,
            checkpoint_dir=str(tmp_path),
            mixed_precision=False,
            device="cpu",
        )

        assert trainer.lora_rank == 4
        assert trainer.lora_alpha == 8.0
        assert trainer.optimizer is not None

    def test_trainer_init_with_slat_mesh(self, dummy_loader, tmp_path):
        """Test trainer initialization with SLatMeshDecoder - MUST inject LoRA."""
        model = create_model(name="slat_mesh", params={
            "resolution": 4,
            "model_channels": 256,
            "latent_channels": 16,
            "num_blocks": 2,
            "num_heads": 4,
            "device": "cpu",
        })

        trainer = MedicalTrainer(
            model=model,
            train_loader=dummy_loader,
            val_loader=dummy_loader,
            lr=1e-3,
            checkpoint_dir=str(tmp_path),
            mixed_precision=False,
            device="cpu",
        )

        # LoRA MUST be injected
        from sam3d_objects.model.lora import LoRALinear

        found = False
        for _, module in trainer.model.named_modules():
            if isinstance(module, LoRALinear):
                found = True
                break

        assert found, "LoRA injection MUST succeed for SLatMeshDecoder"

    def test_lora_layers_injected(self, dummy_loader, tmp_path):
        """Test that LoRA was set up correctly on dummy model."""
        from sam3d_objects.model.lora import LoRALinear

        model = create_dummy_model()
        
        trainer = MedicalTrainer(
            model=model,
            train_loader=dummy_loader,
            checkpoint_dir=str(tmp_path),
            mixed_precision=False,
            device="cpu",
        )

        # to_qkv and to_out MUST be LoRA layers
        assert isinstance(model.to_qkv, LoRALinear), "to_qkv MUST be LoRALinear"
        assert isinstance(model.to_out, LoRALinear), "to_out MUST be LoRALinear"

    def test_train_epoch(self, dummy_loader, tmp_path):
        """Test single training epoch."""
        model = create_dummy_model()
        
        trainer = MedicalTrainer(
            model=model,
            train_loader=dummy_loader,
            checkpoint_dir=str(tmp_path),
            mixed_precision=False,
            device="cpu",
        )

        losses = trainer.train_epoch()

        assert "total" in losses
        assert losses["total"] >= 0

    def test_validate(self, dummy_loader, tmp_path):
        """Test validation."""
        model = create_dummy_model()
        
        trainer = MedicalTrainer(
            model=model,
            train_loader=dummy_loader,
            val_loader=dummy_loader,
            checkpoint_dir=str(tmp_path),
            mixed_precision=False,
            device="cpu",
        )

        losses = trainer.validate()

        assert "total" in losses
        assert losses["total"] >= 0

    def test_save_and_load_checkpoint(self, dummy_loader, tmp_path):
        """Test checkpoint save and load."""
        model = create_dummy_model()
        
        trainer = MedicalTrainer(
            model=model,
            train_loader=dummy_loader,
            checkpoint_dir=str(tmp_path),
            mixed_precision=False,
            device="cpu",
        )

        # Modify a LoRA parameter
        for name, param in trainer.model.named_parameters():
            if "lora_A" in name:
                param.data = torch.randn_like(param)
                break

        # Save
        trainer.save_checkpoint("test.pt")
        checkpoint_path = trainer.checkpoint_dir / "test.pt"
        assert checkpoint_path.exists(), "Checkpoint file MUST be saved"

        # Load into new trainer
        model2 = create_dummy_model()
        trainer2 = MedicalTrainer(
            model=model2,
            train_loader=dummy_loader,
            checkpoint_dir=str(tmp_path),
            mixed_precision=False,
            device="cpu",
        )

        trainer2.load_checkpoint(str(checkpoint_path))

        # Compare LoRA weights - MUST match
        for (n1, p1), (_n2, p2) in zip(
            trainer.model.named_parameters(), trainer2.model.named_parameters()
        ):
            if "lora_A" in n1 or "lora_B" in n1:
                torch.testing.assert_close(p1, p2)

    def test_aggregate_losses(self, dummy_loader, tmp_path):
        """Test loss aggregation."""
        model = create_dummy_model()
        
        trainer = MedicalTrainer(
            model=model,
            train_loader=dummy_loader,
            checkpoint_dir=str(tmp_path),
            mixed_precision=False,
            device="cpu",
        )

        losses_list = [
            {"total": 1.0, "sdf": 0.5},
            {"total": 2.0, "sdf": 1.0},
            {"total": 3.0, "sdf": 1.5},
        ]

        agg = trainer._aggregate_losses(losses_list)

        assert agg["total"] == 2.0
        assert agg["sdf"] == 1.0


# =============================================================================
# Test Two-Stage Training Configuration
# =============================================================================

class TestTwoStageTraining:
    """Tests for two-stage training mode."""

    @pytest.fixture
    def dummy_loader(self):
        """Create a dummy dataloader with occupancy data."""
        dummy_images = torch.randn(8, 1, 256, 256)
        dummy_pointmaps = torch.randn(8, 256, 256, 3)
        dummy_sdfs = torch.randn(8, 256, 256, 1)
        dummy_masks = (torch.rand(8, 256, 256) > 0.5).float()
        dummy_occupancy = (torch.rand(8, 16, 16, 16) > 0.5).float()

        dataset = TensorDataset(dummy_images, dummy_pointmaps, dummy_sdfs, dummy_masks, dummy_occupancy)

        def collate_fn(batch):
            images, pointmaps, sdfs, masks, occs = zip(*batch)
            return {
                "image": torch.stack(images),
                "pointmap": torch.stack(pointmaps),
                "sdf": torch.stack(sdfs),
                "mask": torch.stack(masks),
                "gt_occupancy": torch.stack(occs),
            }

        return DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    def test_trainer_training_mode_stage2_only(self, dummy_loader, tmp_path):
        """Test trainer in stage2_only mode (default)."""
        model = create_dummy_model()
        
        trainer = MedicalTrainer(
            model=model,
            train_loader=dummy_loader,
            checkpoint_dir=str(tmp_path),
            training_mode="stage2_only",
            mixed_precision=False,
            device="cpu",
        )

        assert trainer.training_mode == "stage2_only"
        assert trainer.train_stage1 is False
        assert trainer.ss_generator is None

    def test_trainer_training_mode_stage1_only(self, dummy_loader, tmp_path):
        """Test trainer in stage1_only mode requires ss_generator."""
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        from sam3d_objects.model.io import load_model_from_checkpoint, filter_and_remove_prefix_state_dict_fn
        
        workspace_dir = "/mnt/nas1/disk01/weidongguo/workspace/sam-3d-objects"
        config_path = f"{workspace_dir}/checkpoints/hf/ss_generator.yaml"
        ckpt_path = f"{workspace_dir}/checkpoints/hf/ss_generator.ckpt"
        
        # Load config and extract backbone config
        config = OmegaConf.load(config_path)["module"]["generator"]["backbone"]
        
        # Instantiate and load ss_generator
        ss_generator = instantiate(config)
        state_dict_prefix_func = filter_and_remove_prefix_state_dict_fn("_base_models.generator.")
        ss_generator = load_model_from_checkpoint(
            ss_generator,
            ckpt_path,
            strict=True,
            device="cpu",
            freeze=True,
            eval=True,
            state_dict_key="state_dict",
            state_dict_fn=state_dict_prefix_func,
        )
        
        model = create_dummy_model()
        
        trainer = MedicalTrainer(
            model=model,
            train_loader=dummy_loader,
            checkpoint_dir=str(tmp_path),
            training_mode="stage1_only",
            ss_generator=ss_generator,
            mixed_precision=False,
            device="cpu",
        )

        assert trainer.training_mode == "stage1_only"
        assert trainer.train_stage1 is True
        assert trainer.ss_generator is not None


# =============================================================================
# Integration Tests - STRICT
# =============================================================================

class TestTrainingIntegration:
    """Integration tests for training - STRICT about completion."""

    def test_short_training_run(self, tmp_path):
        """Test a short training run completes without error."""
        model = create_dummy_model()

        # Create minimal dataset
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

        # Train for 2 epochs - MUST complete
        trainer.train(epochs=2, save_every=1, validate_every=1)

        # Outputs MUST exist
        assert (tmp_path / "final.pt").exists(), "final.pt MUST be saved"
        assert (tmp_path / "history.json").exists(), "history.json MUST be saved"
        assert len(trainer.history["train"]) == 2, "MUST have 2 train history entries"
        assert len(trainer.history["val"]) == 2, "MUST have 2 val history entries"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for spconv")
    def test_slat_mesh_training_run(self, tmp_path):
        """Test training with SLatMeshDecoder on CUDA.
        
        This tests the full SLatMeshDecoder training with proper LoRA injection.
        """
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        from sam3d_objects.model.io import load_model_from_checkpoint
        
        device = "cuda"
        workspace_dir = "/mnt/nas1/disk01/weidongguo/workspace/sam-3d-objects"
        config_path = f"{workspace_dir}/checkpoints/hf/slat_decoder_mesh.yaml"
        ckpt_path = f"{workspace_dir}/checkpoints/hf/slat_decoder_mesh.ckpt"
        
        # Load config and instantiate SLatMeshDecoder
        config = OmegaConf.load(config_path)
        model = instantiate(config)
        model = load_model_from_checkpoint(
            model, ckpt_path,
            strict=True,
            device=device,
            freeze=True,
            eval=False,  # Keep in train mode for training
            state_dict_key=None,
        )
        model = model.to(device)

        # Create dataset with proper shapes for SLatMeshDecoder
        dummy_images = torch.randn(4, 1, 64, 64)
        dummy_pointmaps = torch.randn(4, 64, 64, 3)
        dummy_sdfs = torch.randn(4, 1, 16, 16, 16)
        dummy_masks = (torch.rand(4, 16, 16, 16) > 0.5).float()

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
            lr=1e-4,
            checkpoint_dir=str(tmp_path),
            mixed_precision=False,
            device=device,
        )

        # Train for 1 epoch - MUST complete
        trainer.train(epochs=1, save_every=1, validate_every=1)

        # Verify training happened
        assert (tmp_path / "final.pt").exists(), "final.pt MUST be saved"
        assert len(trainer.history["train"]) == 1


# =============================================================================
# Test TensorBoard Logging
# =============================================================================

class TestTensorBoardLogging:
    """Tests for TensorBoard logging."""

    def test_tensorboard_dir_created(self, tmp_path):
        """Test TensorBoard directory is created during training."""
        model = create_dummy_model()

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
            checkpoint_dir=str(tmp_path),
            mixed_precision=False,
            device="cpu",
        )

        # TensorBoard should be available
        from scripts.train_medical import TENSORBOARD_AVAILABLE
        if TENSORBOARD_AVAILABLE:
            assert trainer.writer is not None, "TensorBoard writer MUST be created"
            assert (tmp_path / "tensorboard").exists(), "TensorBoard dir MUST exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
