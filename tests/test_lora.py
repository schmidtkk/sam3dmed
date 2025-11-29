"""Tests for LoRA (Low-Rank Adaptation) utilities."""

import pytest
import torch
import torch.nn as nn

from sam3d_objects.model.lora import (
    LoRALinear,
    count_parameters,
    freeze_base_params,
    get_lora_params,
    get_lora_state_dict,
    inject_lora,
    load_lora_state_dict,
    merge_lora_weights,
    setup_lora_for_medical_finetuning,
)


class DummyAttention(nn.Module):
    """Dummy attention module for testing."""

    def __init__(self, channels: int = 64, num_heads: int = 4):
        super().__init__()
        self.to_qkv = nn.Linear(channels, channels * 3)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x)
        # Simplified: just use values
        _, _, v = qkv.chunk(3, dim=-1)
        return self.to_out(v)


class DummyTransformerBlock(nn.Module):
    """Dummy transformer block with attention and FFN."""

    def __init__(self, channels: int = 64, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = DummyAttention(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.out_layer = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return self.out_layer(x)


class DummyModel(nn.Module):
    """Dummy model with multiple transformer blocks."""

    def __init__(self, channels: int = 64, num_blocks: int = 2):
        super().__init__()
        self.embed = nn.Linear(channels, channels)
        self.blocks = nn.ModuleList([DummyTransformerBlock(channels) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)


class TestLoRALinear:
    """Tests for LoRALinear class."""

    def test_init(self):
        """Test LoRALinear initialization."""
        linear = nn.Linear(64, 128)
        lora = LoRALinear(linear, rank=4, alpha=8.0)

        assert lora.rank == 4
        assert lora.alpha == 8.0
        assert lora.scaling == 2.0  # alpha / rank
        assert lora.lora_A.shape == (4, 64)
        assert lora.lora_B.shape == (128, 4)

    def test_forward_shape(self):
        """Test LoRALinear forward pass preserves shape."""
        linear = nn.Linear(64, 128)
        lora = LoRALinear(linear, rank=4, alpha=8.0)

        x = torch.randn(2, 10, 64)
        y = lora(x)

        assert y.shape == (2, 10, 128)

    def test_initial_output_matches_original(self):
        """Test that with zero B, output matches original linear."""
        linear = nn.Linear(64, 128)
        lora = LoRALinear(linear, rank=4, alpha=8.0)

        x = torch.randn(2, 10, 64)

        # B is initialized to zeros, so LoRA should not change output
        with torch.no_grad():
            original_out = linear(x)
            lora_out = lora(x)

        torch.testing.assert_close(lora_out, original_out)

    def test_lora_path_adds_to_output(self):
        """Test that non-zero B adds to output."""
        linear = nn.Linear(64, 128)
        lora = LoRALinear(linear, rank=4, alpha=8.0)

        # Manually set B to non-zero
        lora.lora_B.data = torch.randn_like(lora.lora_B)

        x = torch.randn(2, 10, 64)

        with torch.no_grad():
            original_out = linear(x)
            lora_out = lora(x)

        # Outputs should now differ
        assert not torch.allclose(lora_out, original_out)

    def test_merge_weights(self):
        """Test weight merging for inference."""
        linear = nn.Linear(64, 128)
        lora = LoRALinear(linear, rank=4, alpha=8.0)

        # Set non-zero LoRA weights
        lora.lora_A.data = torch.randn_like(lora.lora_A)
        lora.lora_B.data = torch.randn_like(lora.lora_B)

        x = torch.randn(2, 10, 64)

        with torch.no_grad():
            lora_out = lora(x)
            merged = lora.merge_weights()
            merged_out = merged(x)

        # Use looser tolerance for floating-point precision
        torch.testing.assert_close(merged_out, lora_out, atol=1e-4, rtol=1e-4)

    def test_dropout(self):
        """Test LoRA with dropout."""
        linear = nn.Linear(64, 128)
        lora = LoRALinear(linear, rank=4, alpha=8.0, dropout=0.5)

        lora.lora_B.data = torch.randn_like(lora.lora_B)

        x = torch.randn(2, 10, 64)

        lora.train()
        outputs = [lora(x) for _ in range(10)]

        # With dropout, outputs should vary
        assert not all(torch.allclose(outputs[0], out) for out in outputs[1:])


class TestInjectLoRA:
    """Tests for inject_lora function."""

    def test_inject_into_attention(self):
        """Test LoRA injection into attention layers."""
        model = DummyModel(channels=64, num_blocks=2)

        num_injected = inject_lora(model, target_modules=["to_qkv", "to_out"], rank=4)

        # Each block has 1 to_qkv + 1 to_out = 2 layers, 2 blocks = 4 total
        assert num_injected == 4

        # Check that layers are now LoRALinear
        for block in model.blocks:
            assert isinstance(block.attn.to_qkv, LoRALinear)
            assert isinstance(block.attn.to_out, LoRALinear)

    def test_inject_custom_rank_alpha(self):
        """Test injection with custom rank and alpha."""
        model = DummyModel(channels=64, num_blocks=1)

        inject_lora(model, target_modules=["to_qkv"], rank=8, alpha=16.0)

        lora_layer = model.blocks[0].attn.to_qkv
        assert lora_layer.rank == 8
        assert lora_layer.alpha == 16.0
        assert lora_layer.scaling == 2.0  # 16 / 8

    def test_inject_no_matching_layers(self):
        """Test that no injection happens when no layers match."""
        model = DummyModel(channels=64, num_blocks=1)

        num_injected = inject_lora(model, target_modules=["nonexistent_layer"])

        assert num_injected == 0


class TestFreezeBaseParams:
    """Tests for freeze_base_params function."""

    def test_freeze_base_unfreeze_lora(self):
        """Test that base params are frozen but LoRA params are trainable."""
        model = DummyModel(channels=64, num_blocks=1)
        inject_lora(model, target_modules=["to_qkv", "to_out"], rank=4)

        num_frozen, num_trainable = freeze_base_params(model)

        assert num_frozen > 0
        assert num_trainable > 0

        # Check that LoRA params are trainable
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert param.requires_grad is True
            else:
                assert param.requires_grad is False


class TestGetLoRAParams:
    """Tests for get_lora_params function."""

    def test_get_lora_params(self):
        """Test that only LoRA parameters are returned."""
        model = DummyModel(channels=64, num_blocks=1)
        inject_lora(model, target_modules=["to_qkv"], rank=4)
        freeze_base_params(model)

        lora_params = get_lora_params(model)

        # 1 to_qkv layer -> 2 params (lora_A, lora_B)
        assert len(lora_params) == 2

        # All should be trainable
        for param in lora_params:
            assert param.requires_grad


class TestLoRAStateDict:
    """Tests for state dict save/load functions."""

    def test_get_and_load_lora_state_dict(self):
        """Test saving and loading LoRA weights."""
        model = DummyModel(channels=64, num_blocks=1)
        inject_lora(model, target_modules=["to_qkv"], rank=4)

        # Modify LoRA weights
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.data = torch.randn_like(param)

        # Save
        state = get_lora_state_dict(model)

        # Create new model and load
        model2 = DummyModel(channels=64, num_blocks=1)
        inject_lora(model2, target_modules=["to_qkv"], rank=4)
        load_lora_state_dict(model2, state)

        # Check weights match
        for (n1, p1), (_n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            if "lora_A" in n1 or "lora_B" in n1:
                torch.testing.assert_close(p1, p2)


class TestMergeLoRAWeights:
    """Tests for merge_lora_weights function."""

    def test_merge_produces_same_output(self):
        """Test that merged model produces same output."""
        model = DummyModel(channels=64, num_blocks=1)
        inject_lora(model, target_modules=["to_qkv", "to_out"], rank=4)

        # Set random LoRA weights
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.data = torch.randn_like(param) * 0.1

        x = torch.randn(2, 10, 64)

        with torch.no_grad():
            out_before = model(x).clone()
            merge_lora_weights(model)
            out_after = model(x)

        torch.testing.assert_close(out_after, out_before)

    def test_merge_removes_lora_modules(self):
        """Test that merge removes LoRALinear modules."""
        model = DummyModel(channels=64, num_blocks=1)
        inject_lora(model, target_modules=["to_qkv"], rank=4)

        # Verify LoRA exists
        assert isinstance(model.blocks[0].attn.to_qkv, LoRALinear)

        merge_lora_weights(model)

        # Verify LoRA is gone
        assert isinstance(model.blocks[0].attn.to_qkv, nn.Linear)
        assert not isinstance(model.blocks[0].attn.to_qkv, LoRALinear)


class TestCountParameters:
    """Tests for count_parameters function."""

    def test_count_parameters(self):
        """Test parameter counting."""
        model = DummyModel(channels=64, num_blocks=1)

        # Before LoRA
        counts_before = count_parameters(model)
        assert counts_before["total"] > 0
        assert counts_before["lora"] == 0

        # After LoRA
        inject_lora(model, target_modules=["to_qkv"], rank=4)
        counts_after = count_parameters(model)

        assert counts_after["lora"] > 0
        assert counts_after["total"] > counts_before["total"]


class TestSetupLoRAForMedicalFinetuning:
    """Tests for setup_lora_for_medical_finetuning function."""

    def test_complete_setup(self):
        """Test complete LoRA setup for medical fine-tuning."""
        model = DummyModel(channels=64, num_blocks=2)

        counts = setup_lora_for_medical_finetuning(
            model,
            rank=4,
            alpha=8.0,
            dropout=0.0,
            unfreeze_output_layers=True,
            output_layer_names=["out_layer", "output_layer"],
        )

        # Check LoRA was injected
        assert counts["lora"] > 0

        # Check output layers are trainable
        output_trainable = False
        for name, param in model.named_parameters():
            if "output_layer" in name or "out_layer" in name:
                if param.requires_grad:
                    output_trainable = True
                    break
        assert output_trainable

    def test_forward_pass_after_setup(self):
        """Test that model works after LoRA setup."""
        model = DummyModel(channels=64, num_blocks=2)
        setup_lora_for_medical_finetuning(model, rank=4, alpha=8.0)

        x = torch.randn(2, 10, 64)
        y = model(x)

        assert y.shape == x.shape

    def test_backward_pass_after_setup(self):
        """Test that gradients flow only to LoRA and output layers."""
        model = DummyModel(channels=64, num_blocks=1)
        setup_lora_for_medical_finetuning(model, rank=4, alpha=8.0, unfreeze_output_layers=True)

        x = torch.randn(2, 10, 64)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Check gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
            else:
                assert param.grad is None, f"Unexpected gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
