"""
LoRA (Low-Rank Adaptation) utilities for SAM3D medical fine-tuning.

This module provides functions to inject LoRA adapters into the SAM3D backbone,
specifically targeting the SLatMeshDecoder for mesh-only training.

Usage:
    from sam3d_objects.model.lora import inject_lora, get_lora_params, freeze_base_params

    model = load_model(...)
    inject_lora(model, target_modules=["to_qkv", "to_out"], rank=4, alpha=8)
    freeze_base_params(model)
    
    # Train only LoRA params
    optimizer = AdamW(get_lora_params(model), lr=1e-3)
"""

from typing import List, Optional, Dict, Tuple, Union
import torch
import torch.nn as nn
from loguru import logger


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation layer that wraps a nn.Linear.
    
    Implements: h = Wx + (alpha/r) * B @ A @ x
    where W is frozen original weight, A and B are trainable low-rank matrices.
    
    Args:
        original_linear: The original nn.Linear module to wrap
        rank: LoRA rank (r)
        alpha: Scaling factor (alpha)
        dropout: Dropout probability for LoRA path
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout for LoRA path
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Initialize A with kaiming uniform (like typical LoRA)
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        # Initialize B with zeros so initial output is same as original
        nn.init.zeros_(self.lora_B)
        
        # Store whether original linear had bias
        self.has_bias = original_linear.bias is not None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original linear pass
        result = self.original_linear(x)
        
        # LoRA path: x -> dropout -> A -> B -> scale
        lora_x = self.lora_dropout(x)
        lora_out = lora_x @ self.lora_A.T  # (*, in_features) @ (in_features, rank) -> (*, rank)
        lora_out = lora_out @ self.lora_B.T  # (*, rank) @ (rank, out_features) -> (*, out_features)
        lora_out = lora_out * self.scaling
        
        return result + lora_out
    
    @property
    def weight(self) -> torch.Tensor:
        """Return merged weight for inference."""
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        return self.original_linear.weight + delta_w
    
    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Return original bias."""
        return self.original_linear.bias
    
    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA weights into original linear and return a new nn.Linear.
        Useful for inference to avoid overhead.
        """
        merged = nn.Linear(
            self.original_linear.in_features,
            self.original_linear.out_features,
            bias=self.has_bias,
        )
        merged.weight.data = self.weight.data.clone()
        if self.has_bias:
            merged.bias.data = self.original_linear.bias.data.clone()
        return merged


def _find_modules_by_name(
    model: nn.Module,
    target_names: List[str],
    parent_name: str = "",
) -> List[Tuple[nn.Module, str, nn.Linear]]:
    """
    Find all nn.Linear modules matching target names.
    
    Returns:
        List of (parent_module, attr_name, linear_module) tuples
    """
    found = []
    
    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        
        if isinstance(module, nn.Linear) and any(t in name for t in target_names):
            found.append((model, name, module))
            logger.debug(f"Found target linear: {full_name}")
        else:
            # Recurse
            found.extend(_find_modules_by_name(module, target_names, full_name))
    
    return found


def inject_lora(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
) -> int:
    """
    Inject LoRA adapters into the model's linear layers.
    
    Args:
        model: The model to modify in-place
        target_modules: List of module name patterns to target. 
                       Default: ["to_qkv", "to_q", "to_kv", "to_out"] for attention layers
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout probability for LoRA path
        
    Returns:
        Number of layers injected with LoRA
    """
    if target_modules is None:
        # Default: target attention projection layers
        target_modules = ["to_qkv", "to_q", "to_kv", "to_out"]
    
    # Find all matching linear layers
    targets = _find_modules_by_name(model, target_modules)
    
    if not targets:
        logger.warning(f"No linear layers found matching {target_modules}")
        return 0
    
    # Replace with LoRA-wrapped versions
    for parent, name, linear in targets:
        lora_linear = LoRALinear(
            original_linear=linear,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        setattr(parent, name, lora_linear)
        logger.info(f"Injected LoRA into {name}: in={linear.in_features}, out={linear.out_features}, rank={rank}")
    
    logger.info(f"Total LoRA layers injected: {len(targets)}")
    return len(targets)


def freeze_base_params(model: nn.Module) -> Tuple[int, int]:
    """
    Freeze all parameters except LoRA adapters.
    
    Returns:
        (num_frozen, num_trainable) parameter counts
    """
    num_frozen = 0
    num_trainable = 0
    
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
            num_trainable += param.numel()
        else:
            param.requires_grad = False
            num_frozen += param.numel()
    
    logger.info(f"Frozen params: {num_frozen:,} | Trainable LoRA params: {num_trainable:,}")
    return num_frozen, num_trainable


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters for optimizer.
    
    Returns:
        List of trainable LoRA parameters
    """
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad and ("lora_A" in name or "lora_B" in name):
            params.append(param)
    return params


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only LoRA weights from model state dict.
    Useful for saving compact LoRA checkpoints.
    
    Returns:
        State dict containing only LoRA weights
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state[name] = param.data.clone()
    return lora_state


def load_lora_state_dict(model: nn.Module, lora_state: Dict[str, torch.Tensor]) -> None:
    """
    Load LoRA weights into model.
    
    Args:
        model: Model with LoRA layers already injected
        lora_state: State dict from get_lora_state_dict
    """
    current_state = model.state_dict()
    for name, tensor in lora_state.items():
        if name in current_state:
            current_state[name] = tensor
        else:
            logger.warning(f"LoRA weight not found in model: {name}")
    model.load_state_dict(current_state, strict=False)


def merge_lora_weights(model: nn.Module) -> None:
    """
    Merge all LoRA weights into base weights in-place.
    After merging, the model can be used without LoRA overhead.
    
    Note: This modifies the model in-place and removes LoRA layers.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            # Get parent and attr name
            parts = name.rsplit(".", 1)
            if len(parts) == 1:
                parent = model
                attr = parts[0]
            else:
                parent = model.get_submodule(parts[0])
                attr = parts[1]
            
            # Merge and replace
            merged_linear = module.merge_weights()
            setattr(parent, attr, merged_linear)
            logger.info(f"Merged LoRA weights for {name}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total, trainable, and LoRA parameters.
    
    Returns:
        Dict with 'total', 'trainable', 'lora' keys
    """
    total = 0
    trainable = 0
    lora = 0
    
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
        if "lora_A" in name or "lora_B" in name:
            lora += param.numel()
    
    return {"total": total, "trainable": trainable, "lora": lora}


# Convenience function for typical medical fine-tuning setup
def setup_lora_for_medical_finetuning(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    unfreeze_output_layers: bool = True,
    output_layer_names: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Complete setup for medical fine-tuning with LoRA.
    
    1. Inject LoRA into attention layers
    2. Freeze base weights
    3. Optionally unfreeze output layers
    
    Args:
        model: Model to setup
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        unfreeze_output_layers: Whether to also train output layers
        output_layer_names: Patterns for output layers to unfreeze
        
    Returns:
        Dict with parameter counts
    """
    # Inject LoRA
    num_lora = inject_lora(
        model,
        target_modules=["to_qkv", "to_q", "to_kv", "to_out"],
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    
    # Freeze base params
    freeze_base_params(model)
    
    # Optionally unfreeze output layers
    if unfreeze_output_layers:
        if output_layer_names is None:
            output_layer_names = ["out_layer", "output_layer", "head"]
        
        unfrozen_count = 0
        for name, param in model.named_parameters():
            if any(ol in name for ol in output_layer_names):
                param.requires_grad = True
                unfrozen_count += param.numel()
        
        if unfrozen_count > 0:
            logger.info(f"Unfroze {unfrozen_count:,} output layer params")
    
    return count_parameters(model)
