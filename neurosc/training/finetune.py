"""
Finetuning strategies including LoRA and other parameter-efficient methods.
"""

from dataclasses import dataclass
from typing import Optional, List
import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA (Low-Rank Adaptation) finetuning.
    
    Parameters
    ----------
    r : int, optional (default: 8)
        Rank of the low-rank decomposition.
    alpha : int, optional (default: 16)
        Scaling factor for LoRA.
    dropout : float, optional (default: 0.1)
        Dropout probability for LoRA layers.
    target_modules : list, optional
        List of module names to apply LoRA to.
    """
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default: apply to query and value projections
            self.target_modules = ["q_proj", "v_proj"]


class LoRALayer(nn.Module):
    """
    LoRA layer for parameter-efficient finetuning.
    
    Parameters
    ----------
    in_features : int
        Input dimension.
    out_features : int
        Output dimension.
    r : int
        Rank of low-rank decomposition.
    alpha : int
        Scaling factor.
    dropout : float
        Dropout probability.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA transformation."""
        # x @ A @ B * scaling
        return self.dropout(x @ self.lora_A @ self.lora_B) * self.scaling


def apply_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """
    Apply LoRA to a model.
    
    Parameters
    ----------
    model : nn.Module
        Model to apply LoRA to.
    config : LoRAConfig
        LoRA configuration.
    
    Returns
    -------
    nn.Module
        Model with LoRA layers added.
    
    Examples
    --------
    >>> from neurosc.training import LoRAConfig, apply_lora
    >>> config = LoRAConfig(r=8, alpha=16)
    >>> model = apply_lora(model, config)
    """
    # Freeze base model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Add LoRA layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module should have LoRA
            module_name = name.split('.')[-1]
            if module_name in config.target_modules or any(target in name for target in config.target_modules):
                # Add LoRA layer
                lora_layer = LoRALayer(
                    module.in_features,
                    module.out_features,
                    r=config.r,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                
                # Store original forward
                original_forward = module.forward
                
                # Create new forward that includes LoRA
                def make_lora_forward(orig_forward, lora):
                    def forward(x):
                        return orig_forward(x) + lora(x)
                    return forward
                
                module.forward = make_lora_forward(original_forward, lora_layer)
                module.lora = lora_layer
    
    return model


def finetune_model(
    model: nn.Module,
    train_dataloader,
    eval_dataloader = None,
    num_classes: Optional[int] = None,
    strategy: str = "full",
    lora_config: Optional[LoRAConfig] = None,
    **training_args
):
    """
    Finetune a foundation model on a downstream task.
    
    Parameters
    ----------
    model : nn.Module
        Pretrained foundation model.
    train_dataloader : DataLoader
        Training data loader.
    eval_dataloader : DataLoader, optional
        Evaluation data loader.
    num_classes : int, optional
        Number of output classes for classification.
    strategy : str, optional (default: 'full')
        Finetuning strategy: 'full', 'lora', 'freeze_backbone'.
    lora_config : LoRAConfig, optional
        LoRA configuration (required if strategy='lora').
    **training_args
        Additional arguments passed to Trainer.
    
    Returns
    -------
    nn.Module
        Finetuned model.
    
    Examples
    --------
    >>> import neurosc as nsc
    >>> from neurosc.training import finetune_model, LoRAConfig
    >>> 
    >>> # Full finetuning
    >>> model = nsc.load_model("scgpt-base-neuroscience")
    >>> finetuned_model = finetune_model(
    ...     model,
    ...     train_dataloader,
    ...     num_classes=10,
    ...     strategy="full",
    ...     output_dir="./output",
    ...     num_epochs=5
    ... )
    >>> 
    >>> # LoRA finetuning
    >>> lora_config = LoRAConfig(r=8, alpha=16)
    >>> finetuned_model = finetune_model(
    ...     model,
    ...     train_dataloader,
    ...     strategy="lora",
    ...     lora_config=lora_config,
    ...     output_dir="./output"
    ... )
    """
    from .trainer import Trainer, TrainingArguments
    
    # Add classification head if num_classes specified
    if num_classes is not None and hasattr(model, 'add_classification_head'):
        model.add_classification_head(num_classes)
    
    # Apply finetuning strategy
    if strategy == "lora":
        if lora_config is None:
            lora_config = LoRAConfig()
        model = apply_lora(model, lora_config)
        print(f"Applied LoRA with rank={lora_config.r}, alpha={lora_config.alpha}")
    
    elif strategy == "freeze_backbone":
        if hasattr(model, 'freeze_backbone'):
            model.freeze_backbone()
        else:
            # Freeze all except classification head
            for name, param in model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
        print("Froze backbone parameters")
    
    elif strategy == "full":
        if hasattr(model, 'unfreeze_backbone'):
            model.unfreeze_backbone()
        print("Full finetuning (all parameters trainable)")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from: 'full', 'lora', 'freeze_backbone'")
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Create training arguments
    args = TrainingArguments(**training_args)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )
    
    # Train
    trainer.train()
    
    return model

