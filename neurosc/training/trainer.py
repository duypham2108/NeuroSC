"""
Training utilities for finetuning foundation models.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class TrainingArguments:
    """
    Arguments for model training.
    
    Parameters
    ----------
    output_dir : str
        Directory to save model checkpoints and logs.
    num_epochs : int, optional (default: 10)
        Number of training epochs.
    learning_rate : float, optional (default: 1e-4)
        Learning rate for optimizer.
    batch_size : int, optional (default: 32)
        Training batch size.
    eval_batch_size : int, optional (default: 64)
        Evaluation batch size.
    weight_decay : float, optional (default: 0.01)
        Weight decay for optimizer.
    warmup_steps : int, optional (default: 500)
        Number of warmup steps for learning rate scheduler.
    gradient_accumulation_steps : int, optional (default: 1)
        Number of gradient accumulation steps.
    max_grad_norm : float, optional (default: 1.0)
        Maximum gradient norm for clipping.
    logging_steps : int, optional (default: 100)
        Log every N steps.
    eval_steps : int, optional (default: 500)
        Evaluate every N steps.
    save_steps : int, optional (default: 1000)
        Save checkpoint every N steps.
    save_total_limit : int, optional (default: 3)
        Maximum number of checkpoints to keep.
    device : str, optional
        Device to use for training ('cuda' or 'cpu').
    fp16 : bool, optional (default: False)
        Whether to use mixed precision training.
    seed : int, optional (default: 42)
        Random seed for reproducibility.
    """
    
    output_dir: str
    num_epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 32
    eval_batch_size: int = 64
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    device: Optional[str] = None
    fp16: bool = False
    seed: int = 42
    
    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        os.makedirs(self.output_dir, exist_ok=True)


class Trainer:
    """
    Trainer for finetuning foundation models.
    
    Parameters
    ----------
    model : nn.Module
        Model to train.
    args : TrainingArguments
        Training arguments.
    train_dataloader : DataLoader
        Training data loader.
    eval_dataloader : DataLoader, optional
        Evaluation data loader.
    loss_fn : callable, optional
        Loss function. If None, uses CrossEntropyLoss.
    callbacks : list, optional
        List of training callbacks.
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        loss_fn: Optional[Callable] = None,
        callbacks: Optional[List] = None,
    ):
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.callbacks = callbacks or []
        
        # Set random seed
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        # Move model to device
        self.model = self.model.to(args.device)
        
        # Loss function
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        
        total_steps = len(train_dataloader) * args.num_epochs
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps
        )
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf')
        
    def train(self):
        """
        Execute training loop.
        
        Returns
        -------
        dict
            Training history (losses, metrics).
        """
        print(f"***** Running Training *****")
        print(f"  Num examples = {len(self.train_dataloader.dataset)}")
        print(f"  Num epochs = {self.args.num_epochs}")
        print(f"  Batch size = {self.args.batch_size}")
        print(f"  Total steps = {len(self.train_dataloader) * self.args.num_epochs}")
        
        history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
        }
        
        # Training loop
        for epoch in range(self.args.num_epochs):
            self.epoch = epoch
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.args.num_epochs}")
            print(f"{'='*60}")
            
            # Train one epoch
            train_loss = self._train_epoch()
            history['train_loss'].append(train_loss)
            
            # Evaluate
            if self.eval_dataloader is not None:
                eval_loss = self.evaluate()
                history['eval_loss'].append(eval_loss)
                print(f"Eval Loss: {eval_loss:.4f}")
            
            # Save checkpoint
            self._save_checkpoint()
        
        print("\n***** Training Complete *****")
        return history
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for step, batch in enumerate(progress_bar):
            loss = self._training_step(batch)
            total_loss += loss
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Logging
            if self.global_step % self.args.logging_steps == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Step {self.global_step}: Loss = {avg_loss:.4f}")
            
            self.global_step += 1
        
        return total_loss / len(self.train_dataloader)
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single training step."""
        # Move batch to device
        batch = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch['genes'], batch['expression'])
                loss = self.loss_fn(outputs['logits'].squeeze(), batch.get('labels', torch.zeros(outputs['logits'].size(0))))
        else:
            outputs = self.model(batch['genes'], batch['expression'])
            loss = self.loss_fn(outputs['logits'].squeeze(), batch.get('labels', torch.zeros(outputs['logits'].size(0))))
        
        # Backward pass
        loss = loss / self.args.gradient_accumulation_steps
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step
        if (self.global_step + 1) % self.args.gradient_accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.item() * self.args.gradient_accumulation_steps
    
    def evaluate(self) -> float:
        """
        Evaluate the model.
        
        Returns
        -------
        float
            Average evaluation loss.
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(batch['genes'], batch['expression'])
                loss = self.loss_fn(outputs['logits'].squeeze(), batch.get('labels', torch.zeros(outputs['logits'].size(0))))
                total_loss += loss.item()
        
        return total_loss / len(self.eval_dataloader)
    
    def _save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(
            self.args.output_dir,
            f"checkpoint-epoch-{self.epoch + 1}"
        )
        print(f"\nSaving checkpoint to {checkpoint_dir}")
        self.model.save_pretrained(checkpoint_dir)

