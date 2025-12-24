"""
Training callbacks for monitoring and control.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import os


class TrainingCallback(ABC):
    """
    Base class for training callbacks.
    """
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Called at the end of each epoch."""
        pass
    
    def on_train_begin(self, logs: Dict[str, Any]):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, logs: Dict[str, Any]):
        """Called at the end of training."""
        pass


class EarlyStoppingCallback(TrainingCallback):
    """
    Stop training when a monitored metric has stopped improving.
    
    Parameters
    ----------
    monitor : str, optional (default: 'eval_loss')
        Metric to monitor.
    patience : int, optional (default: 3)
        Number of epochs with no improvement after which training will be stopped.
    min_delta : float, optional (default: 0.0)
        Minimum change in the monitored quantity to qualify as an improvement.
    mode : str, optional (default: 'min')
        One of 'min' or 'max'. In 'min' mode, training will stop when the quantity
        monitored has stopped decreasing; in 'max' mode it will stop when the
        quantity monitored has stopped increasing.
    """
    
    def __init__(
        self,
        monitor: str = 'eval_loss',
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Check if training should stop."""
        current_value = logs.get(self.monitor)
        
        if current_value is None:
            return
        
        if self.mode == 'min':
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                return True  # Signal to stop training
        
        return False


class CheckpointCallback(TrainingCallback):
    """
    Save model checkpoints during training.
    
    Parameters
    ----------
    save_dir : str
        Directory to save checkpoints.
    monitor : str, optional (default: 'eval_loss')
        Metric to monitor for best model.
    mode : str, optional (default: 'min')
        One of 'min' or 'max'.
    save_best_only : bool, optional (default: True)
        If True, only save when the monitored metric improves.
    """
    
    def __init__(
        self,
        save_dir: str,
        monitor: str = 'eval_loss',
        mode: str = 'min',
        save_best_only: bool = True
    ):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
        os.makedirs(save_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Save checkpoint if conditions are met."""
        current_value = logs.get(self.monitor)
        
        if current_value is None:
            return
        
        should_save = False
        
        if self.save_best_only:
            if self.mode == 'min':
                improved = current_value < self.best_value
            else:
                improved = current_value > self.best_value
            
            if improved:
                self.best_value = current_value
                should_save = True
        else:
            should_save = True
        
        if should_save:
            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            # Save logic would go here
            print(f"Saved checkpoint to {checkpoint_path}")

