import torch
import os
import logging

# Set up logging for callback activities
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Callback:
    """
    A base class for all callbacks. Other callbacks should inherit from this class.
    """
    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        pass

    def on_train_end(self, logs: dict) -> None:
        pass


class EarlyStopping(Callback):
    """
    Stops training if the validation loss does not improve for a specified number of epochs.
    """
    def __init__(self, patience: int = 10, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.wait = 0

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        val_loss = logs.get('val_loss')
        if val_loss is None:
            return  # No validation loss to monitor
        
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            logging.info(f"Early stopping at epoch {epoch} due to no improvement in validation loss for {self.patience} epochs.")
            raise StopIteration("Early stopping triggered")


class ModelCheckpoint(Callback):
    """
    Saves the model if its performance improves (based on validation loss or accuracy).
    """
    def __init__(self, save_path: str, monitor: str = 'val_loss', mode: str = 'min'):
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        metric = logs.get(self.monitor)
        if metric is None:
            return  # No metric to monitor
        
        if self.mode == 'min' and metric < self.best_metric:
            self.best_metric = metric
            self.save_model(epoch, logs)
        elif self.mode == 'max' and metric > self.best_metric:
            self.best_metric = metric
            self.save_model(epoch, logs)

    def save_model(self, epoch: int, logs: dict) -> None:
        logging.info(f"Saving model at epoch {epoch} with {self.monitor}: {logs[self.monitor]:.4f}")
        torch.save(logs['model_state_dict'], os.path.join(self.save_path, f"model_epoch_{epoch}.pth"))
    

class LearningRateScheduler(Callback):
    """
    Adjusts the learning rate based on training progress or validation performance.
    """
    def __init__(self, scheduler_fn):
        self.scheduler_fn = scheduler_fn

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        if 'optimizer' in logs:
            scheduler = self.scheduler_fn(logs['optimizer'], epoch)
            scheduler.step()


def get_callbacks() -> list:
    """
    Returns a list of callbacks to be used during training.
    You can easily add new callbacks here.
    """
    return [
        EarlyStopping(patience=10, delta=0.01),  # Early stopping with patience of 10 epochs
        ModelCheckpoint(save_path='./checkpoints', monitor='val_loss', mode='min'),  # Save model with best val_loss
    ]

