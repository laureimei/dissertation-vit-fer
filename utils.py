import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def should_stop(self, val_loss: float, model: Optional[torch.nn.Module] = None) -> bool:
        """
        Check if training should stop
        
        Args:
            val_loss: Current validation loss
            model: Model to save best weights (optional)
            
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            if model is not None:
                self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and model is not None and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info("Restored best weights")
            return True
        
        return False
    
    def __call__(self, val_loss: float, model: Optional[torch.nn.Module] = None) -> bool:
        """Allow calling the object directly"""
        return self.should_stop(val_loss, model)


class MetricsTracker:
    """Track training and validation metrics over epochs"""
    
    def __init__(self):
        self.history = {
            'train': {'loss': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': []},
            'val': {'loss': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
        }
    
    def update(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Update metrics for current epoch"""
        for metric in ['loss', 'accuracy', 'f1', 'precision', 'recall']:
            self.history['train'][metric].append(train_metrics[metric])
            self.history['val'][metric].append(val_metrics[metric])
    
    def plot_metrics(self, save_dir: str):
        """Plot training history"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics = ['loss', 'accuracy', 'f1', 'precision', 'recall']
        
        for i, metric in enumerate(metrics):
            axes[i].plot(self.history['train'][metric], label=f'Train {metric}', marker='o')
            axes[i].plot(self.history['val'][metric], label=f'Val {metric}', marker='s')
            axes[i].set_title(f'{metric.capitalize()} Over Epochs')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].legend()
            axes[i].grid(True)
        
        # Hide the last subplot
        axes[-1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_best_epoch(self, metric: str = 'f1') -> int:
        """Get epoch with best validation metric"""
        return np.argmax(self.history['val'][metric])
    
    def get_history(self) -> Dict:
        """Get training history"""
        return self.history
    
    def save_history(self, save_path: str):
        """Save training history to JSON"""
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)


class PerformanceAnalyzer:
    """Analyze model performance with detailed metrics"""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
    
    def generate_classification_report(
        self, 
        y_true: List[int], 
        y_pred: List[int], 
        save_path: Optional[str] = None
    ) -> str:
        """Generate detailed classification report"""
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_names, 
            digits=4
        )
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def plot_confusion_matrix(
        self, 
        y_true: List[int], 
        y_pred: List[int], 
        save_path: str,
        normalize: bool = False
    ):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
        )
        
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_per_class_metrics(
        self, 
        y_true: List[int], 
        y_pred: List[int], 
        save_path: str
    ):
        """Plot per-class precision, recall, and F1-score"""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width, precision_per_class, width, label='Precision', alpha=0.8)
        ax.bar(x, recall_per_class, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_per_class, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Emotion Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
            ax.text(i - width, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(i, r + 0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width, f + 0.01, f'{f:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_detailed_analysis(
        self, 
        y_true: List[int], 
        y_pred: List[int], 
        save_dir: str
    ) -> Dict[str, Any]:
        """Generate comprehensive performance analysis"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # Class-wise performance
        class_performance = {}
        for i, class_name in enumerate(self.class_names):
            class_performance[class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i])
            }
        
        # Generate plots
        self.plot_confusion_matrix(y_true, y_pred, os.path.join(save_dir, 'confusion_matrix.png'))
        self.plot_confusion_matrix(y_true, y_pred, os.path.join(save_dir, 'confusion_matrix_normalized.png'), normalize=True)
        self.plot_per_class_metrics(y_true, y_pred, os.path.join(save_dir, 'per_class_metrics.png'))
        
        # Generate classification report
        report = self.generate_classification_report(y_true, y_pred, os.path.join(save_dir, 'classification_report.txt'))
        
        # Compile results
        analysis = {
            'overall_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'class_performance': class_performance,
            'classification_report': report
        }
        
        # Save analysis
        with open(os.path.join(save_dir, 'detailed_analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    metrics: Dict,
    filepath: str
):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict:
    """Load training checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Checkpoint loaded: {filepath}")
    return checkpoint


def calculate_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """Calculate model size and parameters"""
    param_size = 0
    param_count = 0
    
    for param in model.parameters():
        param_count += param.numel()
        param_size += param.numel() * param.element_size()
    
    buffer_size = 0
    buffer_count = 0
    
    for buffer in model.buffers():
        buffer_count += buffer.numel()
        buffer_size += buffer.numel() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return {
        'param_count': param_count,
        'param_size_mb': param_size / 1024**2,
        'buffer_count': buffer_count,
        'buffer_size_mb': buffer_size / 1024**2,
        'total_size_mb': size_all_mb
    }


def setup_logging(log_dir: str, log_level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('fer_training')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create handlers
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create experiment directory structure"""
    exp_dir = os.path.join(base_dir, experiment_name)
    
    # Create subdirectories
    subdirs = ['checkpoints', 'logs', 'plots', 'results']
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
    
    return exp_dir


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params
    }


if __name__ == "__main__":
    # Example usage
    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Test performance analyzer
    analyzer = PerformanceAnalyzer(class_names)
    
    # Generate dummy data for testing
    np.random.seed(42)
    y_true = np.random.randint(0, 7, 1000)
    y_pred = np.random.randint(0, 7, 1000)
    
    # Generate analysis
    os.makedirs('test_analysis', exist_ok=True)
    analysis = analyzer.generate_detailed_analysis(y_true.tolist(), y_pred.tolist(), 'test_analysis')
    
    print("Performance analysis completed!")
    print(f"Overall accuracy: {analysis['overall_metrics']['accuracy']:.4f}")
    print(f"Overall F1-score: {analysis['overall_metrics']['f1_score']:.4f}")