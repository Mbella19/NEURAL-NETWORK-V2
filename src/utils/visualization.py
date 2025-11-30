"""
Training visualization utilities.

Features:
- Training curves (loss, accuracy, learning rate)
- Prediction vs target plots
- Confusion matrices for direction
- Distribution plots
- Memory and timing charts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime


class TrainingVisualizer:
    """
    Comprehensive training visualization.

    Creates detailed plots for monitoring training progress.
    """

    def __init__(
        self,
        save_dir: Optional[str] = None,
        style: str = 'seaborn-v0_8-darkgrid'
    ):
        """
        Args:
            save_dir: Directory to save plots
            style: Matplotlib style
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Try to set style, fall back to default if not available
        try:
            plt.style.use(style)
        except OSError:
            try:
                plt.style.use('seaborn-darkgrid')
            except OSError:
                pass  # Use default style

        # Color scheme
        self.colors = {
            'train': '#2E86AB',      # Blue
            'val': '#E94F37',         # Red
            'accent': '#F39C12',      # Orange
            'positive': '#27AE60',    # Green
            'negative': '#C0392B',    # Dark Red
            'neutral': '#95A5A6'      # Gray
        }

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        title: str = "Training Progress",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comprehensive training curves.

        Args:
            history: Dictionary with keys like 'train_loss', 'val_loss', etc.
            title: Plot title
            save_name: Filename to save plot

        Returns:
            matplotlib Figure
        """
        # Determine which metrics we have
        has_loss = 'train_loss' in history and 'val_loss' in history
        has_acc = 'train_acc' in history and 'val_acc' in history
        has_dir_acc = 'train_direction_acc' in history and 'val_direction_acc' in history
        has_lr = 'learning_rate' in history
        has_grad = 'grad_norm' in history
        has_memory = 'memory_mb' in history

        # Calculate number of subplots needed
        n_plots = sum([has_loss, has_acc, has_dir_acc, has_lr, has_grad, has_memory])
        if n_plots == 0:
            n_plots = 1

        # Create figure
        fig, axes = plt.subplots(
            (n_plots + 1) // 2, 2,
            figsize=(14, 4 * ((n_plots + 1) // 2))
        )
        if n_plots == 1:
            axes = np.array([[axes, axes]])
        axes = axes.flatten()

        plot_idx = 0
        epochs = range(1, len(history.get('train_loss', history.get('val_loss', [0]))) + 1)

        # Loss plot
        if has_loss:
            ax = axes[plot_idx]
            ax.plot(epochs, history['train_loss'], label='Train Loss',
                   color=self.colors['train'], linewidth=2)
            ax.plot(epochs, history['val_loss'], label='Val Loss',
                   color=self.colors['val'], linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Mark best validation loss
            best_epoch = np.argmin(history['val_loss']) + 1
            best_loss = min(history['val_loss'])
            ax.axvline(x=best_epoch, color=self.colors['accent'],
                      linestyle='--', alpha=0.7, label=f'Best: {best_loss:.6f}')
            ax.scatter([best_epoch], [best_loss], color=self.colors['accent'],
                      s=100, zorder=5, marker='*')
            plot_idx += 1

        # Accuracy plot
        if has_acc:
            ax = axes[plot_idx]
            ax.plot(epochs, [a * 100 for a in history['train_acc']],
                   label='Train Acc', color=self.colors['train'], linewidth=2)
            ax.plot(epochs, [a * 100 for a in history['val_acc']],
                   label='Val Acc', color=self.colors['val'], linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Training and Validation Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 100])
            plot_idx += 1

        # Direction accuracy plot
        if has_dir_acc:
            ax = axes[plot_idx]
            ax.plot(epochs, [a * 100 for a in history['train_direction_acc']],
                   label='Train Dir Acc', color=self.colors['train'], linewidth=2)
            ax.plot(epochs, [a * 100 for a in history['val_direction_acc']],
                   label='Val Dir Acc', color=self.colors['val'], linewidth=2)
            ax.axhline(y=50, color=self.colors['neutral'], linestyle='--',
                      alpha=0.7, label='Random (50%)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Direction Accuracy (%)')
            ax.set_title('Direction Prediction Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 100])
            plot_idx += 1

        # Learning rate plot
        if has_lr:
            ax = axes[plot_idx]
            ax.plot(epochs, history['learning_rate'],
                   color=self.colors['accent'], linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # Gradient norm plot
        if has_grad:
            ax = axes[plot_idx]
            ax.plot(epochs, history['grad_norm'],
                   color=self.colors['accent'], linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Norm (Stability Indicator)')
            ax.grid(True, alpha=0.3)
            # Add warning threshold
            ax.axhline(y=10.0, color=self.colors['negative'], linestyle='--',
                      alpha=0.7, label='Warning Threshold')
            ax.legend()
            plot_idx += 1

        # Memory usage plot
        if has_memory:
            ax = axes[plot_idx]
            ax.plot(epochs, history['memory_mb'],
                   color=self.colors['train'], linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Memory (MB)')
            ax.set_title('Memory Usage Over Training')
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')

        return fig

    def plot_predictions_vs_targets(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        title: str = "Predictions vs Targets",
        save_name: Optional[str] = None,
        sample_size: int = 1000
    ) -> plt.Figure:
        """
        Plot predictions against actual targets.

        Args:
            predictions: Model predictions
            targets: Actual values
            title: Plot title
            save_name: Filename to save
            sample_size: Number of points to plot

        Returns:
            matplotlib Figure
        """
        predictions = predictions.flatten()
        targets = targets.flatten()

        # Sample if too many points
        if len(predictions) > sample_size:
            indices = np.random.choice(len(predictions), sample_size, replace=False)
            predictions = predictions[indices]
            targets = targets[indices]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Scatter plot
        ax = axes[0]
        colors = np.where((predictions > 0) == (targets > 0),
                         self.colors['positive'], self.colors['negative'])
        ax.scatter(targets, predictions, c=colors, alpha=0.5, s=20)
        ax.plot([targets.min(), targets.max()],
                [targets.min(), targets.max()],
                'k--', linewidth=2, label='Perfect')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Scatter: Predicted vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Residual plot
        ax = axes[1]
        residuals = predictions - targets
        ax.scatter(targets, residuals, alpha=0.5, s=20, color=self.colors['train'])
        ax.axhline(y=0, color='k', linestyle='--', linewidth=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Residual (Pred - Actual)')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)

        # Distribution comparison
        ax = axes[2]
        ax.hist(targets, bins=50, alpha=0.5, label='Actual',
               color=self.colors['train'], density=True)
        ax.hist(predictions, bins=50, alpha=0.5, label='Predicted',
               color=self.colors['val'], density=True)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')

        return fig

    def plot_direction_confusion(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        title: str = "Direction Classification",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot direction confusion matrix.

        Args:
            predictions: Model predictions
            targets: Actual values
            title: Plot title
            save_name: Filename to save

        Returns:
            matplotlib Figure
        """
        predictions = predictions.flatten()
        targets = targets.flatten()

        # Classify directions
        pred_up = predictions > 0
        pred_down = predictions <= 0
        actual_up = targets > 0
        actual_down = targets <= 0

        # Confusion matrix
        tp_up = (pred_up & actual_up).sum()
        fp_up = (pred_up & actual_down).sum()
        fn_up = (pred_down & actual_up).sum()
        tn_up = (pred_down & actual_down).sum()

        confusion = np.array([
            [tp_up, fp_up],
            [fn_up, tn_up]
        ])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Confusion matrix heatmap
        ax = axes[0]
        im = ax.imshow(confusion, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred Up', 'Pred Down'])
        ax.set_yticklabels(['Actual Up', 'Actual Down'])
        ax.set_title('Confusion Matrix')

        # Annotate
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, f'{confusion[i, j]}\n({confusion[i,j]/confusion.sum()*100:.1f}%)',
                              ha='center', va='center', fontsize=12)

        plt.colorbar(im, ax=ax)

        # Metrics bar chart
        ax = axes[1]
        total = confusion.sum()
        accuracy = (tp_up + tn_up) / total if total > 0 else 0
        precision = tp_up / (tp_up + fp_up) if (tp_up + fp_up) > 0 else 0
        recall = tp_up / (tp_up + fn_up) if (tp_up + fn_up) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [accuracy, precision, recall, f1]
        colors = [self.colors['train'], self.colors['val'],
                 self.colors['accent'], self.colors['positive']]

        bars = ax.bar(metrics, values, color=colors)
        ax.set_ylim([0, 1])
        ax.set_ylabel('Score')
        ax.set_title('Classification Metrics')
        ax.axhline(y=0.5, color=self.colors['neutral'], linestyle='--',
                  alpha=0.7, label='Random Baseline')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        ax.legend()

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')

        return fig

    def plot_epoch_summary(
        self,
        epoch: int,
        train_predictions: np.ndarray,
        train_targets: np.ndarray,
        val_predictions: np.ndarray,
        val_targets: np.ndarray,
        metrics: Dict[str, float],
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive epoch summary visualization.

        Args:
            epoch: Current epoch number
            train_predictions: Training predictions
            train_targets: Training targets
            val_predictions: Validation predictions
            val_targets: Validation targets
            metrics: Dictionary of metrics
            save_name: Filename to save

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig)

        # Train scatter
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(train_targets.flatten(), train_predictions.flatten(),
                   alpha=0.3, s=10, color=self.colors['train'])
        ax1.plot([train_targets.min(), train_targets.max()],
                [train_targets.min(), train_targets.max()], 'k--')
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title('Train: Predictions vs Targets')

        # Val scatter
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(val_targets.flatten(), val_predictions.flatten(),
                   alpha=0.3, s=10, color=self.colors['val'])
        ax2.plot([val_targets.min(), val_targets.max()],
                [val_targets.min(), val_targets.max()], 'k--')
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.set_title('Val: Predictions vs Targets')

        # Prediction distributions
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(train_predictions.flatten(), bins=50, alpha=0.5,
                label='Train', color=self.colors['train'], density=True)
        ax3.hist(val_predictions.flatten(), bins=50, alpha=0.5,
                label='Val', color=self.colors['val'], density=True)
        ax3.set_xlabel('Prediction Value')
        ax3.set_ylabel('Density')
        ax3.set_title('Prediction Distributions')
        ax3.legend()

        # Residuals
        ax4 = fig.add_subplot(gs[1, 0])
        train_residuals = train_predictions.flatten() - train_targets.flatten()
        ax4.hist(train_residuals, bins=50, color=self.colors['train'], alpha=0.7)
        ax4.axvline(x=0, color='k', linestyle='--')
        ax4.set_xlabel('Residual')
        ax4.set_ylabel('Count')
        ax4.set_title(f'Train Residuals (mean={train_residuals.mean():.6f})')

        ax5 = fig.add_subplot(gs[1, 1])
        val_residuals = val_predictions.flatten() - val_targets.flatten()
        ax5.hist(val_residuals, bins=50, color=self.colors['val'], alpha=0.7)
        ax5.axvline(x=0, color='k', linestyle='--')
        ax5.set_xlabel('Residual')
        ax5.set_ylabel('Count')
        ax5.set_title(f'Val Residuals (mean={val_residuals.mean():.6f})')

        # Metrics summary
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        metrics_text = f"Epoch {epoch} Metrics\n" + "=" * 30 + "\n"
        for key, value in metrics.items():
            if isinstance(value, float):
                metrics_text += f"{key}: {value:.6f}\n"
            else:
                metrics_text += f"{key}: {value}\n"
        ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Direction accuracy over time (sample)
        ax7 = fig.add_subplot(gs[2, :])
        sample_size = min(200, len(val_targets))
        indices = np.linspace(0, len(val_targets)-1, sample_size, dtype=int)

        correct = ((val_predictions[indices] > 0) == (val_targets[indices] > 0)).flatten()
        colors = np.where(correct, self.colors['positive'], self.colors['negative'])

        ax7.bar(range(len(indices)), val_predictions[indices].flatten(),
               color=colors, alpha=0.7, width=1.0)
        ax7.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax7.set_xlabel('Sample Index')
        ax7.set_ylabel('Prediction')
        ax7.set_title('Sample Predictions (Green=Correct Direction, Red=Wrong)')

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        plt.suptitle(f'Epoch {epoch} Summary - {timestamp}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')

        return fig

    def plot_learning_dynamics(
        self,
        batch_losses: List[float],
        batch_grad_norms: List[float],
        title: str = "Learning Dynamics",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot batch-level learning dynamics.

        Args:
            batch_losses: List of batch losses
            batch_grad_norms: List of gradient norms
            title: Plot title
            save_name: Filename to save

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        batches = range(1, len(batch_losses) + 1)

        # Raw loss
        ax = axes[0, 0]
        ax.plot(batches, batch_losses, alpha=0.5, color=self.colors['train'])
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title('Batch Loss (Raw)')
        ax.grid(True, alpha=0.3)

        # Smoothed loss
        ax = axes[0, 1]
        window = min(50, len(batch_losses) // 10 + 1)
        smoothed = np.convolve(batch_losses, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(batch_losses)+1), smoothed,
               color=self.colors['train'], linewidth=2)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Batch Loss (Smoothed, window={window})')
        ax.grid(True, alpha=0.3)

        # Gradient norms
        ax = axes[1, 0]
        ax.plot(batches, batch_grad_norms, alpha=0.5, color=self.colors['accent'])
        ax.axhline(y=1.0, color=self.colors['positive'], linestyle='--',
                  alpha=0.7, label='Ideal')
        ax.axhline(y=10.0, color=self.colors['negative'], linestyle='--',
                  alpha=0.7, label='Warning')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Batch Gradient Norms')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Loss vs gradient norm scatter
        ax = axes[1, 1]
        ax.scatter(batch_grad_norms, batch_losses, alpha=0.3,
                  s=10, color=self.colors['train'])
        ax.set_xlabel('Gradient Norm')
        ax.set_ylabel('Loss')
        ax.set_title('Loss vs Gradient Norm')
        ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')

        return fig

    def close_all(self):
        """Close all matplotlib figures to free memory."""
        plt.close('all')


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Convenience function to plot training history.

    Args:
        history: Training history dictionary
        save_path: Path to save figure
        show: Whether to display the plot

    Returns:
        matplotlib Figure
    """
    visualizer = TrainingVisualizer()
    fig = visualizer.plot_training_curves(
        history,
        title="Training History",
        save_name=None
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig
