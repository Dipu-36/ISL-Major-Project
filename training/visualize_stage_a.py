#!/usr/bin/env python3
"""
Visualize Stage A training history and save graphs.
Usage:
    python training/visualize_stage_a.py --run-dir artifacts/phase9_encoder_pretrain/stageA_v3
"""
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_history(history_path):
    """Load history.json from a Stage A run."""
    with open(history_path, 'r') as f:
        return json.load(f)


def create_wpp_loss_plot(history, output_dir):
    """Plot WPP training and validation loss."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]

    ax.plot(epochs, train_loss, 'o-', linewidth=2, markersize=4, label='Train WPP Loss', color='#2E86AB')
    ax.plot(epochs, val_loss, 's-', linewidth=2, markersize=4, label='Val WPP Loss', color='#A23B72')

    best_epoch = min(range(len(history)), key=lambda i: history[i]['val_loss'])
    ax.axvline(best_epoch + 1, color='green', linestyle='--', alpha=0.5, label=f'Best Val Loss (Epoch {best_epoch + 1})')

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('WPP Loss', fontsize=12, fontweight='bold')
    ax.set_title('Stage A: Word Presence Prediction Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_auc_plot(history, output_dir):
    """Plot mean AUC over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = [h['epoch'] for h in history]
    mean_auc = [h.get('mean_auc', 0) for h in history]

    ax.plot(epochs, mean_auc, 'o-', linewidth=2, markersize=4, color='#06A77D')
    ax.axhline(0.62, color='red', linestyle='--', alpha=0.5, label='Gate Threshold (0.62)')

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean AUC', fontsize=12, fontweight='bold')
    ax.set_title('Stage A: Mean Word Presence AUC', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, min(0.8, max(mean_auc) * 1.1)])
    plt.tight_layout()
    return fig


def create_lr_plot(history, output_dir):
    """Plot learning rate schedule."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = [h['epoch'] for h in history]
    lr = [h['lr'] for h in history]

    ax.plot(epochs, lr, 'o-', linewidth=2, markersize=4, color='#F18F01')

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title('Stage A: Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    return fig


def create_summary_plot(history, output_dir):
    """Create comprehensive summary plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = [h['epoch'] for h in history]

    # Top-left: WPP Loss
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    axes[0, 0].plot(epochs, train_loss, 'o-', linewidth=1.5, markersize=3, label='Train', color='#2E86AB')
    axes[0, 0].plot(epochs, val_loss, 's-', linewidth=1.5, markersize=3, label='Val', color='#A23B72')
    axes[0, 0].set_ylabel('WPP Loss', fontsize=10, fontweight='bold')
    axes[0, 0].set_title('Word Presence Prediction Loss', fontsize=11, fontweight='bold')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: AUC
    mean_auc = [h.get('mean_auc', 0) for h in history]
    axes[0, 1].plot(epochs, mean_auc, 'o-', linewidth=1.5, markersize=3, color='#06A77D')
    axes[0, 1].axhline(0.62, color='red', linestyle='--', alpha=0.5, label='Gate (0.62)')
    axes[0, 1].set_ylabel('Mean AUC', fontsize=10, fontweight='bold')
    axes[0, 1].set_title('Word Presence AUC', fontsize=11, fontweight='bold')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0.5, 0.8])

    # Bottom-left: Words Above Threshold
    words_above = [h.get('words_above_threshold', 0) for h in history]
    axes[1, 0].plot(epochs, words_above, 'o-', linewidth=1.5, markersize=3, color='#E63946')
    axes[1, 0].axhline(25, color='green', linestyle='--', alpha=0.5, label='Gate (25 words)')
    axes[1, 0].set_xlabel('Epoch', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Words Above Threshold', fontsize=10, fontweight='bold')
    axes[1, 0].set_title('Gate Progress', fontsize=11, fontweight='bold')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Learning Rate
    lr = [h['lr'] for h in history]
    axes[1, 1].plot(epochs, lr, 'o-', linewidth=1.5, markersize=3, color='#F18F01')
    axes[1, 1].set_xlabel('Epoch', fontsize=10, fontweight='bold')
    axes[1, 1].set_ylabel('Learning Rate', fontsize=10, fontweight='bold')
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, which='both')
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    return fig


def create_training_summary(history, output_dir):
    """Generate text summary of training."""
    best_epoch = min(range(len(history)), key=lambda i: history[i]['val_loss'])
    best_loss = history[best_epoch]['val_loss']
    final_auc = history[-1].get('mean_auc', 0)
    words_above = history[-1].get('words_above_threshold', 0)

    summary = f"""
================================================================================
                    STAGE A V3 TRAINING SUMMARY
================================================================================

OVERVIEW:
  Total Epochs Trained: {len(history)}
  Best Validation Loss: {best_loss:.5f} (Epoch {best_epoch + 1})
  Final Mean AUC:       {final_auc:.4f}
  Words Above Gate:     {words_above}/150 (threshold: 25)

GATE STATUS:
  {'✓ PASSED' if words_above >= 25 and final_auc >= 0.62 else '✗ FAILED'}

METRICS PROGRESS:
  Train Loss: {history[0]['train_loss']:.4f} → {history[-1]['train_loss']:.4f}
  Val Loss:   {history[0]['val_loss']:.4f} → {history[-1]['val_loss']:.4f}

================================================================================
"""
    return summary.strip()


def main():
    parser = argparse.ArgumentParser(description='Visualize Stage A v3 training history')
    parser.add_argument('--run-dir', required=True, help='Path to Stage A run directory')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    history_file = run_dir / 'history.json'

    if not history_file.exists():
        print(f"ERROR: history.json not found at {history_file}")
        return

    print(f"Loading history from {history_file}")
    history = load_history(history_file)
    print(f"Loaded {len(history)} epochs")

    graphs_dir = run_dir / 'graphs'
    graphs_dir.mkdir(exist_ok=True)

    plots = {
        'wpp_loss.png': create_wpp_loss_plot,
        'auc.png': create_auc_plot,
        'learning_rate.png': create_lr_plot,
        'summary_4panel.png': create_summary_plot,
    }

    for filename, plot_func in plots.items():
        print(f"Creating {filename}...")
        fig = plot_func(history, graphs_dir)
        output_path = graphs_dir / filename
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  → Saved to {output_path}")

    print("Generating training summary...")
    summary = create_training_summary(history, graphs_dir)
    summary_file = run_dir / 'training_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(summary)
    print(f"  → Saved to {summary_file}")
    print(summary)

    print(f"\n✓ All visualizations saved to {graphs_dir}")


if __name__ == '__main__':
    main()
