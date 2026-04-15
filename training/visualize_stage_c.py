#!/usr/bin/env python3
"""
Visualize Stage C training history and save graphs.
Usage:
    python training/visualize_stage_c.py --run-dir artifacts/phase9_decoder_finetune/stageC_v2
    python training/visualize_stage_c.py --run-dir artifacts/phase9_decoder_finetune/stageC_v3_retry1
"""
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np


def load_history(history_path):
    """Load history.json from a Stage C run."""
    with open(history_path, 'r') as f:
        return json.load(f)


def find_best_epochs(history):
    """Find epoch numbers for best BLEU and best loss."""
    best_bleu_epoch = max(range(len(history)), key=lambda i: history[i]['bleu'])
    best_loss_epoch = min(range(len(history)), key=lambda i: history[i]['val_ce'])
    return best_bleu_epoch, best_loss_epoch


def create_loss_plot(history, output_dir):
    """Plot training and validation CE loss."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = [h['epoch'] for h in history]
    train_ce = [h['train_ce'] for h in history]
    val_ce = [h['val_ce'] for h in history]
    
    ax.plot(epochs, train_ce, 'o-', linewidth=2, markersize=4, label='Train CE', color='#2E86AB')
    ax.plot(epochs, val_ce, 's-', linewidth=2, markersize=4, label='Val CE', color='#A23B72')
    
    best_loss_epoch = min(range(len(history)), key=lambda i: history[i]['val_ce'])
    ax.axvline(best_loss_epoch + 1, color='red', linestyle='--', alpha=0.5, label=f'Best Val CE (Epoch {best_loss_epoch + 1})')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cross-Entropy Loss', fontsize=12, fontweight='bold')
    ax.set_title('Stage C: Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_bleu_rouge_plot(history, output_dir):
    """Plot BLEU and ROUGE-L scores."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = [h['epoch'] for h in history]
    bleu = [h['bleu'] for h in history]
    rouge_l = [h['rouge_l'] for h in history]
    
    ax.plot(epochs, bleu, 'o-', linewidth=2, markersize=4, label='BLEU', color='#06A77D')
    ax.plot(epochs, rouge_l, 's-', linewidth=2, markersize=4, label='ROUGE-L', color='#F18F01')
    
    best_bleu_epoch = max(range(len(history)), key=lambda i: history[i]['bleu'])
    ax.axvline(best_bleu_epoch + 1, color='green', linestyle='--', alpha=0.5, label=f'Best BLEU (Epoch {best_bleu_epoch + 1})')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Stage C: BLEU and ROUGE-L Scores', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_dominance_plot(history, output_dir):
    """Plot dominant ratio and generic ratio."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = [h['epoch'] for h in history]
    dominant = [h['dominant_ratio'] for h in history]
    generic = [h['generic_ratio'] for h in history]
    
    ax.fill_between(epochs, dominant, alpha=0.3, label='Dominant Ratio', color='#E63946')
    ax.plot(epochs, dominant, 'o-', linewidth=2, markersize=4, color='#E63946')
    
    ax.fill_between(epochs, generic, alpha=0.3, label='Generic Ratio', color='#FF9500')
    ax.plot(epochs, generic, 's-', linewidth=2, markersize=4, color='#FF9500')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Stage C: Repetition & Genericity Analysis', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(max(dominant), max(generic)) * 1.1])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_wpp_aux_plot(history, output_dir):
    """Plot WPP auxiliary loss and weight schedule."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = [h['epoch'] for h in history]
    train_wpp = [h['train_wpp_aux'] for h in history]
    val_wpp = [h['val_wpp_aux'] for h in history]
    aux_weight = [h['aux_weight'] for h in history]
    
    # Left: WPP losses
    ax1.plot(epochs, train_wpp, 'o-', linewidth=2, markersize=4, label='Train WPP Aux', color='#457B9D')
    ax1.plot(epochs, val_wpp, 's-', linewidth=2, markersize=4, label='Val WPP Aux', color='#1D3557')
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('WPP Auxiliary Loss', fontsize=11, fontweight='bold')
    ax1.set_title('WPP Auxiliary Loss Over Time', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right: aux weight schedule
    ax2.plot(epochs, aux_weight, 'D-', linewidth=2, markersize=4, color='#D62828')
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('WPP Auxiliary Weight', fontsize=11, fontweight='bold')
    ax2.set_title('WPP Weight Decay Schedule', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_learning_rate_plot(history, output_dir):
    """Plot encoder and decoder learning rates."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = [h['epoch'] for h in history]
    enc_lr = [h['enc_lr'] for h in history]
    dec_lr = [h['dec_lr'] for h in history]
    
    ax.plot(epochs, enc_lr, 'o-', linewidth=2, markersize=4, label='Encoder LR', color='#1F77B4')
    ax.plot(epochs, dec_lr, 's-', linewidth=2, markersize=4, label='Decoder LR', color='#FF7F0E')
    
    # Highlight decoder unfreeze point (epoch 9)
    if len(history) >= 8:
        ax.axvline(9, color='green', linestyle='--', alpha=0.5, label='Decoder Unfrozen (Epoch 9)')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title('Stage C: Learning Rate Schedules', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_yscale('log')
    plt.tight_layout()
    return fig


def create_summary_plot(history, output_dir):
    """Create a comprehensive 2x2 summary plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Stage C V2 Training Summary (Full Dataset: 103K samples)', fontsize=14, fontweight='bold', y=0.98)
    
    epochs = [h['epoch'] for h in history]
    
    # Top-left: Loss
    train_ce = [h['train_ce'] for h in history]
    val_ce = [h['val_ce'] for h in history]
    axes[0, 0].plot(epochs, train_ce, 'o-', linewidth=1.5, markersize=3, label='Train', color='#2E86AB')
    axes[0, 0].plot(epochs, val_ce, 's-', linewidth=1.5, markersize=3, label='Val', color='#A23B72')
    axes[0, 0].set_ylabel('CE Loss', fontsize=10, fontweight='bold')
    axes[0, 0].set_title('Loss', fontsize=11, fontweight='bold')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Top-right: Metrics
    bleu = [h['bleu'] for h in history]
    rouge_l = [h['rouge_l'] for h in history]
    ax2 = axes[0, 1]
    color = '#06A77D'
    line1 = ax2.plot(epochs, bleu, 'o-', linewidth=1.5, markersize=3, color=color, label='BLEU')
    ax2.set_ylabel('BLEU', fontsize=10, fontweight='bold', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2_twin = ax2.twinx()
    line2 = ax2_twin.plot(epochs, rouge_l, 's-', linewidth=1.5, markersize=3, color='#F18F01', label='ROUGE-L')
    ax2_twin.set_ylabel('ROUGE-L', fontsize=10, fontweight='bold', color='#F18F01')
    ax2_twin.tick_params(axis='y', labelcolor='#F18F01')
    axes[0, 1].set_title('Translation Quality', fontsize=11, fontweight='bold')
    # Combine legends from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bottom-left: Dominance & Genericity
    dominant = [h['dominant_ratio'] for h in history]
    generic = [h['generic_ratio'] for h in history]
    axes[1, 0].plot(epochs, dominant, 'o-', linewidth=1.5, markersize=3, label='Dominant', color='#E63946')
    axes[1, 0].plot(epochs, generic, 's-', linewidth=1.5, markersize=3, label='Generic', color='#FF9500')
    axes[1, 0].set_xlabel('Epoch', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Ratio', fontsize=10, fontweight='bold')
    axes[1, 0].set_title('Decoder Pathology', fontsize=11, fontweight='bold')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bottom-right: Learning Rates + WPP weight
    enc_lr = [h['enc_lr'] for h in history]
    dec_lr = [h['dec_lr'] for h in history]
    aux_weight = [h['aux_weight'] for h in history]
    ax4 = axes[1, 1]
    line3 = ax4.plot(epochs, enc_lr, 'o-', linewidth=1.5, markersize=3, label='Enc LR', color='#1F77B4')
    line4 = ax4.plot(epochs, dec_lr, 's-', linewidth=1.5, markersize=3, label='Dec LR', color='#FF7F0E')
    ax4.set_xlabel('Epoch', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Learning Rate', fontsize=10, fontweight='bold')
    ax4.set_yscale('log')
    ax4_twin = ax4.twinx()
    line5 = ax4_twin.plot(epochs, aux_weight, '^-', linewidth=1.5, markersize=3, label='WPP weight', color='#D62828')
    ax4_twin.set_ylabel('WPP Aux Weight', fontsize=10, fontweight='bold', color='#D62828')
    ax4_twin.tick_params(axis='y', labelcolor='#D62828')
    axes[1, 1].set_title('Schedules', fontsize=11, fontweight='bold')
    # Combine legends from both axes
    lines_all = line3 + line4 + line5
    labels_all = [l.get_label() for l in lines_all]
    ax4.legend(lines_all, labels_all, loc='center right', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return fig


def create_training_stats_text(history, output_dir):
    """Generate a text summary of training statistics."""
    best_bleu_epoch = max(range(len(history)), key=lambda i: history[i]['bleu'])
    best_loss_epoch = min(range(len(history)), key=lambda i: history[i]['val_ce'])
    
    best_bleu_val = history[best_bleu_epoch]['bleu']
    best_loss_val = history[best_loss_epoch]['val_ce']
    
    final_bleu = history[-1]['bleu']
    final_loss = history[-1]['val_ce']
    
    avg_dominant = np.mean([h['dominant_ratio'] for h in history])
    avg_generic = np.mean([h['generic_ratio'] for h in history])
    
    stats = f"""
================================================================================
              STAGE C V2 TRAINING SUMMARY (FULL DATASET)
================================================================================

DATASET:
  Training Samples:   103,276 (full dataset)
  Validation Samples: 12,240
  Cache Entries:      127,237

OVERVIEW:
  Total Epochs Trained: {len(history)}
  Training Duration: {sum(h['elapsed_s'] for h in history) / 3600:.1f} hours ({sum(h['elapsed_s'] for h in history) / 60:.0f} minutes)

LOSS METRICS:
  Best Validation CE:   {best_loss_val:.5f} (Epoch {best_loss_epoch + 1}/{len(history)})
  Final Validation CE:  {final_loss:.5f}
  Training CE Trend:    {history[0]['train_ce']:.4f} → {history[-1]['train_ce']:.4f}

TRANSLATION QUALITY:
  Best BLEU Score:   {best_bleu_val:.3f} (Epoch {best_bleu_epoch + 1}/{len(history)})
  Final BLEU Score:  {final_bleu:.3f}
  Best ROUGE-L:      {max(h['rouge_l'] for h in history):.2f}
  Final ROUGE-L:     {history[-1]['rouge_l']:.2f}

DECODER PATHOLOGY (avg over all epochs):
  Dominant Ratio:    {avg_dominant:.3f}  (target: minimize)
  Generic Ratio:     {avg_generic:.3f}  (target: minimize)

AUXILIARY LOSS (WPP):
  Initial Weight:    {history[0]['aux_weight']:.4f}
  Final Weight:      {history[-1]['aux_weight']:.4f}
  Initial Train Aux: {history[0]['train_wpp_aux']:.5f}
  Final Train Aux:   {history[-1]['train_wpp_aux']:.5f}

LEARNING RATES (initial → final):
  Encoder LR:        {history[0]['enc_lr']:.2e} → {history[-1]['enc_lr']:.2e}
  Decoder LR:        {history[0]['dec_lr']:.2e} → {history[-1]['dec_lr']:.2e}

KEY EVENTS:
  Decoder Frozen:    Epochs 1-8 (WPP aux only)
  Decoder Unfrozen:  Epoch 9+ (top-1 block unfrozen)
  Early Stopping:    Triggered after epoch {len(history)} (patience: 8 epochs)

================================================================================
"""
    return stats.strip()


def main():
    parser = argparse.ArgumentParser(description='Visualize Stage C v2 training history')
    parser.add_argument('--run-dir', required=True, help='Path to Stage C run directory')
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    history_file = run_dir / 'history.json'
    
    if not history_file.exists():
        print(f"ERROR: history.json not found at {history_file}")
        return
    
    print(f"Loading history from {history_file}")
    history = load_history(history_file)
    print(f"Loaded {len(history)} epochs")
    
    # Create output directory for graphs
    graphs_dir = run_dir / 'graphs'
    graphs_dir.mkdir(exist_ok=True)
    
    # Generate all plots
    plots = {
        'loss.png': create_loss_plot,
        'bleu_rouge.png': create_bleu_rouge_plot,
        'dominance.png': create_dominance_plot,
        'wpp_auxiliary.png': create_wpp_aux_plot,
        'learning_rates.png': create_learning_rate_plot,
        'summary_4panel.png': create_summary_plot,
    }
    
    for filename, plot_func in plots.items():
        print(f"Creating {filename}...")
        fig = plot_func(history, graphs_dir)
        output_path = graphs_dir / filename
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  → Saved to {output_path}")
    
    # Generate text summary
    print("Generating training summary...")
    stats_text = create_training_stats_text(history, graphs_dir)
    stats_file = run_dir / 'training_summary.txt'
    with open(stats_file, 'w') as f:
        f.write(stats_text)
    print(f"  → Saved to {stats_file}")
    print(stats_text)
    
    print(f"\n✓ All visualizations saved to {graphs_dir}")


if __name__ == '__main__':
    main()
