#!/usr/bin/env python3
"""
Compare multiple Stage C runs side-by-side.
Usage:
    python training/compare_stage_c_runs.py \
        --run1 artifacts/phase9_decoder_finetune/stageC_v2 \
        --run2 artifacts/phase9_decoder_finetune/stageC_v3_retry1
"""
import json
import argparse
from pathlib import Path
import numpy as np


def load_history(run_dir):
    """Load history from a Stage C run directory."""
    history_file = Path(run_dir) / 'history.json'
    if not history_file.exists():
        raise FileNotFoundError(f"history.json not found in {run_dir}")
    
    with open(history_file, 'r') as f:
        return json.load(f)


def compute_metrics(history):
    """Compute summary metrics from history."""
    best_bleu_idx = max(range(len(history)), key=lambda i: history[i]['bleu'])
    best_loss_idx = min(range(len(history)), key=lambda i: history[i]['val_ce'])
    
    return {
        'total_epochs': len(history),
        'total_time_h': sum(h['elapsed_s'] for h in history) / 3600,
        'best_bleu': history[best_bleu_idx]['bleu'],
        'best_bleu_epoch': best_bleu_idx + 1,
        'final_bleu': history[-1]['bleu'],
        'best_val_ce': history[best_loss_idx]['val_ce'],
        'best_val_ce_epoch': best_loss_idx + 1,
        'final_val_ce': history[-1]['val_ce'],
        'best_rouge_l': max(h['rouge_l'] for h in history),
        'final_rouge_l': history[-1]['rouge_l'],
        'avg_dominance': np.mean([h['dominant_ratio'] for h in history]),
        'avg_genericity': np.mean([h['generic_ratio'] for h in history]),
        'init_enc_lr': history[0]['enc_lr'],
        'final_enc_lr': history[-1]['enc_lr'],
        'init_dec_lr': history[0]['dec_lr'],
        'final_dec_lr': history[-1]['dec_lr'],
        'init_aux_weight': history[0]['aux_weight'],
        'final_aux_weight': history[-1]['aux_weight'],
    }


def format_comparison(run_names, metrics_list):
    """Format metrics for side-by-side comparison."""
    output = "\n" + "="*100 + "\n"
    output += f"{'COMPARISON':^100}\n"
    output += "="*100 + "\n\n"
    
    # Header
    header = f"{'Metric':<40} {run_names[0]:<25} {run_names[1]:<25}"
    output += header + "\n"
    output += "-" * 100 + "\n"
    
    # OVERVIEW
    output += "OVERVIEW:\n"
    for name, label in [('total_epochs', 'Total Epochs'), ('total_time_h', 'Training Time (h)')]:
        v1, v2 = metrics_list[0][name], metrics_list[1][name]
        v1_f = f"{v1:.1f}" if isinstance(v1, float) else f"{int(v1)}"
        v2_f = f"{v2:.1f}" if isinstance(v2, float) else f"{int(v2)}"
        better = "✓" if (name == 'total_epochs' and v1 > v2) or (name != 'total_epochs' and v1 < v2) else ""
        output += f"  {label:<38} {v1_f:>23}  {v2_f:>23} {better}\n"
    output += "\n"
    
    # BLEU
    output += "TRANSLATION QUALITY (BLEU):\n"
    for name, label in [
        ('best_bleu', 'Peak BLEU Score'),
        ('best_bleu_epoch', ' └─ at Epoch'),
        ('final_bleu', 'Final BLEU'),
    ]:
        v1, v2 = metrics_list[0][name], metrics_list[1][name]
        v1_f = f"{v1:.4f}" if isinstance(v1, float) else f"{int(v1)}"
        v2_f = f"{v2:.4f}" if isinstance(v2, float) else f"{int(v2)}"
        better = "✓" if v1 > v2 else ""
        output += f"  {label:<38} {v1_f:>23}  {v2_f:>23} {better}\n"
    
    for name, label in [
        ('best_rouge_l', 'Peak ROUGE-L'),
        ('final_rouge_l', 'Final ROUGE-L'),
    ]:
        v1, v2 = metrics_list[0][name], metrics_list[1][name]
        v1_f = f"{v1:.2f}" if isinstance(v1, float) else f"{int(v1)}"
        v2_f = f"{v2:.2f}" if isinstance(v2, float) else f"{int(v2)}"
        better = "✓" if v1 > v2 else ""
        output += f"  {label:<38} {v1_f:>23}  {v2_f:>23} {better}\n"
    output += "\n"
    
    # LOSS
    output += "CONVERGENCE (VALIDATION CE):\n"
    for name, label in [
        ('best_val_ce', 'Best Val CE'),
        ('best_val_ce_epoch', ' └─ at Epoch'),
        ('final_val_ce', 'Final Val CE'),
    ]:
        v1, v2 = metrics_list[0][name], metrics_list[1][name]
        v1_f = f"{v1:.5f}" if isinstance(v1, float) else f"{int(v1)}"
        v2_f = f"{v2:.5f}" if isinstance(v2, float) else f"{int(v2)}"
        better = "✓" if v1 < v2 else ""
        output += f"  {label:<38} {v1_f:>23}  {v2_f:>23} {better}\n"
    output += "\n"
    
    # PATHOLOGY
    output += "DECODER PATHOLOGY (avg over all epochs):\n"
    for name, label in [
        ('avg_dominance', 'Dominant Ratio'),
        ('avg_genericity', 'Generic Ratio'),
    ]:
        v1, v2 = metrics_list[0][name], metrics_list[1][name]
        v1_f = f"{v1:.4f}"
        v2_f = f"{v2:.4f}"
        better = "✓" if v1 < v2 else ""
        output += f"  {label:<38} {v1_f:>23}  {v2_f:>23} {better}\n"
    output += "\n"
    
    # LEARNING RATES
    output += "LEARNING RATES (initial → final):\n"
    for name, init_label, final_label in [
        ('enc_lr', 'Encoder LR init', 'Encoder LR final'),
        ('dec_lr', 'Decoder LR init', 'Decoder LR final'),
    ]:
        for suffix, label in [('init', init_label), ('final', final_label)]:
            key = f"{suffix}_{name}"
            v1, v2 = metrics_list[0][key], metrics_list[1][key]
            v1_f = f"{v1:.2e}"
            v2_f = f"{v2:.2e}"
            output += f"  {label:<38} {v1_f:>23}  {v2_f:>23}\n"
        output += "\n"
    
    # AUX WEIGHT
    output += "WPP AUXILIARY (initial → final):\n"
    for name, label in [
        ('init_aux_weight', 'WPP Weight init'),
        ('final_aux_weight', 'WPP Weight final'),
    ]:
        v1, v2 = metrics_list[0][name], metrics_list[1][name]
        v1_f = f"{v1:.4f}"
        v2_f = f"{v2:.4f}"
        output += f"  {label:<38} {v1_f:>23}  {v2_f:>23}\n"
    output += "\n"
    
    output += "="*100 + "\n\n"
    output += "Legend:\n"
    output += "  ✓  = Better value for this run\n"
    output += "  For BLEU/ROUGE-L/Epochs: higher is better\n"
    output += "  For CE/Dominance/Genericity: lower is better\n"
    
    return output


def main():
    parser = argparse.ArgumentParser(description='Compare two Stage C runs')
    parser.add_argument('--run1', required=True, help='First Stage C run directory')
    parser.add_argument('--run2', required=True, help='Second Stage C run directory')
    args = parser.parse_args()
    
    run_names = [Path(args.run1).name, Path(args.run2).name]
    
    print(f"Loading {run_names[0]}...")
    history1 = load_history(args.run1)
    metrics1 = compute_metrics(history1)
    
    print(f"Loading {run_names[1]}...")
    history2 = load_history(args.run2)
    metrics2 = compute_metrics(history2)
    
    comparison = format_comparison(run_names, [metrics1, metrics2])
    print(comparison)
    
    # Save comparison to file
    output_file = Path(args.run2).parent / 'run_comparison.txt'
    with open(output_file, 'w') as f:
        f.write(comparison)
    print(f"Comparison saved to {output_file}")


if __name__ == '__main__':
    main()
