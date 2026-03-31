"""
Comprehensive fairness evaluation with detailed metrics and visualizations.
Compares baseline vs mitigated models with confusion matrices, per-group analysis,
and visual reports.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json


class UTKFaceTestDataset(Dataset):
    """Test dataset with proper error handling."""
    
    def __init__(self, test_image_dir, test_labels_csv, transform=None):
        self.test_image_dir = test_image_dir
        self.transform = transform
        
        # Load labels
        df = pd.read_csv(test_labels_csv)
        
        self.race_map = {
            'Caucasian': 0,
            'African American': 1,
            'Asian': 2,
            'Indian': 3,
            'Others': 4
        }
        
        # Filter to only existing images
        self.samples = []
        for idx, row in df.iterrows():
            img_path = os.path.join(test_image_dir, row['filename'])
            if os.path.exists(img_path):
                self.samples.append({
                    'path': img_path,
                    'race_label': self.race_map[row['race']],
                    'race_name': row['race']
                })
        
        print(f"Loaded {len(self.samples)} test samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, sample['race_label'], sample['race_name']


class FairnessEvaluator:
    """Comprehensive fairness evaluation."""
    
    RACE_NAMES = ['Caucasian', 'African American', 'Asian', 'Indian', 'Others']
    
    def __init__(self, model_path, device='cuda', model_type='fair'):
        """
        Args:
            model_path: Path to model checkpoint
            device: 'cuda' or 'cpu'
            model_type: 'baseline' or 'fair' (determines architecture)
        """
        self.device = device
        self.model_type = model_type
        
        # Load model with correct architecture
        self.model = models.resnet18(pretrained=False)
        
        if model_type == 'fair':
            # Architecture used in train_fair_loss.py
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, 5)
            )
        else:
            # Baseline architecture
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 5)
        
        # Load weights
        try:
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict, strict=True)
            print(f"✓ Loaded {model_type} model from {model_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not load model strictly: {e}")
            print("Attempting flexible loading...")
            self.model.load_state_dict(state_dict, strict=False)
        
        self.model.to(device)
        self.model.eval()
    
    def evaluate(self, test_loader):
        """
        Comprehensive evaluation with detailed metrics.
        
        Returns:
            results: Dictionary with all metrics and predictions
        """
        all_preds = []
        all_labels = []
        all_race_names = []
        all_probs = []
        
        print("Running evaluation...")
        with torch.no_grad():
            for images, labels, races in test_loader:
                images = images.to(self.device)
                logits = self.model(images)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_race_names.extend(races)
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Overall metrics
        overall_acc = (all_preds == all_labels).mean()
        
        # Per-race metrics
        per_race_metrics = {}
        for race_id, race_name in enumerate(self.RACE_NAMES):
            mask = np.array([r == race_name for r in all_race_names])
            if mask.sum() > 0:
                race_preds = all_preds[mask]
                race_labels = all_labels[mask]
                race_probs = all_probs[mask]
                
                acc = (race_preds == race_labels).mean()
                
                # Confidence metrics
                pred_confidences = race_probs[np.arange(len(race_probs)), race_preds]
                avg_confidence = pred_confidences.mean()
                
                per_race_metrics[race_name] = {
                    'accuracy': float(acc),
                    'count': int(mask.sum()),
                    'avg_confidence': float(avg_confidence),
                    'correct': int((race_preds == race_labels).sum()),
                    'incorrect': int((race_preds != race_labels).sum())
                }
        
        # Fairness metrics
        accuracies = [m['accuracy'] for m in per_race_metrics.values()]
        accuracy_variance = np.var(accuracies)
        accuracy_std = np.std(accuracies)
        max_min_gap = max(accuracies) - min(accuracies)
        mean_accuracy = np.mean(accuracies)
        
        # Demographic parity (prediction distribution balance)
        pred_dist = {}
        for race_name in self.RACE_NAMES:
            race_count = sum(1 for r in all_race_names if r == race_name)
            if race_count > 0:
                pred_dist[race_name] = race_count / len(all_race_names)
        
        # Equalized odds (TPR and FPR per race)
        equalized_odds = {}
        for race_id, race_name in enumerate(self.RACE_NAMES):
            mask = np.array([r == race_name for r in all_race_names])
            if mask.sum() > 0:
                race_preds = all_preds[mask]
                race_labels = all_labels[mask]
                
                # True positive rate (sensitivity)
                tp = ((race_preds == race_id) & (race_labels == race_id)).sum()
                fn = ((race_preds != race_id) & (race_labels == race_id)).sum()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                # False positive rate
                fp = ((race_preds == race_id) & (race_labels != race_id)).sum()
                tn = ((race_preds != race_id) & (race_labels != race_id)).sum()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                equalized_odds[race_name] = {
                    'tpr': float(tpr),
                    'fpr': float(fpr)
                }
        
        return {
            'overall_accuracy': overall_acc,
            'per_race_metrics': per_race_metrics,
            'accuracy_variance': accuracy_variance,
            'accuracy_std': accuracy_std,
            'max_min_gap': max_min_gap,
            'mean_accuracy': mean_accuracy,
            'predictions': all_preds,
            'labels': all_labels,
            'race_names': all_race_names,
            'probabilities': all_probs,
            'demographic_parity': pred_dist,
            'equalized_odds': equalized_odds
        }


def plot_comparison(baseline_results, mitigated_results, output_dir='evaluation_results'):
    """Create comprehensive comparison visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    race_names = ['Caucasian', 'African American', 'Asian', 'Indian', 'Others']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Per-race accuracy comparison
    ax1 = fig.add_subplot(gs[0, 0])
    baseline_accs = [baseline_results['per_race_metrics'][r]['accuracy'] for r in race_names]
    mitigated_accs = [mitigated_results['per_race_metrics'][r]['accuracy'] for r in race_names]
    
    x = np.arange(len(race_names))
    width = 0.35
    ax1.bar(x - width/2, baseline_accs, width, label='Baseline', alpha=0.8)
    ax1.bar(x + width/2, mitigated_accs, width, label='Mitigated', alpha=0.8)
    ax1.set_xlabel('Race')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Per-Race Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(race_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Confusion matrix - Baseline
    ax2 = fig.add_subplot(gs[0, 1])
    cm_baseline = confusion_matrix(baseline_results['labels'], baseline_results['predictions'])
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=race_names, yticklabels=race_names)
    ax2.set_title('Baseline Confusion Matrix')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # 3. Confusion matrix - Mitigated
    ax3 = fig.add_subplot(gs[0, 2])
    cm_mitigated = confusion_matrix(mitigated_results['labels'], mitigated_results['predictions'])
    sns.heatmap(cm_mitigated, annot=True, fmt='d', cmap='Greens', ax=ax3,
                xticklabels=race_names, yticklabels=race_names)
    ax3.set_title('Mitigated Confusion Matrix')
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    
    # 4. Fairness metrics comparison
    ax4 = fig.add_subplot(gs[1, 0])
    metrics = ['Std Dev', 'Variance', 'Max-Min Gap']
    baseline_fair = [baseline_results['accuracy_std'], 
                     baseline_results['accuracy_variance'],
                     baseline_results['max_min_gap']]
    mitigated_fair = [mitigated_results['accuracy_std'],
                      mitigated_results['accuracy_variance'],
                      mitigated_results['max_min_gap']]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax4.bar(x - width/2, baseline_fair, width, label='Baseline', alpha=0.8)
    ax4.bar(x + width/2, mitigated_fair, width, label='Mitigated', alpha=0.8)
    ax4.set_ylabel('Value (Lower = Fairer)')
    ax4.set_title('Fairness Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Sample distribution
    ax5 = fig.add_subplot(gs[1, 1])
    baseline_counts = [baseline_results['per_race_metrics'][r]['count'] for r in race_names]
    ax5.bar(race_names, baseline_counts, alpha=0.8, color='skyblue')
    ax5.set_ylabel('Number of Samples')
    ax5.set_title('Test Set Distribution')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Prediction confidence comparison
    ax6 = fig.add_subplot(gs[1, 2])
    baseline_conf = [baseline_results['per_race_metrics'][r]['avg_confidence'] for r in race_names]
    mitigated_conf = [mitigated_results['per_race_metrics'][r]['avg_confidence'] for r in race_names]
    
    x = np.arange(len(race_names))
    width = 0.35
    ax6.bar(x - width/2, baseline_conf, width, label='Baseline', alpha=0.8)
    ax6.bar(x + width/2, mitigated_conf, width, label='Mitigated', alpha=0.8)
    ax6.set_ylabel('Average Confidence')
    ax6.set_title('Prediction Confidence by Race')
    ax6.set_xticks(x)
    ax6.set_xticklabels(race_names, rotation=45, ha='right')
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. TPR comparison (Equalized Odds)
    ax7 = fig.add_subplot(gs[2, 0])
    baseline_tpr = [baseline_results['equalized_odds'][r]['tpr'] for r in race_names]
    mitigated_tpr = [mitigated_results['equalized_odds'][r]['tpr'] for r in race_names]
    
    x = np.arange(len(race_names))
    width = 0.35
    ax7.bar(x - width/2, baseline_tpr, width, label='Baseline', alpha=0.8)
    ax7.bar(x + width/2, mitigated_tpr, width, label='Mitigated', alpha=0.8)
    ax7.set_ylabel('True Positive Rate')
    ax7.set_title('True Positive Rate by Race')
    ax7.set_xticks(x)
    ax7.set_xticklabels(race_names, rotation=45, ha='right')
    ax7.legend()
    ax7.grid(axis='y', alpha=0.3)
    ax7.axhline(y=np.mean(baseline_tpr), color='blue', linestyle='--', alpha=0.5)
    ax7.axhline(y=np.mean(mitigated_tpr), color='orange', linestyle='--', alpha=0.5)
    
    # 8. Accuracy improvement per race
    ax8 = fig.add_subplot(gs[2, 1])
    improvements = [mitigated_accs[i] - baseline_accs[i] for i in range(len(race_names))]
    colors = ['green' if x > 0 else 'red' for x in improvements]
    ax8.bar(race_names, improvements, color=colors, alpha=0.8)
    ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax8.set_ylabel('Accuracy Change')
    ax8.set_title('Per-Race Accuracy Change (Mitigated - Baseline)')
    ax8.tick_params(axis='x', rotation=45)
    ax8.grid(axis='y', alpha=0.3)
    
    # 9. Overall metrics summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_text = f"""
OVERALL SUMMARY

Baseline:
  • Mean Accuracy: {baseline_results['mean_accuracy']:.4f}
  • Std Dev: {baseline_results['accuracy_std']:.4f}
  • Max-Min Gap: {baseline_results['max_min_gap']:.4f}

Mitigated:
  • Mean Accuracy: {mitigated_results['mean_accuracy']:.4f}
  • Std Dev: {mitigated_results['accuracy_std']:.4f}
  • Max-Min Gap: {mitigated_results['max_min_gap']:.4f}

Changes:
  • Accuracy: {(mitigated_results['mean_accuracy'] - baseline_results['mean_accuracy']):.4f}
  • Std Dev: {(mitigated_results['accuracy_std'] - baseline_results['accuracy_std']):.4f}
  • Gap: {(mitigated_results['max_min_gap'] - baseline_results['max_min_gap']):.4f}

Fairness Status:
  {"✓ IMPROVED" if mitigated_results['accuracy_std'] < baseline_results['accuracy_std'] else "✗ DEGRADED"}
    """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Comprehensive Fairness Evaluation: Baseline vs Mitigated', 
                 fontsize=16, fontweight='bold')
    
    plt.savefig(f'{output_dir}/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved comprehensive comparison to {output_dir}/comprehensive_comparison.png")
    plt.close()


def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    return obj

def generate_detailed_report(baseline_results, mitigated_results, output_dir='evaluation_results'):
    """Generate detailed text report and JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    race_names = ['Caucasian', 'African American', 'Asian', 'Indian', 'Others']
    
    # Console report
    print("\n" + "="*100)
    print("COMPREHENSIVE FAIRNESS EVALUATION REPORT")
    print("="*100)
    
    print("\n📊 PER-RACE ACCURACY ANALYSIS")
    print("-" * 100)
    print(f"{'Race':<25} {'Baseline Acc':>15} {'Mitigated Acc':>15} {'Change':>15} {'% Change':>15}")
    print("-" * 100)
    
    for race in race_names:
        base_acc = baseline_results['per_race_metrics'][race]['accuracy']
        mit_acc = mitigated_results['per_race_metrics'][race]['accuracy']
        change = mit_acc - base_acc
        pct_change = (change / base_acc * 100) if base_acc > 0 else float('inf')
        
        print(f"{race:<25} {base_acc:>15.4f} {mit_acc:>15.4f} {change:>+15.4f} {pct_change:>+14.2f}%")
    
    print("\n📈 FAIRNESS METRICS (Lower = Better)")
    print("-" * 100)
    print(f"{'Metric':<35} {'Baseline':>15} {'Mitigated':>15} {'Change':>15} {'Status':>15}")
    print("-" * 100)
    
    metrics = [
        ('Accuracy Std Deviation', 'accuracy_std'),
        ('Accuracy Variance', 'accuracy_variance'),
        ('Max-Min Accuracy Gap', 'max_min_gap'),
    ]
    
    for metric_name, metric_key in metrics:
        base_val = baseline_results[metric_key]
        mit_val = mitigated_results[metric_key]
        change = mit_val - base_val
        status = "✓ Better" if change < 0 else "✗ Worse"
        
        print(f"{metric_name:<35} {base_val:>15.6f} {mit_val:>15.6f} {change:>+15.6f} {status:>15}")
    
    print("\n🎯 UTILITY METRICS (Higher = Better)")
    print("-" * 100)
    print(f"{'Metric':<35} {'Baseline':>15} {'Mitigated':>15} {'Change':>15} {'Status':>15}")
    print("-" * 100)
    
    base_mean = baseline_results['mean_accuracy']
    mit_mean = mitigated_results['mean_accuracy']
    mean_change = mit_mean - base_mean
    mean_status = "✓ Better" if mean_change > 0 else "✗ Worse"
    
    print(f"{'Mean Accuracy':<35} {base_mean:>15.4f} {mit_mean:>15.4f} {mean_change:>+15.4f} {mean_status:>15}")
    
    print("\n🔍 PREDICTION DISTRIBUTION")
    print("-" * 100)
    print(f"{'Race':<25} {'Baseline Samples':>20} {'Mitigated Samples':>20}")
    print("-" * 100)
    
    for race in race_names:
        base_count = baseline_results['per_race_metrics'][race]['count']
        mit_count = mitigated_results['per_race_metrics'][race]['count']
        print(f"{race:<25} {base_count:>20} {mit_count:>20}")
    
    print("\n" + "="*100)
    print("FINAL VERDICT")
    print("="*100)
    
    fairness_improved = mitigated_results['accuracy_std'] < baseline_results['accuracy_std']
    utility_maintained = mit_mean >= base_mean * 0.95  # Within 5% of baseline
    
    if fairness_improved and utility_maintained:
        print("✓ SUCCESS: Fairness improved while maintaining utility")
    elif fairness_improved and not utility_maintained:
        print("⚠ PARTIAL SUCCESS: Fairness improved but significant utility loss (>5%)")
    elif not fairness_improved and utility_maintained:
        print("✗ FAILURE: Fairness degraded despite maintaining utility")
    else:
        print("✗ CRITICAL FAILURE: Both fairness and utility degraded")
    
    print("="*100 + "\n")
    
    # Save JSON report
    report = {
        'baseline': {k: convert_to_serializable(v) 
                    for k, v in baseline_results.items() 
                    if k not in ['predictions', 'labels', 'race_names', 'probabilities']},
        'mitigated': {k: convert_to_serializable(v) 
                     for k, v in mitigated_results.items()
                     if k not in ['predictions', 'labels', 'race_names', 'probabilities']},
        'verdict': {
            'fairness_improved': bool(fairness_improved),
            'utility_maintained': bool(utility_maintained),
            'std_reduction': float(baseline_results['accuracy_std'] - mitigated_results['accuracy_std']),
            'gap_reduction': float(baseline_results['max_min_gap'] - mitigated_results['max_min_gap']),
            'mean_accuracy_change': float(mean_change)
        }
    }
    
    with open(f'{output_dir}/evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Saved detailed report to {output_dir}/evaluation_report.json")
    
    print(f"✓ Detailed report saved to {output_dir}/evaluation_report.json")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Fairness Evaluation')
    parser.add_argument('--baseline', type=str, required=True, 
                       help='Path to baseline model')
    parser.add_argument('--mitigated', type=str, required=True,
                       help='Path to mitigated model')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Test image directory')
    parser.add_argument('--test_csv', type=str, required=True,
                       help='Test labels CSV')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--baseline_type', type=str, default='baseline',
                       choices=['baseline', 'fair'],
                       help='Architecture type for baseline model')
    parser.add_argument('--mitigated_type', type=str, default='fair',
                       choices=['baseline', 'fair'],
                       help='Architecture type for mitigated model')
    
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load test data
    print("Loading test dataset...")
    test_dataset = UTKFaceTestDataset(args.test_dir, args.test_csv, transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=0)
    
    # Evaluate baseline
    print("\nEvaluating baseline model...")
    baseline_eval = FairnessEvaluator(args.baseline, device, model_type=args.baseline_type)
    baseline_results = baseline_eval.evaluate(test_loader)
    
    # Evaluate mitigated
    print("\nEvaluating mitigated model...")
    mitigated_eval = FairnessEvaluator(args.mitigated, device, model_type=args.mitigated_type)
    mitigated_results = mitigated_eval.evaluate(test_loader)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_comparison(baseline_results, mitigated_results, args.output_dir)
    
    # Generate detailed report
    generate_detailed_report(baseline_results, mitigated_results, args.output_dir)
    
    print(f"\n✓ Evaluation complete! Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
