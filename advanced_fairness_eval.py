"""
COMPREHENSIVE FAIRNESS EVALUATION FRAMEWORK
============================================

Production-grade evaluation script for bias mitigation in face recognition.

Includes:
1. Equalized Odds metrics (FPR/TPR parity)
2. Demographic Parity (prediction rates)
3. Per-race precision/recall/F1
4. Calibration analysis
5. ROC-AUC per group
6. Fairness visualizations
7. Statistical significance testing

References:
- Hardt et al. (2016): Equality of Opportunity
- Buolamwini & Gebru (2018): Gender Shades
- Mitchell et al. (2019): Model Cards for Model Reporting
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from scipy.stats import chi2_contingency, fisher_exact
import warnings
warnings.filterwarnings('ignore')


class ComprehensiveFairnessEvaluator:
    """
    Complete fairness evaluation suite for multi-class race recognition.
    
    Computes all major fairness metrics and generates publication-quality visualizations.
    """
    
    RACE_NAMES = ['Caucasian', 'African American', 'Asian', 'Indian', 'Others']
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def __init__(self, output_dir='fairness_evaluation'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def load_model_predictions(self, model_path, test_loader, device='cuda'):
        """Load model and get predictions on test set."""
        import torchvision.models as models
        
        # Create model architecture
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 5)
        )
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        all_logits = []
        all_preds = []
        all_targets = []
        all_races = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch[0].to(device)
                labels = batch[1]
                races = batch[2]
                
                logits = model(images)
                preds = logits.argmax(dim=1)
                
                all_logits.append(logits.cpu())
                all_preds.append(preds.cpu())
                all_targets.append(labels)
                all_races.append(races)
        
        logits = torch.cat(all_logits, dim=0).numpy()
        preds = torch.cat(all_preds, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        races = torch.cat(all_races, dim=0).numpy()
        
        return logits, preds, targets, races
    
    def compute_equalized_odds(self, preds, targets, races):
        """
        Compute Equalized Odds metrics.
        
        Equalized Odds: FPR and TPR should be equal across demographic groups.
        
        Returns:
            Dict with FPR/TPR per race and parity metrics
        """
        metrics = {
            'fpr': {},
            'fnr': {},
            'tpr': {},
            'tnr': {},
            'fpr_parity': None,
            'fnr_parity': None,
            'tpr_parity': None,
        }
        
        fprs = []
        fnrs = []
        tprs = []
        
        for race_id, race_name in enumerate(self.RACE_NAMES):
            # Get samples for this race
            race_mask = (races == race_id)
            race_preds = preds[race_mask]
            race_targets = targets[race_mask]
            
            if race_mask.sum() == 0:
                continue
            
            # For multi-class, compute "vs all" metrics
            # This race is positive class, all others are negative
            is_positive = (race_targets == race_id)
            predicted_positive = (race_preds == race_id)
            
            # TP: correctly predicted this race
            tp = ((predicted_positive) & (is_positive)).sum()
            # FP: incorrectly predicted this race
            fp = ((predicted_positive) & (~is_positive)).sum()
            # TN: correctly rejected this race
            tn = ((~predicted_positive) & (~is_positive)).sum()
            # FN: missed this race
            fn = ((~predicted_positive) & (is_positive)).sum()
            
            # Compute rates
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics['fpr'][race_name] = float(fpr)
            metrics['fnr'][race_name] = float(fnr)
            metrics['tpr'][race_name] = float(tpr)
            metrics['tnr'][race_name] = float(tnr)
            
            fprs.append(fpr)
            fnrs.append(fnr)
            tprs.append(tpr)
        
        # Parity: how much do rates vary across groups?
        if len(fprs) > 1:
            metrics['fpr_parity'] = float(np.std(fprs))  # Lower is more fair
            metrics['fnr_parity'] = float(np.std(fnrs))
            metrics['tpr_parity'] = float(np.std(tprs))
        
        return metrics
    
    def compute_demographic_parity(self, preds, races):
        """
        Compute Demographic Parity.
        
        DP: Proportion of positive predictions should be equal across groups.
        
        Returns:
            Dict with prediction rates per race
        """
        metrics = {
            'positive_prediction_rate': {},
            'prediction_rate_parity': None
        }
        
        pred_rates = []
        
        for race_id, race_name in enumerate(self.RACE_NAMES):
            race_mask = (races == race_id)
            if race_mask.sum() == 0:
                continue
            
            race_preds = preds[race_mask]
            # Proportion predicted as this race
            pred_rate = (race_preds == race_id).mean()
            metrics['positive_prediction_rate'][race_name] = float(pred_rate)
            pred_rates.append(pred_rate)
        
        if len(pred_rates) > 1:
            metrics['prediction_rate_parity'] = float(np.std(pred_rates))
        
        return metrics
    
    def compute_per_race_metrics(self, preds, targets, races):
        """
        Compute precision, recall, F1 per race.
        
        For multi-class: compute "race i vs all others" metrics.
        """
        metrics = {}
        
        for race_id, race_name in enumerate(self.RACE_NAMES):
            race_mask = (races == race_id)
            if race_mask.sum() == 0:
                continue
            
            race_preds = preds[race_mask]
            race_targets = targets[race_mask]
            
            # Binary classification: this race vs others
            is_positive = (race_targets == race_id).astype(int)
            predicted_positive = (race_preds == race_id).astype(int)
            
            # Compute metrics
            tp = ((predicted_positive == 1) & (is_positive == 1)).sum()
            fp = ((predicted_positive == 1) & (is_positive == 0)).sum()
            fn = ((predicted_positive == 0) & (is_positive == 1)).sum()
            tn = ((predicted_positive == 0) & (is_positive == 0)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            metrics[race_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'accuracy': float(accuracy),
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn),
                'support': int(race_mask.sum())
            }
        
        return metrics
    
    def compute_calibration(self, logits, targets, races):
        """
        Compute calibration metrics.
        
        Calibration: Is the model's confidence aligned with actual accuracy?
        """
        metrics = {}
        
        for race_id, race_name in enumerate(self.RACE_NAMES):
            race_mask = (races == race_id)
            if race_mask.sum() < 10:  # Need minimum samples
                continue
            
            race_logits = logits[race_mask]
            race_targets = targets[race_mask]
            
            # Softmax to get probabilities
            probs = np.exp(race_logits) / np.exp(race_logits).sum(axis=1, keepdims=True)
            max_probs = probs.max(axis=1)
            preds = probs.argmax(axis=1)
            
            # Accuracy per confidence bin
            correct = (preds == race_targets).astype(int)
            
            # Expected calibration error (ECE)
            bins = np.linspace(0, 1, 11)
            bin_accs = []
            bin_confs = []
            
            for i in range(len(bins) - 1):
                mask = (max_probs >= bins[i]) & (max_probs < bins[i+1])
                if mask.sum() > 0:
                    bin_acc = correct[mask].mean()
                    bin_conf = max_probs[mask].mean()
                    bin_accs.append(bin_acc)
                    bin_confs.append(bin_conf)
            
            if len(bin_accs) > 0:
                ece = np.mean(np.abs(np.array(bin_accs) - np.array(bin_confs)))
            else:
                ece = 0.0
            
            # Maximum calibration error
            mce = np.max(np.abs(np.array(bin_accs) - np.array(bin_confs))) if len(bin_accs) > 0 else 0
            
            metrics[race_name] = {
                'ece': float(ece),  # Expected Calibration Error
                'mce': float(mce),  # Maximum Calibration Error
                'mean_confidence': float(max_probs.mean()),
                'accuracy': float(correct.mean())
            }
        
        return metrics
    
    def compute_roc_auc(self, logits, targets, races):
        """
        Compute ROC-AUC for each race (one-vs-all).
        """
        metrics = {}
        
        for race_id, race_name in enumerate(self.RACE_NAMES):
            race_mask = (races == race_id)
            if race_mask.sum() < 10:
                continue
            
            race_logits = logits[race_mask]
            race_targets = targets[race_mask]
            
            # Binary: this race vs others
            y_true = (race_targets == race_id).astype(int)
            y_scores = race_logits[:, race_id]  # Logits for this class
            
            try:
                roc_auc = roc_auc_score(y_true, y_scores)
                ap = average_precision_score(y_true, y_scores)
            except:
                roc_auc = 0.0
                ap = 0.0
            
            metrics[race_name] = {
                'roc_auc': float(roc_auc),
                'average_precision': float(ap)
            }
        
        return metrics
    
    def compute_confusion_matrices(self, preds, targets, races):
        """
        Compute full confusion matrix and per-race confusion matrices.
        """
        matrices = {}
        
        # Overall confusion matrix
        matrices['overall'] = confusion_matrix(targets, preds, labels=range(5))
        
        # Per-race confusion matrices (one-vs-all)
        for race_id, race_name in enumerate(self.RACE_NAMES):
            race_mask = (races == race_id)
            if race_mask.sum() == 0:
                continue
            
            race_targets = targets[race_mask]
            race_preds = preds[race_mask]
            
            # Binary confusion matrix
            y_true = (race_targets == race_id).astype(int)
            y_pred = (race_preds == race_id).astype(int)
            
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            matrices[race_name] = cm
        
        return matrices
    
    def statistical_parity_test(self, preds, races):
        """
        Chi-square test for statistical parity.
        
        H0: Prediction rates are equal across groups
        """
        # Build contingency table
        contingency = []
        for race_id in range(5):
            race_mask = (races == race_id)
            if race_mask.sum() == 0:
                continue
            
            race_preds = preds[race_mask]
            pred_positive = (race_preds == race_id).sum()
            pred_negative = (race_preds != race_id).sum()
            
            contingency.append([pred_positive, pred_negative])
        
        contingency = np.array(contingency)
        
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            return {
                'chi2_statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'is_parity': bool(p_value > 0.05)
            }
        except:
            return {
                'chi2_statistic': None,
                'p_value': None,
                'degrees_of_freedom': None,
                'is_parity': None
            }
    
    def generate_visualizations(self, all_metrics):
        """Generate comprehensive fairness visualizations."""
        
        # 1. FPR/TPR per race (Equalized Odds)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        races = list(all_metrics['equalized_odds']['fpr'].keys())
        fprs = [all_metrics['equalized_odds']['fpr'][r] for r in races]
        tprs = [all_metrics['equalized_odds']['tpr'][r] for r in races]
        
        x = np.arange(len(races))
        width = 0.35
        
        axes[0].bar(x - width/2, fprs, width, label='FPR', color='#e74c3c')
        axes[0].bar(x + width/2, tprs, width, label='TPR', color='#3498db')
        axes[0].set_ylabel('Rate')
        axes[0].set_title('Equalized Odds: FPR/TPR per Race')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(races, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(alpha=0.3, axis='y')
        axes[0].axhline(y=np.mean(fprs), color='#e74c3c', linestyle='--', alpha=0.5, label='Mean FPR')
        axes[0].axhline(y=np.mean(tprs), color='#3498db', linestyle='--', alpha=0.5, label='Mean TPR')
        
        # 2. Prediction rates (Demographic Parity)
        pred_rates = [all_metrics['demographic_parity']['positive_prediction_rate'][r] for r in races]
        axes[1].bar(races, pred_rates, color=self.COLORS)
        axes[1].set_ylabel('Positive Prediction Rate')
        axes[1].set_title('Demographic Parity: Prediction Rates per Race')
        axes[1].set_xticklabels(races, rotation=45, ha='right')
        axes[1].grid(alpha=0.3, axis='y')
        axes[1].axhline(y=np.mean(pred_rates), color='red', linestyle='--', label='Mean')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fairness_metrics.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved fairness_metrics.png")
        plt.close()
        
        # 3. Per-race F1 scores
        fig, ax = plt.subplots(figsize=(10, 6))
        
        races = list(all_metrics['per_race_metrics'].keys())
        f1_scores = [all_metrics['per_race_metrics'][r]['f1'] for r in races]
        precisions = [all_metrics['per_race_metrics'][r]['precision'] for r in races]
        recalls = [all_metrics['per_race_metrics'][r]['recall'] for r in races]
        
        x = np.arange(len(races))
        width = 0.25
        
        ax.bar(x - width, precisions, width, label='Precision', color='#2ecc71')
        ax.bar(x, recalls, width, label='Recall', color='#f39c12')
        ax.bar(x + width, f1_scores, width, label='F1', color='#9b59b6')
        
        ax.set_ylabel('Score')
        ax.set_title('Per-Race Classification Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(races, rotation=45, ha='right')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_race_metrics.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved per_race_metrics.png")
        plt.close()
        
        # 4. Calibration per race
        fig, ax = plt.subplots(figsize=(10, 6))
        
        races = list(all_metrics['calibration'].keys())
        ecess = [all_metrics['calibration'][r]['ece'] for r in races]
        
        colors = ['#e74c3c' if ece > 0.1 else '#2ecc71' for ece in ecess]
        ax.barh(races, ecess, color=colors)
        ax.set_xlabel('Expected Calibration Error (ECE)')
        ax.set_title('Model Calibration per Race (Lower is Better)')
        ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='Warning threshold')
        ax.grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'calibration.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved calibration.png")
        plt.close()
        
        # 5. Confusion matrix heatmap (overall)
        fig, ax = plt.subplots(figsize=(8, 7))
        cm = all_metrics['confusion_matrices']['overall']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.RACE_NAMES,
                   yticklabels=self.RACE_NAMES,
                   ax=ax, cbar_kws={'label': 'Count'})
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Overall Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix_overall.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion_matrix_overall.png")
        plt.close()
    
    def evaluate(self, model_path, test_loader, device='cuda'):
        """
        Run complete fairness evaluation.
        """
        print(f"\n{'='*80}")
        print("COMPREHENSIVE FAIRNESS EVALUATION")
        print(f"{'='*80}\n")
        
        # Load predictions
        print("Loading model predictions...")
        logits, preds, targets, races = self.load_model_predictions(model_path, test_loader, device)
        
        # Compute all metrics
        print("Computing metrics...")
        all_metrics = {
            'equalized_odds': self.compute_equalized_odds(preds, targets, races),
            'demographic_parity': self.compute_demographic_parity(preds, races),
            'per_race_metrics': self.compute_per_race_metrics(preds, targets, races),
            'calibration': self.compute_calibration(logits, targets, races),
            'roc_auc': self.compute_roc_auc(logits, targets, races),
            'confusion_matrices': self.compute_confusion_matrices(preds, targets, races),
            'statistical_parity': self.statistical_parity_test(preds, races)
        }
        
        # Generate visualizations
        print("Generating visualizations...")
        self.generate_visualizations(all_metrics)
        
        # Save detailed report
        print("Saving report...")
        self._save_report(all_metrics)
        
        return all_metrics
    
    def _save_report(self, all_metrics):
        """Save comprehensive evaluation report."""
        report = {
            'equalized_odds': all_metrics['equalized_odds'],
            'demographic_parity': all_metrics['demographic_parity'],
            'per_race_metrics': all_metrics['per_race_metrics'],
            'calibration': all_metrics['calibration'],
            'roc_auc': all_metrics['roc_auc'],
            'statistical_parity': all_metrics['statistical_parity']
        }
        
        with open(self.output_dir / 'comprehensive_fairness_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Saved comprehensive_fairness_report.json")
        
        # Print summary
        self._print_summary(all_metrics)
    
    def _print_summary(self, metrics):
        """Print human-readable summary."""
        print(f"\n{'='*80}")
        print("FAIRNESS EVALUATION SUMMARY")
        print(f"{'='*80}\n")
        
        print("📊 EQUALIZED ODDS (Lower parity = More Fair)")
        print("-" * 80)
        print(f"FPR Parity (std): {metrics['equalized_odds']['fpr_parity']:.4f}")
        print(f"TPR Parity (std): {metrics['equalized_odds']['tpr_parity']:.4f}")
        print(f"FNR Parity (std): {metrics['equalized_odds']['fnr_parity']:.4f}")
        print()
        
        print("Per-Race FPR/TPR:")
        for race in self.RACE_NAMES:
            if race in metrics['equalized_odds']['fpr']:
                fpr = metrics['equalized_odds']['fpr'][race]
                tpr = metrics['equalized_odds']['tpr'][race]
                print(f"  {race:20s}: FPR={fpr:.4f}, TPR={tpr:.4f}")
        print()
        
        print("📊 DEMOGRAPHIC PARITY")
        print("-" * 80)
        print(f"Prediction Rate Parity: {metrics['demographic_parity']['prediction_rate_parity']:.4f}")
        print("\nPer-Race Positive Prediction Rates:")
        for race in self.RACE_NAMES:
            if race in metrics['demographic_parity']['positive_prediction_rate']:
                rate = metrics['demographic_parity']['positive_prediction_rate'][race]
                print(f"  {race:20s}: {rate:.4f}")
        print()
        
        print("📊 PER-RACE PERFORMANCE")
        print("-" * 80)
        for race in self.RACE_NAMES:
            if race in metrics['per_race_metrics']:
                m = metrics['per_race_metrics'][race]
                print(f"{race}:")
                print(f"  Precision: {m['precision']:.4f}")
                print(f"  Recall:    {m['recall']:.4f}")
                print(f"  F1:        {m['f1']:.4f}")
                print(f"  Accuracy:  {m['accuracy']:.4f}")
        print()
        
        print("📊 CALIBRATION (Lower = Better)")
        print("-" * 80)
        for race in self.RACE_NAMES:
            if race in metrics['calibration']:
                ece = metrics['calibration'][race]['ece']
                acc = metrics['calibration'][race]['accuracy']
                conf = metrics['calibration'][race]['mean_confidence']
                print(f"{race}:")
                print(f"  ECE:                {ece:.4f}")
                print(f"  Accuracy:           {acc:.4f}")
                print(f"  Mean Confidence:    {conf:.4f}")
        print()
        
        print("📊 ROC-AUC")
        print("-" * 80)
        for race in self.RACE_NAMES:
            if race in metrics['roc_auc']:
                roc = metrics['roc_auc'][race]['roc_auc']
                ap = metrics['roc_auc'][race]['average_precision']
                print(f"{race}: ROC-AUC={roc:.4f}, AP={ap:.4f}")
        print()
        
        print("📊 STATISTICAL PARITY TEST")
        print("-" * 80)
        sp = metrics['statistical_parity']
        print(f"Chi-square statistic: {sp['chi2_statistic']:.4f}")
        print(f"P-value:              {sp['p_value']:.4f}")
        print(f"Is Parity (p>0.05):   {sp['is_parity']}")
        print()


if __name__ == '__main__':
    """
    Usage:
    
    evaluator = ComprehensiveFairnessEvaluator(output_dir='fairness_evaluation')
    metrics = evaluator.evaluate(
        model_path='best_fair_model_fixed.pth',
        test_loader=test_loader,
        device='cuda'
    )
    """
    pass
