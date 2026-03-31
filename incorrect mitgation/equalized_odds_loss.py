"""
Equalized Odds Loss for Fair Face Recognition

Implements equalized odds fairness constraint as described in:
- Hardt et al. (2016): "Equality of Opportunity in Supervised Learning"
- Kotwal & Marcel (2025): "Face Recognition in the Age of Synthetic Faces"

Key innovation: Minimize False Positive Rate (FPR) variance across demographic groups
instead of accuracy variance, which is more principled for fairness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EqualizedOddsLoss(nn.Module):
    """
    Computes Equalized Odds fairness loss.
    
    Penalizes difference in False Positive Rates (FPR) across demographic groups.
    
    Formula:
        total_loss = CE_loss + lambda * Var(FPR_0, FPR_1, ..., FPR_k)
    
    Where:
        FPR_i = False Positives for class i / (FP_i + TN_i)
        False Positive = Predicted class i but true class != i
        True Negative = Predicted class != i AND true class != i
    
    Args:
        num_races (int): Number of demographic groups (5 for UTKFace)
        lambda_fairness (float): Weight for fairness penalty (0.05-0.5 recommended)
        use_fpr (bool): If True, minimize FPR variance. If False, minimize TPR variance.
        device (str): 'cuda' or 'cpu'
    """
    
    def __init__(self, num_races=5, lambda_fairness=0.1, use_fpr=True, device='cuda'):
        super().__init__()
        self.num_races = num_races
        self.lambda_fairness = lambda_fairness
        self.use_fpr = use_fpr
        self.device = device
        
        # Track running statistics for numerical stability
        self.register_buffer('ema_fprs', torch.ones(num_races) * 0.5)
        self.ema_alpha = 0.1
    
    def compute_fpr_for_race(self, preds, targets, race_id):
        """
        Compute False Positive Rate for a specific race class.
        
        FPR = FP / (FP + TN)
        where:
          FP = predicted class race_id but true class != race_id
          TN = predicted class != race_id AND true class != race_id
        
        Args:
            preds (torch.Tensor): Predicted class indices (batch_size,)
            targets (torch.Tensor): True class indices (batch_size,)
            race_id (int): Race class to compute FPR for
        
        Returns:
            float: FPR for this race (0 to 1), or 0 if no samples
        """
        # Samples NOT of this race
        non_race_mask = (targets != race_id)
        
        if non_race_mask.sum() == 0:
            return torch.tensor(0.0, device=preds.device)
        
        non_race_samples = preds[non_race_mask]
        non_race_targets = targets[non_race_mask]
        
        # False Positives: predicted as race_id but true != race_id
        false_positives = (non_race_samples == race_id).float().sum()
        
        # True Negatives: predicted != race_id AND true != race_id
        true_negatives = (non_race_samples != race_id).float().sum()
        
        # FPR = FP / (FP + TN)
        denominator = false_positives + true_negatives
        
        if denominator == 0:
            return torch.tensor(0.0, device=preds.device)
        
        fpr = false_positives / denominator
        return fpr
    
    def compute_tpr_for_race(self, preds, targets, race_id):
        """
        Compute True Positive Rate (Recall) for a specific race class.
        
        TPR = TP / (TP + FN)
        where:
          TP = predicted class race_id AND true class == race_id
          FN = predicted class != race_id but true class == race_id
        
        This is useful for ensuring equal recall across groups.
        """
        # Samples OF this race
        race_mask = (targets == race_id)
        
        if race_mask.sum() == 0:
            return torch.tensor(0.0, device=preds.device)
        
        race_preds = preds[race_mask]
        race_targets = targets[race_mask]
        
        # True Positives
        true_positives = (race_preds == race_id).float().sum()
        
        # False Negatives
        false_negatives = (race_preds != race_id).float().sum()
        
        # TPR = TP / (TP + FN)
        denominator = true_positives + false_negatives
        
        if denominator == 0:
            return torch.tensor(0.0, device=preds.device)
        
        tpr = true_positives / denominator
        return tpr
    
    def forward(self, logits, targets, race_labels, sample_weights=None):
        """
        Compute Equalized Odds loss.
        
        Args:
            logits (torch.Tensor): Model outputs (batch_size, num_races)
            targets (torch.Tensor): True race labels (batch_size,)
            race_labels (torch.Tensor): Race group for each sample (batch_size,) 
                                        [same as targets for race classification]
            sample_weights (torch.Tensor, optional): Per-sample weights (batch_size,)
        
        Returns:
            total_loss (torch.Tensor): Combined loss
            ce_loss (torch.Tensor): Cross-entropy component
            fairness_penalty (torch.Tensor): Equalized odds penalty component
            metrics (dict): Additional metrics for logging
        """
        
        # === STEP 1: Compute Standard CE Loss ===
        
        # Numerical stability: clamp logits
        logits = torch.clamp(logits, -100, 100)
        
        # CE loss
        ce_loss_unreduced = F.cross_entropy(logits, targets, reduction='none')
        
        # Apply sample weights if provided
        if sample_weights is not None:
            # Ensure weights are normalized
            sample_weights = sample_weights / sample_weights.mean()
            ce_loss = (ce_loss_unreduced * sample_weights).mean()
        else:
            ce_loss = ce_loss_unreduced.mean()
        
        # === STEP 2: Compute Equalized Odds (FPR/TPR Variance) ===
        
        with torch.no_grad():
            preds = logits.argmax(dim=1)
        
        # Compute FPR or TPR for each race
        error_rates = []
        
        for race_id in range(self.num_races):
            if self.use_fpr:
                error_rate = self.compute_fpr_for_race(preds, targets, race_id)
            else:
                error_rate = self.compute_tpr_for_race(preds, targets, race_id)
            
            error_rates.append(error_rate)
        
        error_rates = torch.stack(error_rates)
        
        # Update EMA for numerical stability
        with torch.no_grad():
            self.ema_fprs = (1 - self.ema_alpha) * self.ema_fprs + self.ema_alpha * error_rates.detach()
        
        # Fairness penalty: variance of error rates across groups
        fairness_penalty = error_rates.var(dim=0)
        
        # Clamp to prevent numerical issues
        fairness_penalty = torch.clamp(fairness_penalty, min=0, max=1.0)
        
        # === STEP 3: Combine Losses ===
        
        total_loss = ce_loss + self.lambda_fairness * fairness_penalty
        
        # === STEP 4: Return with metrics for logging ===
        
        metrics = {
            'error_rates': error_rates.detach().cpu().numpy(),
            'error_rates_var': fairness_penalty.item(),
            'error_rates_std': error_rates.std().item(),
        }
        
        return total_loss, ce_loss, fairness_penalty, metrics
    
    def set_lambda(self, new_lambda):
        """Dynamically adjust fairness weight during training."""
        self.lambda_fairness = new_lambda


class EqualizedOddsWithTPRFNRLoss(nn.Module):
    """
    Advanced Equalized Odds: Minimize both TPR and FPR variance.
    
    This ensures:
    - Equal True Positive Rates (equal recall/sensitivity across groups)
    - Equal False Positive Rates (equal false alarm rates across groups)
    
    More comprehensive fairness constraint than FPR alone.
    """
    
    def __init__(self, num_races=5, lambda_fairness=0.1, lambda_tpr=0.5, 
                 lambda_fpr=0.5, device='cuda'):
        super().__init__()
        self.num_races = num_races
        self.lambda_fairness = lambda_fairness
        self.lambda_tpr = lambda_tpr  # Weight for TPR variance
        self.lambda_fpr = lambda_fpr  # Weight for FPR variance
        self.device = device
    
    def compute_fpr_for_race(self, preds, targets, race_id):
        """Compute FPR (false alarm rate)."""
        non_race_mask = (targets != race_id)
        if non_race_mask.sum() == 0:
            return torch.tensor(0.0, device=preds.device)
        
        non_race_preds = preds[non_race_mask]
        fp = (non_race_preds == race_id).float().sum()
        tn = (non_race_preds != race_id).float().sum()
        
        if fp + tn == 0:
            return torch.tensor(0.0, device=preds.device)
        
        return fp / (fp + tn)
    
    def compute_tpr_for_race(self, preds, targets, race_id):
        """Compute TPR (recall/sensitivity)."""
        race_mask = (targets == race_id)
        if race_mask.sum() == 0:
            return torch.tensor(0.0, device=preds.device)
        
        race_preds = preds[race_mask]
        tp = (race_preds == race_id).float().sum()
        fn = (race_preds != race_id).float().sum()
        
        if tp + fn == 0:
            return torch.tensor(0.0, device=preds.device)
        
        return tp / (tp + fn)
    
    def forward(self, logits, targets, race_labels, sample_weights=None):
        """
        Compute loss with both TPR and FPR fairness constraints.
        """
        
        # CE Loss
        logits = torch.clamp(logits, -100, 100)
        ce_loss_unreduced = F.cross_entropy(logits, targets, reduction='none')
        
        if sample_weights is not None:
            sample_weights = sample_weights / sample_weights.mean()
            ce_loss = (ce_loss_unreduced * sample_weights).mean()
        else:
            ce_loss = ce_loss_unreduced.mean()
        
        # Compute FPR and TPR for each race
        with torch.no_grad():
            preds = logits.argmax(dim=1)
        
        fprs = []
        tprs = []
        
        for race_id in range(self.num_races):
            fpr = self.compute_fpr_for_race(preds, targets, race_id)
            tpr = self.compute_tpr_for_race(preds, targets, race_id)
            
            fprs.append(fpr)
            tprs.append(tpr)
        
        fprs = torch.stack(fprs)
        tprs = torch.stack(tprs)
        
        # Fairness penalties
        fpr_penalty = fprs.var()
        tpr_penalty = tprs.var()
        
        fairness_penalty = (self.lambda_tpr * tpr_penalty + 
                           self.lambda_fpr * fpr_penalty)
        
        # Total loss
        total_loss = ce_loss + self.lambda_fairness * fairness_penalty
        
        metrics = {
            'fprs': fprs.detach().cpu().numpy(),
            'tprs': tprs.detach().cpu().numpy(),
            'fpr_var': fpr_penalty.item(),
            'tpr_var': tpr_penalty.item(),
        }
        
        return total_loss, ce_loss, fairness_penalty, metrics


if __name__ == '__main__':
    """Test the Equalized Odds Loss"""
    
    # Simulate batch
    batch_size = 32
    num_races = 5
    
    logits = torch.randn(batch_size, num_races)
    targets = torch.randint(0, num_races, (batch_size,))
    race_labels = targets.clone()
    sample_weights = torch.ones(batch_size) * 0.5
    
    # Test basic EqualizedOddsLoss
    criterion = EqualizedOddsLoss(num_races=num_races, lambda_fairness=0.1)
    total_loss, ce_loss, fairness_penalty, metrics = criterion(
        logits, targets, race_labels, sample_weights
    )
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"CE Loss: {ce_loss.item():.4f}")
    print(f"Fairness Penalty: {fairness_penalty.item():.4f}")
    print(f"FPR per race: {metrics['error_rates']}")
    print(f"FPR Variance: {metrics['error_rates_var']:.6f}")
    
    # Test advanced loss
    print("\n--- Advanced Loss (TPR + FPR) ---")
    criterion_advanced = EqualizedOddsWithTPRFNRLoss(
        num_races=num_races, 
        lambda_fairness=0.1,
        lambda_tpr=0.5,
        lambda_fpr=0.5
    )
    
    total_loss, ce_loss, fairness_penalty, metrics = criterion_advanced(
        logits, targets, race_labels, sample_weights
    )
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"CE Loss: {ce_loss.item():.4f}")
    print(f"Fairness Penalty: {fairness_penalty.item():.4f}")
    print(f"TPR per race: {metrics['tprs']}")
    print(f"FPR per race: {metrics['fprs']}")
    print(f"TPR Variance: {metrics['tpr_var']:.6f}")
    print(f"FPR Variance: {metrics['fpr_var']:.6f}")
