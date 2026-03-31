# Fairness Training Analysis: Why Class Collapse Occurred

## Problem Summary

Both fairness training approaches resulted in **class collapse** where Caucasian accuracy dropped to near 0%:

| Approach | Caucasian Acc | Other Races | Status |
|----------|---------------|-------------|--------|
| **Baseline** | 89.78% | 21-72% | Biased but functional |
| **Accuracy Variance** | 0.00% | 88-98% | ❌ Collapsed |
| **Equalized Odds** | 0.62% | 15-78% | ❌ Collapsed |

## Root Cause Analysis

### 1. **Sample Weight Imbalance** (Primary Cause)

Even with "conservative" 3:1 ratio, the weights still heavily favor minorities:

```
Caucasian:        0.632× (5,265 samples) → Effective: 3,327 samples
African American: 1.897× (405 samples)   → Effective: 768 samples  
Indian:           1.598× (1,452 samples) → Effective: 2,320 samples
```

**Problem**: The model sees Caucasian samples with 63% weight, meaning it's penalized less for getting them wrong. Over 25 epochs, the model learns to ignore Caucasians entirely.

### 2. **Fairness Loss Encourages Collapse**

**Training History Shows**:
- **Epoch 1**: Caucasian 11.1%, African American 15.6%, Asian 54.2%
- **Epoch 25**: Caucasian 0.36%, African American 99.7%, Asian 99.9%

**Why this happened**:
1. Fairness loss minimizes variance in per-race accuracies
2. Easiest way to reduce variance: make all accuracies equal
3. With downweighted Caucasians, model can achieve low loss by:
   - Getting minorities 100% correct (high weight)
   - Getting Caucasians 0% correct (low weight)
   - Result: Low CE loss (weighted) + Low fairness penalty (similar accuracies)

### 3. **Lambda Scheduling Made It Worse**

Lambda increased from 0.075 → 0.224 over training, meaning fairness penalty became MORE important than accuracy. This accelerated the collapse.

## Why Equalized Odds Didn't Help

Equalized Odds (FPR/TPR variance) should theoretically prevent this, but:

1. **Still uses sample weights** - Caucasians downweighted in CE loss
2. **FPR/TPR variance can be minimized** by making all races perform equally (even if that means 0% for one group)
3. **No explicit constraint** preventing any single group from collapsing

## Solutions

### ✅ **Solution 1: Remove Sample Weights** (Recommended)

**Approach**: Train with fairness loss but WITHOUT sample weights

```python
# In train_fair_eo.py, modify forward pass:
ce_loss = F.cross_entropy(logits, targets)  # No weights!
total_loss = ce_loss + lambda_fairness * fairness_penalty
```

**Why this works**:
- CE loss treats all samples equally
- Fairness loss still encourages equal performance
- No incentive to sacrifice majority class

### ✅ **Solution 2: Add Minimum Accuracy Constraint**

**Approach**: Add penalty if any race drops below threshold

```python
class ConstrainedEqualizedOddsLoss(nn.Module):
    def forward(self, logits, targets, race_labels, sample_weights=None):
        # ... existing code ...
        
        # Compute per-race accuracies
        race_accs = []
        for race_id in range(self.num_races):
            mask = (race_labels == race_id)
            if mask.sum() > 0:
                acc = (preds[mask] == targets[mask]).float().mean()
                race_accs.append(acc)
        
        # Add penalty for any race below 50%
        min_acc_penalty = torch.relu(0.5 - torch.stack(race_accs).min())
        
        total_loss = ce_loss + lambda_fairness * fairness_penalty + 10.0 * min_acc_penalty
        return total_loss, ce_loss, fairness_penalty, metrics
```

### ✅ **Solution 3: Balanced Sampling Instead of Weighting**

**Approach**: Oversample minorities, undersample majority

```python
from torch.utils.data import WeightedRandomSampler

# Compute sampling weights (inverse frequency)
race_counts = {...}  # Count per race
sample_weights = [1.0 / race_counts[race] for race in races]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True
)

train_loader = DataLoader(dataset, sampler=sampler, ...)
```

**Why this works**:
- Model sees balanced data distribution
- No need to downweight any samples in loss
- All races treated equally during training

### ✅ **Solution 4: Group DRO (Distributionally Robust Optimization)**

**Approach**: Minimize worst-case group loss

```python
class GroupDROLoss(nn.Module):
    def forward(self, logits, targets, race_labels):
        # Compute loss per race
        race_losses = []
        for race_id in range(self.num_races):
            mask = (race_labels == race_id)
            if mask.sum() > 0:
                race_loss = F.cross_entropy(logits[mask], targets[mask])
                race_losses.append(race_loss)
        
        # Minimize maximum loss (worst-performing group)
        total_loss = torch.stack(race_losses).max()
        return total_loss
```

**Why this works**:
- Explicitly optimizes for worst-performing group
- Prevents any group from being sacrificed
- Well-studied in fairness literature

## Recommended Next Steps

### **Option A: Quick Fix** (1-2 hours)
1. Remove sample weights from `train_fair_eo.py`
2. Retrain with same Equalized Odds loss
3. Expected result: All races 70-85% accuracy, much more balanced

### **Option B: Proper Solution** (3-4 hours)
1. Implement balanced sampling with `WeightedRandomSampler`
2. Use Equalized Odds loss WITHOUT sample weights
3. Add minimum accuracy constraint (50% threshold)
4. Expected result: All races 75-90% accuracy, excellent fairness

### **Option C: Research-Grade** (1 day)
1. Implement Group DRO
2. Compare with Equalized Odds
3. Run ablation studies
4. Document findings for publication

## Key Takeaways

1. **Sample weighting is dangerous** - Can easily cause class collapse
2. **Fairness losses alone aren't enough** - Need constraints to prevent collapse
3. **Equalized Odds is still better** - More principled than accuracy variance
4. **Balanced sampling > Sample weighting** - Safer and more effective

## References

- Sagawa et al. (2020): "Distributionally Robust Neural Networks" (Group DRO)
- Hardt et al. (2016): "Equality of Opportunity in Supervised Learning" (Equalized Odds)
- Kearns et al. (2018): "Preventing Fairness Gerrymandering" (Constraints)

---

**Bottom Line**: The conservative weights (3:1 ratio) were still too aggressive. The solution is to either:
1. Remove weights entirely, OR
2. Use balanced sampling instead of weighted loss
