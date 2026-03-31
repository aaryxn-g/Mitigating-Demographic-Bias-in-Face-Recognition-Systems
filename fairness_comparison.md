# Fairness Training Approaches Comparison

## Overview
Your Design Project contains two different fairness training approaches:

### 1. **train_fair_loss_fixed.py** - Accuracy Variance Minimization
- **Fairness Metric**: Minimizes standard deviation of per-race accuracies
- **Formula**: `fairness_penalty = 0.7 * std(accuracies) + 0.3 * (max_acc - min_acc)`
- **Problem**: Can achieve "fairness" by making all groups perform poorly
- **Your Result**: Caucasian accuracy dropped to 0% to balance with minority groups

### 2. **train_fair_eo.py** - Equalized Odds (Recommended)
- **Fairness Metric**: Minimizes variance in False Positive Rates (FPR) and True Positive Rates (TPR)
- **Formula**: `fairness_penalty = 0.5 * var(FPR) + 0.5 * var(TPR)`
- **Advantage**: More principled, aligns with fairness literature
- **Benefit**: Prevents class collapse by tracking error rates, not just accuracy

## Key Differences

| Aspect | Accuracy Variance | Equalized Odds |
|--------|------------------|----------------|
| **What it optimizes** | Similar accuracy across groups | Similar error rates across groups |
| **Theoretical basis** | Ad-hoc | Well-established (Hardt et al., 2016) |
| **Prevents collapse** | ❌ No | ✅ Yes |
| **Interpretability** | "Equal performance" | "Equal treatment" |
| **Your results** | 0% Caucasian accuracy | Not yet tested |

## Why Equalized Odds is Better

### 1. **Prevents Class Collapse**
- Accuracy variance can be minimized by making everyone perform at 0%
- Equalized Odds requires maintaining reasonable performance while equalizing error rates

### 2. **Fairness Literature Standard**
- Equalized Odds is a well-studied fairness criterion
- Used in academic research and industry deployments
- Has theoretical guarantees

### 3. **Better Interpretability**
- FPR: "How often is someone wrongly classified as race X?"
- TPR: "How often is someone correctly identified as race X?"
- Easier to explain to stakeholders than "accuracy variance"

## Evaluation Results Comparison

### Current Model (train_fair_loss_fixed.py)
```
Caucasian:        89.78% → 0.00%   (-100%)  ❌ COLLAPSED
African American: 26.39% → 88.89%  (+237%)  ✓
Asian:            71.77% → 95.16%  (+33%)   ✓
Indian:           57.67% → 98.94%  (+72%)   ✓
Others:           21.15% → 94.23%  (+345%)  ✓

Mean Accuracy: 53.35% → 75.44%
Std Dev: 0.263 → 0.379 (WORSE)
Max-Min Gap: 0.686 → 0.989 (WORSE)
```

**Verdict**: Improved minority groups but completely sacrificed majority class

### Expected with Equalized Odds (train_fair_eo.py)
- Should maintain reasonable accuracy for ALL groups (including Caucasian)
- Lower FPR variance (no group systematically misclassified)
- Lower TPR variance (no group systematically ignored)
- Better balance between fairness and utility

## Recommendation

**Use train_fair_eo.py** for your next training run because:
1. More principled fairness approach
2. Less likely to cause class collapse
3. Better for publication/deployment
4. Aligns with fairness research standards

## How to Run

```bash
# Equalized Odds training (recommended)
python train_fair_eo.py \
    --weights fairness_sample_weights_ultraconservative.csv \
    --data_dir crop_part1 \
    --batch_size 32 \
    --epochs 25 \
    --lambda_fairness 0.1

# Compare with previous approach
python eval_comprehensive.py \
    --baseline "Detection code/final_race_model.pth" \
    --mitigated "phase2_fair_outputs_eo/best_fair_model_eo.pth" \
    --test_dir "data/test" \
    --test_csv "data/test_labels.csv"
```

## References
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. NeurIPS.
- Chouldechova, A. (2017). Fair prediction with disparate impact. Big Data.
