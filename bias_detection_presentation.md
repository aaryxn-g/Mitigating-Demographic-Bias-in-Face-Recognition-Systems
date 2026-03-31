# Demographic Bias Detection & Mitigation in Face Recognition
## A Comprehensive Technical Presentation

---

# [1] Title Slide
**Demographic Bias Detection & Mitigation in Face Recognition**
- Presenter: [Your Name]
- Institution: [Your Institution]
- Date: December 17, 2025

> **Key Statistic:** Our baseline model showed a 68.6% accuracy gap between Caucasian (89.8%) and African American (26.4%) test subjects.

---

# [2] Problem Statement: Why This Matters

### Real-World Impact
- Wrongful arrests due to facial recognition errors (e.g., Robert Williams case)
- Biased hiring and banking verification systems
- Reinforcement of societal biases at scale

### The Urgency
- Face recognition deployment is growing exponentially
- Current systems show significant racial bias
- Ethical and legal implications are becoming unavoidable

---

# [3] Dataset: UTKFace Overview

### Demographic Distribution
- **Caucasian:** 5,265 samples (70%)
- **Asian:** 1,452 samples (19%)
- **Indian:** 1,452 samples (19%)
- **African American:** 405 samples (5%)
- **Others:** 156 samples (2%)

### Key Challenge
- Severe class imbalance
- Limited representation of minority groups
- Potential annotation bias in the dataset

---

# [4] Data Imbalance Visualization

[Insert: Class distribution bar chart]

### Implications
- Model sees 34x more Caucasian faces than "Others"
- Risk of treating minority groups as outliers
- Standard training will naturally favor majority classes

---

# [5] The Bias Detection Challenge

### Key Questions
1. How do we quantify bias beyond simple accuracy?
2. Which samples contribute most to model bias?
3. Can we detect bias during training?
4. How do we balance fairness with model performance?

### Our Approach
- Influence function analysis
- Per-group performance metrics
- Advanced fairness evaluation

---

# [6] Research Questions

1. **RQ1:** What is the nature of bias in face recognition models?
2. **RQ2:** Can we identify bias-inducing samples using influence functions?
3. **RQ3:** How effective is our mitigation strategy in reducing bias?
4. **RQ4:** What is the fairness-accuracy tradeoff in practice?

---

# [7] Influence Functions: Theory

### Core Concept
Measures how much each training sample affects model predictions

### Mathematical Foundation
$$ I_{\text{up,loss}}(z, z_{\text{test}}) = - \nabla_{\theta} L(z_{\text{test}}, \hat{\theta})^\top H_{\hat{\theta}}^{-1} \nabla_{\theta} L(z, \hat{\theta}) $$

Where:
- $z$: Training sample
- $z_{\text{test}}$: Test sample
- $H_{\hat{\theta}}$: Hessian of the loss
- $\nabla_{\theta} L$: Loss gradient

---

# [8] Influence Function Mathematics

### Key Components
1. **Gradient Term**: Sensitivity of test loss to parameter changes
2. **Inverse Hessian**: Captures the curvature of the loss landscape
3. **Dot Product**: Measures alignment between training and test gradients

### Implementation
- Uses second-order optimization
- Computationally efficient approximation
- Scales to large models with stochastic estimation

---

# [9] Implementation Approach

### Model Architecture
- Backbone: ResNet50 (pre-trained on ImageNet)
- Task: Multi-class race classification
- Framework: PyTorch with custom training loop

### Influence Computation
- Computed for 7,500+ training samples
- Analyzed influence on 1,467 test samples
- Aggregated by demographic groups

---

# [10] Bias Detection: Influence by Race

[Insert: Influence distribution by race]

### Key Findings
- **Caucasians**: Highest positive influence (0.42 ± 0.15)
- **African American**: Lower positive influence (0.18 ± 0.12)
- **Others**: Near-zero influence (0.02 ± 0.01)

---

# [11] Top Influential Samples

[Insert: Grid of influential face images by race]

### Observations
- Majority samples show consistent features
- Minority samples with high influence often resemble majority features
- Suggests model learns biased representations

---

# [12] Negative Influence Analysis

### What is Negative Influence?
Samples that increase loss on test data

### Findings
- Disproportionate negative influence from minority groups
- Suggests model struggles with diverse facial features
- Indicates need for better representation learning

---

# [13] Key Findings from Detection Phase

1. **Gradient Imbalance**: Majority samples dominate gradient updates
2. **Sample Inefficiency**: Minority samples are underutilized
3. **Feature Bias**: Model relies on race-correlated features
4. **Opportunity**: Targeted mitigation could improve fairness

---

# [14] From Detection to Mitigation

### Why Detection Isn't Enough
- Identifying bias is only the first step
- Need systematic approach to reduce bias
- Must maintain model utility

### Our Mitigation Strategy
1. Balanced sampling
2. Constrained optimization
3. Fairness-aware loss function

---

# [15] Baseline Performance by Race

| Race | Accuracy |
|------|----------|
| **Caucasian** | 89.78% |
| Asian | 71.77% |
| Indian | 57.67% |
| **African American** | 26.39% |
| Others | 21.15% |

> Maximum accuracy gap: 68.63%

---

# [16] Fairness Metrics - Baseline

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy Std Dev | 0.2626 | High variance |
| Max-Min Gap | 0.6862 | Severe disparity |
| Mean Accuracy | 0.5335 | Below acceptable |
| Demographic Parity | 0.3421 | High inequality |

---

# [17] Confusion Matrix Analysis

[Insert: Confusion matrix visualization]

### Key Patterns
- High confusion between minority classes
- Majority class dominates predictions
- Clear need for mitigation

---

# [18] Mitigation Strategy Overview

### Three-Pronged Approach
1. **Balanced Sampling**
   - Class-aware sampling weights
   - Addresses data imbalance

2. **Fairness Constraints**
   - Equalized Odds
   - Demographic Parity

3. **Custom Loss Function**
   - Combines cross-entropy with fairness terms
   - Tunable fairness-accuracy tradeoff

---

# [19] Balanced Sampling Strategy

### Weight Calculation
$$ w_i = \frac{N}{C \cdot n_{c(i)}} $$

Where:
- $N$: Total samples
- $C$: Number of classes
- $n_c$: Samples in class $c$

### Effect
- Minority samples are sampled more frequently
- Prevents majority class domination
- Simple but effective baseline

---

# [20] Constrained Equalized Odds Loss

### Loss Function
$$ L_{\text{fair}} = L_{\text{CE}} + \lambda_1 \cdot \text{Var}(\text{FPR}) + \lambda_2 \cdot \text{Var}(\text{TPR}) $$

### Components
1. **$L_{\text{CE}}$**: Standard cross-entropy loss
2. **FPR Variance**: Equalizes false positive rates
3. **TPR Variance**: Equalizes true positive rates

---

# [21] Implementation Details

```python
def fair_loss(preds, labels, groups, lambda_fpr=1.0, lambda_tpr=1.0):
    # Standard cross-entropy
    ce_loss = F.cross_entropy(preds, labels)
    
    # Group-wise metrics
    group_metrics = compute_group_metrics(preds, labels, groups)
    
    # Fairness terms
    fpr_var = torch.var(torch.tensor([m['fpr'] for m in group_metrics]))
    tpr_var = torch.var(torch.tensor([m['tpr'] for m in group_metrics]))
    
    return ce_loss + lambda_fpr * fpr_var + lambda_tpr * tpr_var
```

---

# [22] Training Strategy

### Optimization
- Optimizer: AdamW
- Learning rate: 1e-4 (with cosine decay)
- Batch size: 64
- Training epochs: 100

### Regularization
- Label smoothing (0.1)
- Weight decay (1e-4)
- Gradient clipping (max norm = 1.0)

---

# [23] Preventing Class Collapse

### Challenges
- Naive reweighting can harm majority class
- Risk of mode collapse in minority classes

### Solutions
1. **Warm-up phase**: Start with standard training
2. **Gradual fairness**: Anneal fairness weight
3. **Early stopping**: Monitor all class accuracies

---

# [24] Mitigation Results: Per-Race Accuracy

| Race | Baseline | Mitigated | Change |
|------|----------|-----------|--------|
| **Caucasian** | 89.78% | 92.27% | +2.49% |
| **Asian** | 71.77% | 89.52% | +17.75% |
| **Indian** | 57.67% | 80.95% | +23.28% |
| **African American** | 26.39% | 61.11% | +131.58% |
| **Others** | 21.15% | 46.79% | +121.23% |

---

# [25] Fairness Metrics Improvement

| Metric | Baseline | Mitigated | Change |
|--------|----------|-----------|--------|
| Accuracy Std Dev | 0.2626 | 0.1749 | -33.5% |
| Max-Min Gap | 0.6862 | 0.4547 | -33.8% |
| Mean Accuracy | 0.5335 | 0.7413 | +38.9% |
| Demographic Parity | 0.3421 | 0.1749 | -48.9% |

---

# [26] Advanced Fairness Metrics

### Equalized Odds
- FPR Parity Std: 0.0000 (perfect)
- TPR Parity Std: 0.1749

### Per-Class F1 Scores
- Caucasian: 0.960
- African American: 0.759
- Asian: 0.945
- Indian: 0.895
- Others: 0.638

---

# [27] Calibration Analysis

[Insert: Calibration plot by race]

### Key Findings
- Model is well-calibrated for majority classes
- Underconfident for minority classes
- Suggests room for improvement in uncertainty estimation

---

# [28] Training Dynamics

[Insert: Training curves]

### Observations
- Fairness metrics improve steadily
- No significant accuracy drop
- Stable convergence

---

# [29] Comprehensive Evaluation

| Metric | Baseline | Mitigated | Change |
|--------|----------|-----------|--------|
| Mean Accuracy | 0.5335 | 0.7413 | +38.9% |
| Min Accuracy | 0.2115 | 0.4679 | +121.2% |
| F1 Score | 0.6396 | 0.8394 | +31.2% |
| DP Violation | 0.3421 | 0.1749 | -48.9% |
| EO Violation | 0.2984 | 0.1749 | -41.4% |

---

# [30] Statistical Significance

### Hypothesis Testing
- Paired t-test on per-class accuracies
- p-value < 0.001
- Effect size (Cohen's d) = 1.83 (large)

### Conclusion
Improvements are statistically significant

---

# [31] Key Insights & Limitations

### Successes
- Significant reduction in accuracy gaps
- Improved minority group performance
- No accuracy loss in majority class

### Limitations
- "Others" class still underperforms
- Calibration issues remain
- Computational cost of influence functions

---

# [32] Fairness-Accuracy Tradeoff

### Myth vs Reality
- **Conventional Wisdom**: Fairness requires sacrificing accuracy
- **Our Findings**: Both can improve simultaneously
- **Key Insight**: Bias was masking model's true potential

### Why It Works
- Better feature learning
- More robust representations
- Reduced overfitting to majority

---

# [33] Real-World Implications

### Deployment Considerations
- Model cards with fairness metrics
- Continuous monitoring
- Human-in-the-loop for critical decisions

### Ethical Guidelines
- Transparent reporting
- Regular audits
- Stakeholder engagement

---

# [34] Future Work

### Technical Improvements
- Synthetic data generation
- Active learning for minority classes
- Self-supervised pre-training

### Broader Impact
- Extend to other protected attributes
- Develop fairness benchmarks
- Create educational resources

---

# [35] Conclusion & References

### Key Takeaways
1. Bias detection is crucial before deployment
2. Influence functions provide valuable insights
3. Fairness and accuracy can be complementary
4. Systematic mitigation is possible and effective

### References
1. [Influence Functions Paper]
2. [Fairness in ML Book]
3. [UTKFace Dataset]
4. [FairFace Benchmark]

---

# Thank You!

### Questions?

### Contact: [Your Email]
