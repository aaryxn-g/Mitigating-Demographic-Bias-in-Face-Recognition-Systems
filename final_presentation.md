# Demographic Bias Detection and Mitigation in Face Recognition
## From Bias Quantification to Fair Model Development

**Presenter:** Aaryan
**Date:** December 17, 2025
**Institution:** [Institution Name]

> **Statistic:** Face recognition systems have been found to be up to **34% less accurate** for darker-skinned females compared to lighter-skinned males (Buolamwini & Gebru, 2018).

---

## Motivation - Why Face Recognition Bias Matters

### Real-World Deployment Failures
- **Wrongful Arrests:** Multiple documented cases where facial recognition mismatch led to wrongful detainment of African American individuals (e.g., Robert Williams case).
- **Service Denial:** Bias in identity verification services can deny access to banking, employment, or government services.

### The Accuracy Gap
- Typical Commercial Performance:
    - **Caucasian:** ~99% Accuracy
    - **African American:** ~65%-85% Accuracy (in severe cases)
- **Our Baseline:** Caucasian 89.78% vs. African American 26.39%

### High Stakes & Ethical Imperative
- **Domains:** Criminal Justice, Border Control, Mass Surveillance.
- **Imperative:** As AI systems scale, "fairness" is not optional—it is a requirement for safety and equity.

---

## UTKFace Dataset Composition

| Race | Count | Percentage |
| :--- | :--- | :--- |
| **Caucasian** | 5,265 | ~70% |
| **Asian** | 1,452 | ~19% |
| **Indian** | 1,452 | ~19% |
| **African American** | 405 | ~5% |
| **Others** | 156 | ~2% |

### The Problem: Severe Underrepresentation
- **Visualized:** The dataset is heavily skewed towards Caucasians.
- **Impact:** The model views minority faces as "outliers" rather than core data distributions.

---

## The Class Imbalance Problem

### Training Dynamics
- **74.6%** of training samples are Caucasian.
- Optimization algorithms (SGD/Adam) prioritize minimizing average error.
- Minimizing error on the majority class (Caucasian) yields the best global loss, ignoring minorities.

### Inference Consequence
- The model learns "Caucasian" features as "Universal" features.
- Resulting Baseline Accuracy:
    - **Caucasian:** 89.78%
    - **African American:** 26.39%

---

## Bias Detection Challenge

**Key Questions:**
1.  How do we *know* which samples are driving the bias? Is it just count, or quality?
2.  How do we measure fairness beyond simple accuracy?
3.  Can we improve minority performance without destroying majority performance ("The Pareto Frontier")?
4.  Standard metrics (Global Accuracy) hide these disparities—we need granular forensics.

---

## Research Questions

- **RQ1:** Which specific training samples cause bias amplification during the learning process?
- **RQ2:** Can constrained optimization techniques enforce fairness while preserving model utility?
- **RQ3:** What fairness definitions (Equalized Odds vs. Demographic Parity) are appropriate for deployment?
- **RQ4:** Can we achieve a "win-win" scenario where both fairness and accuracy improve?

---

## Influence Functions: Introduction

### Concept
- **Goal:** Measure exactly how much training sample $x$ contributed to the prediction for test sample $x_{test}$.
- **Mechanism:** "If we removed $x$ from the training set and retrained, how would the loss on $x_{test}$ change?"

### Formula
$$ I_{\text{up,loss}}(z, z_{\text{test}}) = - \nabla_{\theta} L(z_{\text{test}}, \hat{\theta})^\top H_{\hat{\theta}}^{-1} \nabla_{\theta} L(z, \hat{\theta}) $$
Where:
- $z$: Training sample
- $z_{\text{test}}$: Test sample
- $H_{\hat{\theta}}$: Hessian of the loss function (curvature)
- $\nabla_{\theta} L$: Gradient of the loss

---

## Influence Function Mathematics

### Derivation
- Based on a second-order Taylor expansion of the loss around the optimal parameters $\hat{\theta}$.
- Approximates the effect of upweighting a training point by $\epsilon$.

### Computational Efficiency
- Calculating the inverse Hessian $H^{-1}$ is expensive ($O(p^3)$).
- **Approximation:** Stochastic Estimation or leave-one-out approximations.
- **Implementation:** We used `torch.autograd` for efficient gradient computation, targeting the ResNet50 backbone layers.

---

## Implementation Details

### Architecture
- **Model:** ResNet50 (pre-trained on ImageNet, fine-tuned on UTKFace).
- **Task:** Multi-class classification (Age/Gender/Race prediction focus on Race).

### Influence Computation
- **Scope:** Computed influence of **7,500+** training samples on **1,467** test samples.
- **Aggregation:** We aggregated influence scores by demographic group to see which race "helps" or "hurts" predicting which other race.
- **Outputs:** `influence_by_race.csv`, `top_influential_by_race.csv`.

---

## Bias Detection Results - Influence Distribution

### Key Findings
- **Caucasians:** Highest average positive influence. The model "trusts" these samples most to build its feature extractors.
- **African American:** Significantly lower positive influence. The model struggles to leverage these samples effectively.
- **Others:** Near-zero influence—essentially "invisible" to the gradient updates.

*(Insert Bar Chart: Mean Influence Score per Race)*

---

## Top Influential Samples per Race

### Analysis
- We extracted the top-10 most "helpful" training images for each test group.
- **Observation:** High-influence samples often had standard lighting and frontal poses.
- **Diversity:** For minorities, the "influential" samples were often those that looked most similar to the majority (colorism bias).
- **Insight:** The model relies on specific, often non-representative features for minority recognition.

---

## Negative Influence Analysis

### What is Negative Influence?
- Samples that *increase* the loss on the test set (i.e., they confuse the model).
- **Findings:** A disproportionate number of "confusing" samples were found in minority groups, likely due to the model's inability to reconcile their features with the majority-dominated feature space.
- **Action:** These samples suggest the need for better data cleaning or specific re-weighting (mitigation).

---

## Distribution of Influence Across Races

### Variance Analysis
- **Caucasian Influence:** Tight distribution, consistent helper samples.
- **Minority Influence:** High variance. Some samples help a lot, many do nothing.
- **Statistical Test:** ANOVA confirmed significant differences in influence distributions between races ($p < 0.05$).
- **Implication:** The "value" of a training data point depends heavily on its demographic tag in a biased model.

---

## Conclusions from Bias Detection

1.  **Mechanism Identified:** Majority samples dominate the gradient direction during training.
2.  **Sample Inefficiency:** Minority samples are not just few; they are computationally "ignored" (low influence).
3.  **Path to Mitigation:** We must artificially boost the "signal" of these minority samples to counteract their low native influence.
4.  **Transition:** Detection is complete. We now know *why* the bias exists.

---

## Per-Race Accuracy - Baseline Model

| Race | Accuracy |
| :--- | :--- |
| **Caucasian** | **89.78%** |
| **Asian** | 71.77% |
| **Indian** | 57.67% |
| **African American** | **26.39%** |
| **Others** | 21.15% |

> **Gap:** A staggering **68.63%** difference between maximum and minimum group accuracy.

---

## Fairness Metrics - Baseline

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy Std Dev** | 0.2626 | High variance = High Unfairness |
| **Max-Min Gap** | 0.6862 | The "Worst-Case" disparity |
| **Mean Accuracy** | 0.5335 | Poor overall system performance |
| **Demographic Parity**| 0.3421 | High inequality in prediction rates |

**Conclusion:** The baseline model is deployable ONLY for Caucasians. It fails basic fairness checks.

---

## Baseline Confusion Matrix & Analysis

- **Confusion:** The model frequently misclassifies African Americans and Indians as "Caucasian" or "Asian" (the dominant classes).
- **Recall:**
    - Caucasian Recall: ~0.90
    - African American Recall: ~0.26
- **Precision:** High for Caucasians, low for minorities.
- **Insight:** The model is "confident but wrong" on minorities, defaulting to the majority class priors.

---

## Mitigation Strategy Overview

### Three-Pronged Approach

1.  **Balanced Sampling:**
    - Counteract data quantity imbalance.
    - *Method:* Inverse frequency weighting + WeightedRandomSampler.

2.  **Constrained Equalized Odds Loss:**
    - Counteract data quality/difficulty imbalance.
    - *Method:* Penalize differences in performance (FPR/TPR) across groups.

3.  **Utility Constraints:**
    - Prevent "crashing" the model to achieve fairness (e.g., getting 0% on everyone is "fair" but useless).
    - *Method:* Minimum accuracy threshold $\tau$.

---

## Balanced Sampling Strategy

### Weights Computation
$$ w_i = \frac{\max(\text{class\_counts})}{\text{class\_count}_i} $$

- **Effect:**
    - Caucasian Weight: 1.0
    - African American Weight: ~13.0
- **Implementation:** During training, an African American face is sampled ~13x more often (statistically) than a Caucasian face.
- **Result:** The model "sees" a balanced effective dataset.

---

## Constrained Equalized Odds Loss

### The Formula

$$ L_{\text{total}} = \alpha \underbrace{L_{\text{ce}}(y, \hat{y})}_{\text{Accuracy}} + \beta \underbrace{\sum_{r} (FPR_r - \bar{FPR})^2 + (TPR_r - \bar{TPR})^2}_{\text{Fairness (Equalized Odds)}} + \gamma \underbrace{\sum_{r} \max(0, \tau - \text{acc}_r)}_{\text{Utility Constraint}} $$

### Components
1.  **Cross Entropy:** Standard classification loss.
2.  **Equalized Odds Penalty:** Forces False Positive and True Positive rates to be similar across races.
3.  **Hinge Loss Constraint:** Penalizes the model instantly if any group's accuracy drops below $\tau$.

---

## Loss Function Explanation

```python
# Pseudo-code for Fairness Loss
def fairness_loss(preds, labels, race_indices):
    ce_loss = cross_entropy(preds, labels)
    
    fprs, tprs, accs = [], [], []
    for r in races:
        # Calculate per-race metrics
        stats = calculate_metrics(preds[race_indices[r]], labels[race_indices[r]])
        fprs.append(stats.fpr); tprs.append(stats.tpr); accs.append(stats.acc)
        
    # Variance of rates
    fairness_penalty = var(fprs) + var(tprs)
    
    # Accuracy constraint
    constraint_penalty = sum([max(0, threshold - acc) for acc in accs])
    
    return alpha * ce_loss + beta * fairness_penalty + gamma * constraint_penalty
```

---

## Hyperparameter Selection

- **$\alpha$ (Accuracy):** 1.0
- **$\beta$ (Fairness):** 0.5 (Found via grid search to balance trade-off)
- **$\gamma$ (Constraint):** 1.0 (Strict enforcement)
- **$\tau$ (Min Accuracy):** 0.55 (Ensures minorities reach at least better-than-random performance)

**Why these values?**
Higher $\beta$ caused unstable training; higher $\gamma$ caused the model to oscillate. This set provided smooth convergence.

---

## Training Dynamics & Convergence

- **Epochs:** 30
- **Curve Analysis:**
    - *Loss:* Rapid drop in first 10 epochs.
    - *Fairness Gap:* Steadily decreased from 0.68 to 0.45.
    - *Majority Accuracy:* Remained stable (~92%).
- **Stability:** No "class collapse" (degenerate solution where model predicts only one class) was observed, validating the constraint term.

---

## Per-Race Accuracy - After Mitigation

| Race | Baseline | **Mitigated** | Improvement |
| :--- | :--- | :--- | :--- |
| **Caucasian** | 89.78% | **92.27%** | +2.49% |
| **Asian** | 71.77% | **89.52%** | +24.72% |
| **Indian** | 57.67% | **80.95%** | +40.37% |
| **African American** | 26.39% | **61.11%** | **+131.58%** |
| **Others** | 21.15% | **46.79%** | +121.21% |

**Result:** Massive gains for underrepresented groups without hurting the majority!

---

## Fairness Metrics After Mitigation

| Metric | Baseline | **Mitigated** | Change |
| :--- | :--- | :--- | :--- |
| **Accuracy Std Dev** | 0.2626 | **0.1749** | -33.5% (Better) |
| **Max-Min Gap** | 0.6862 | **0.4547** | -33.8% (Better) |
| **Mean Accuracy** | 0.5335 | **0.7413** | +38.9% (Better) |

**Verdict:** The system is objectively fairer and more accurate overall.

---

## Advanced Fairness Evaluation Results

### Equalized Odds
- **FPR Parity Std Dev:** **0.0000** (Perfect Alignment)
- **TPR Parity Std Dev:** 0.1749
    - *Interpretation:* The model makes False Positive errors at exactly the same rate for everyone. True Positives still vary slightly but are much closer.

### Statistical Significance
- **Chi-Square Test:** $p = 0.0000$
- The improvement in fairness is statistically significant and not due to random training noise.

---

## Calibration Analysis (Model Confidence)

### Expected Calibration Error (ECE)
- **Caucasian:** 0.2683
- **African American:** 0.3294 (Overconfident)
- **Asian:** 0.2399 (Best calibrated)
- **Indian:** 0.3172

**Implication:** Even though accuracy improved, the model is still "overconfident" when it makes mistakes on African American faces. This is a key area for future work (e.g., Temperature Scaling).

---

## Confusion Matrix Post-Mitigation

- **Precision:** 1.0000 across ALL groups.
    - The model effectively eliminated False Positives.
- **Recall:**
    - Caucasian: ~92%
    - African American: ~61%
- **F1 Score:** Harmonic mean of precision and recall shows consistent improvement across the board. The "diagonal" of the confusion matrix is much stronger.

---

## Baseline vs. Mitigated - Comparison Table

| Metric | Baseline | Mitigated | Change |
| :--- | :--- | :--- | :--- |
| **Caucasian** | 89.78% | 92.27% | +2.49% |
| **African American** | 26.39% | 61.11% | **+131.58%** |
| **Asian** | 71.77% | 89.52% | +24.72% |
| **Gap (Max-Min)** | 0.6862 | 0.4547 | -33.8% |
| **Mean Acc** | 0.5335 | 0.7413 | +38.9% |
| **FPR Parity** | High | 0.0000 | **PERFECT** |

---

## Statistical Significance & Hypothesis Testing

- **Null Hypothesis ($H_0$):** Mitigation strategy has no effect on demographic fairness.
- **Test:** Chi-square test on prediction distributions across races.
- **Result:** $\chi^2 = 236.23, p = 0.0000$.
- **Conclusion:** We reject $H_0$. The mitigation strategy fundamentally altered the model's behavior towards a fairer distribution.

---

## Key Insights & Remaining Limitations

### Achievements
- **Win-Win:** Fairness improved WITHOUT sacrificing majority accuracy (which actually rose by 2.5%).
- **Stability:** Constraint formulation successfully prevented model degradation.

### Limitations
- **Data constraint:** African American recall capped at 61%. With only 405 samples, there is an information theoretic limit.
- **Calibration:** Model confidence needs tuning.
- **Others class:** Still below 50% accuracy due to extreme scarcity (156 samples).

---

## The Fairness-Utility Tradeoff Myth

- **Myth:** "To make a model fair, you must make it worse for the majority."
- **Reality:** In our project, fairness **improved** utility.
- **Reasoning:** Removing reliance on spurious, majority-only features forces the model to learn robust, generalized facial features that work better for *everyone*.
- **Takeaway:** Fairness is a quality control metric, not a tax.

---

## Real-World Deployment Implications

- **Current Status:**
    - **Ready:** For Caucasian/Asian/Indian deployment.
    - **Conditional:** For African American (61% is better, but risky for high-stakes).
- **Recommendation:**
    - Use "Human-in-the-loop" for low-confidence predictions (especially for minorities checking Calibration).
    - Deploy as a screening tool, not a final arbiter.
- **Monitoring:** Continuous bias auditing required as population demographics shift.

---

## Future Work

### Short-Term (1-2 Weeks)
- **Temperature Scaling:** To fix the ECE/Calibration issues.
- **Uncertainty Thresholds:** Reject predictions where confidence < 0.7.

### Medium-Term
- **Synthetic Data (GANs):** Generate 1,000 synthetic African American faces to balance the dataset.
- **Active Learning:** Specifically target data collection for high-uncertainty minority samples.

### Long-Term
- **Federated Learning:** Train on private minority datasets without exposing data.
- **Pareto-Frontier Optimization:** Automating the search for $\alpha, \beta, \gamma$.

---

## Conclusion

1.  **Bias Detected:** We quantified severe negligence of minority samples using Influence Functions.
2.  **Bias Mitigated:** We implemented a custom **Constrained Equalized Odds Loss**.
3.  **Result:** **33.8% fairer** and **38.9% more accurate** system.
4.  **Message:** Algorithmic fairness is not just an ethical ideal—it is an engineering discipline that produces superior, more robust AI systems.

**Thank You.**
