# Project Workflow and Execution Pipeline

This document outlines the complete execution order for the UTKFace Bias Detection and Mitigation project. It details the logical flow from raw data to final publication-quality evaluation, including all analysis and mitigation steps.

## 0. Prerequisites

**Environment**: `python 3.8+` with `torch`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`.
**Root Directory**: Ensure you are in the root directory where `crop_part1` behaves as the source data folder.

---

## 1. Data Visualization (Exploratory Analysis)

**Goal**: Understand the raw dataset distribution and identify initial imbalances (e.g. lack of minority samples).

*   **Script**: `Detection code/visualize_dataset.py`
*   **Input**: `crop_part1` directory.
*   **Command**:
    ```bash
    python "Detection code/visualize_dataset.py"
    ```
*   **Outputs** (`analysis_results/`):
    *   `race_distribution.png`: Bar chart of sample counts.
    *   `race_distribution_donut.png`: Percentage breakdowns.
    *   `samples_*.png`: Example images for each race.

---

## 2. Data Preparation

**Goal**: Standardize the dataset into Train/Validation/Test splits.

*   **Script**: `prepare_dataset.py`
*   **Input**: `crop_part1` directory.
*   **Command**:
    ```bash
    python prepare_dataset.py
    ```
*   **Outputs** (`data/`):
    *   `train/`, `val/`, `test/` directories with images.
    *   `train_labels.csv`, `val_labels.csv`, `test_labels.csv`.

---

## 3. Baseline Model Training

**Goal**: Train a standard ResNet18 classifier to establish a performance baseline and reveal bias.

*   **Script**: `Detection code/train_baseline_resnet.py`
*   **Input**: `crop_part1` (Note: Script does internal stratified split).
*   **Command**:
    ```bash
    python "Detection code/train_baseline_resnet.py"
    ```
*   **Outputs**:
    *   `final_race_model.pth`: Saved model weights (Critical for next steps).
    *   `logs/`: Training metrics.

---

## 4. Bias Detection (Influence Calculation)

**Goal**: Calculate *Influence Functions* to identify which specific training samples drive the model's predictions (and bias).

*   **Script**: `Detection code/calc_influence.py`
*   **Input**:
    *   `final_race_model.pth` (Must be in root dir).
    *   `crop_part1` (Data).
*   **Command**:
    ```bash
    python "Detection code/calc_influence.py"
    ```
*   **Outputs** (`Detection code/all_influence/`):
    *   `full_dataset_influence_scores.csv`: Core file mapping every image to its influence score.
    *   `ihvp.pt`: Intermediate Hessian-Vector Products.
    *   `bias_analysis/`: Plots showing influence per race.

---

## 5. Mitigation Preparation (Sample Weighting)

**Goal**: Analyze influence scores to compute ideal sample weights. Even if using "Fixed" training (Step 6), this script provides critical analysis of how much up-weighting minority samples *should* theoretically receive.

*   **Script**: `compute_sample_weights_balanced.py`
*   **Input**: `Detection code/all_influence/full_dataset_influence_scores.csv` (From Step 4).
*   **Command**:
    ```bash
    python compute_sample_weights_balanced.py \
      --influence_csv "Detection code/all_influence/full_dataset_influence_scores.csv"
    ```
*   **Outputs**:
    *   `fairness_sample_weights_balanced.csv`: Recommended weights for every sample.
    *   `weight_distribution_balanced.png`: Visualization of weight distribution.

---

## 6. Mitigation Training (Fair Fixed Method)

**Goal**: Train the final Fair Model using the "Fixed" strategy (Balanced Sampling + Min Accuracy Constraint + Equalized Odds).

*   **Script**: `train_fair_fixed.py`
*   **Input**: `data/train_labels.csv` (From Step 2).
*   **Command**:
    ```bash
    python train_fair_fixed.py --epochs 25 --output_dir phase2_fixed_outputs
    ```
*   **Why this method?**: Instead of fragile sample weights, this method uses `WeightedRandomSampler` (robust balancing) and strict loss constraints to prevent class collapse.
*   **Outputs** (`phase2_fixed_outputs/`):
    *   `best_fair_model_fixed.pth`: The optimal fair model.
    *   `training_history_fixed.json`: Training logs.

---

## 7. Comparative Evaluation

**Goal**: Compare Baseline vs. Fair Model on standard metrics to prove improvement.

*   **Script**: `eval_comprehensive.py`
*   **Command**:
    ```bash
    python eval_comprehensive.py \
      --baseline "final_race_model.pth" \
      --mitigated "phase2_fixed_outputs/best_fair_model_fixed.pth" \
      --test_dir "data/test" \
      --test_csv "data/test_labels.csv" \
      --output_dir "evaluation_results"
    ```
*   **Outputs** (`evaluation_results/`):
    *   `comprehensive_comparison.png`: Side-by-side performance charts.
    *   `evaluation_report.json`: Quantitative metrics comparison.

---

## 8. Advanced Fairness Evaluation

**Goal**: Generate publication-ready fairness statistics (ROC-AUC, Calibration, Equalized Odds).

*   **Script**: `run_advanced_eval.py`
*   **Input**: `phase2_fixed_outputs/best_fair_model_fixed.pth` (Automatically targeted).
*   **Command**:
    ```bash
    python run_advanced_eval.py
    ```
*   **Outputs** (`advanced_eval/`):
    *   `fairness_metrics.png`: Demographic Parity & Equalized Odds plots.
    *   `calibration.png`: Model reliability diagrams.
    *   `fairness_report.json`: Deep statistical analysis.

---

## Full Pipeline Summary

1.  **Visualize**: `python "Detection code/visualize_dataset.py"`
2.  **Prep**: `python prepare_dataset.py`
3.  **Baseline**: `python "Detection code/train_baseline_resnet.py"`
4.  **Influence**: `python "Detection code/calc_influence.py"`
5.  **Weight Analysis**: `python compute_sample_weights_balanced.py`
6.  **Fair Training**: `python train_fair_fixed.py`
7.  **Comparison**: `python eval_comprehensive.py ...` (See args above)
8.  **Deep Dive**: `python run_advanced_eval.py`
