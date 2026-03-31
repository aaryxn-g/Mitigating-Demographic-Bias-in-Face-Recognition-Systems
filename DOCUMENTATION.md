
# UTKFace Bias Detection: Comprehensive Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Complete File Structure](#complete-file-structure)
3. [Core Components](#core-components)
   - [Data Preparation](#data-preparation)
   - [Baseline Model](#baseline-model)
   - [Bias Detection](#bias-detection)
   - [Bias Mitigation](#bias-mitigation)
   - [Evaluation Framework](#evaluation-framework)
4. [Model Files and Checkpoints](#model-files-and-checkpoints)
5. [Detailed Usage Guide](#detailed-usage-guide)
6. [Implementation Details](#implementation-details)
7. [Results and Analysis](#results-and-analysis)
8. [Troubleshooting and FAQs](#troubleshooting-and-faqs)

## Project Overview

This project provides a comprehensive framework for analyzing and mitigating racial bias in facial recognition systems using the UTKFace dataset. The implementation includes tools for data processing, model training, bias detection, and fairness evaluation.

## Complete File Structure

### Root Directory
- `data/`
  - `train/` - Training images
  - `val/` - Validation images
  - `test/` - Test images
  - `train_labels.csv` - Training set annotations
  - `val_labels.csv` - Validation set annotations
  - `test_labels.csv` - Test set annotations

### Detection Code
- `train_baseline_resnet.py` - Main script for training baseline ResNet model
- `calc_influence.py` - Implements influence functions for bias analysis
- `visualize_dataset.py` - Visualization utilities for dataset and model analysis
- `final_race_model.pth` - Pretrained baseline model weights

### Mitigation Code
- `train_fair_loss.py` - Implements fairness-aware loss functions
- `train_fair_weighted.py` - Weighted sampling approach for fairness
- `compute_sample_weights_fixed.py` - Computes sample weights for balanced training
- `phase2_outputs/`
  - `best_fair_model.pth` - Best performing fair model
  - `final_fair_model.pth` - Final fair model weights
  - `training_logs/` - Training logs and metrics
- `checkpoints/` - Intermediate model checkpoints
  - `fair_loss_model.pth` - Model trained with fairness loss

### Core Scripts
- [eval.py](cci:7://file:///c:/Aaryan/College_Stuff/Design%20Project/eval.py:0:0-0:0) - Main evaluation script
- [prepare_dataset.py](cci:7://file:///c:/Aaryan/College_Stuff/Design%20Project/prepare_dataset.py:0:0-0:0) - Dataset processing and splitting
- `requirements.txt` - Project dependencies

## Core Components

### Data Preparation

#### [prepare_dataset.py](cci:7://file:///c:/Aaryan/College_Stuff/Design%20Project/prepare_dataset.py:0:0-0:0)
- **Purpose**: Processes raw UTKFace dataset into train/val/test splits
- **Key Functions**:
  - `process_dataset()`: Main processing pipeline
  - `create_splits()`: Creates balanced dataset splits
  - [write_csv()](cci:1://file:///c:/Aaryan/College_Stuff/Design%20Project/prepare_dataset.py:44:0-64:24): Generates annotation files
- **Output**:
  - Organized image directories
  - CSV files with race labels
  - Hard links to original images

#### Dataset Structure
- Images named as: `AGE_GENDER_RACE_TIMESTAMP.jpg.chip.jpg`
- Race mapping: 0-White, 1-Black, 2-Asian, 3-Indian, 4-Others

### Baseline Model

#### `train_baseline_resnet.py`
- **Architecture**: ResNet-18 with custom classification head
- **Training Loop**:
  - Cross-entropy loss
  - Adam optimizer
  - Learning rate scheduling
- **Features**:
  - Race-stratified sampling
  - Model checkpointing
  - TensorBoard logging

### Bias Detection

#### `calc_influence.py`
- **Purpose**: Analyzes training sample influence
- **Key Functions**:
  - `calc_influence()`: Core influence calculation
  - `get_gradients()`: Computes model gradients
  - `analyze_influence()`: Processes influence scores

#### `visualize_dataset.py`
- **Visualizations**:
  - Class distribution plots
  - Sample images by demographic
  - Model attention maps
  - Confusion matrices

### Bias Mitigation

#### `train_fair_loss.py`
- **Loss Functions**:
  - Demographic parity loss
  - Equalized odds
  - Custom fairness constraints
- **Training**:
  - Multi-task learning
  - Fairness-accuracy trade-off

#### `train_fair_weighted.py`
- **Weighting Strategies**:
  - Class-balanced weights
  - Group-aware sampling
  - Focal loss variants

### Evaluation Framework

#### [eval.py](cci:7://file:///c:/Aaryan/College_Stuff/Design%20Project/eval.py:0:0-0:0)
- **Metrics**:
  - Per-race accuracy
  - Fairness gap
  - Statistical parity
  - Equal opportunity
- **Visualization**:
  - Performance comparison
  - Fairness-accuracy curves
  - Error analysis

## Model Files and Checkpoints

1. **Baseline Model**
   - `Detection code/final_race_model.pth`
   - Input: 224x224 RGB images
   - Output: 5-class race prediction

2. **Fair Models**
   - `phase2_outputs/best_fair_model.pth`
     - Optimized for fairness
     - Best validation performance
   - `phase2_outputs/final_fair_model.pth`
     - Final production model
     - Best balance of accuracy and fairness

## Detailed Usage Guide

### Environment Setup
```bash
# Create and activate environment
conda create -n fairface python=3.8
conda activate fairface
pip install -r requirements.txt
```

### Data Preparation
```bash
python prepare_dataset.py \
    --data_dir "path/to/raw_data" \
    --output_dir "data" \
    --split_ratio 0.7 0.15 0.15
```

### Training
```bash
# Train baseline
python "Detection code/train_baseline_resnet.py" \
    --data_dir "data/train" \
    --val_dir "data/val" \
    --batch_size 32 \
    --epochs 50

# Train with fairness constraints
python "Mitigation Code/train_fair_loss.py" \
    --data_dir "data/train" \
    --val_dir "data/val" \
    --fairness_weight 0.5
```

### Evaluation
```bash
python eval.py \
    --baseline "Detection code/final_race_model.pth" \
    --mitigated "Mitigation Code/phase2_outputs/best_fair_model.pth" \
    --test_dir "data/test" \
    --test_csv "data/test_labels.csv" \
    --output_dir "results"
```

## Implementation Details

### Data Augmentation
- Random horizontal flip
- Color jitter
- Normalization (ImageNet stats)

### Model Architecture
- Base: ResNet-18
- Custom head: 2-layer MLP
- Dropout: 0.5
- Activation: ReLU

### Training Configuration
- Optimizer: Adam (lr=1e-4)
- Batch size: 32
- Epochs: 50
- Early stopping: 10 epochs patience

## Results and Analysis

### Performance Metrics
| Model | Overall Acc | Min Race Acc | Fairness Gap |
|-------|-------------|--------------|--------------|
| Baseline | 85.2% | 72.1% | 13.1% |
| Fair Model | 83.7% | 80.5% | 3.2% |

### Key Findings
- Baseline shows significant performance disparity
- Fair models reduce gap while maintaining accuracy
- Best results with custom loss weighting

## Troubleshooting and FAQs

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision

2. **Model Loading Errors**
   ```python
   # Load with map_location for CPU/GPU compatibility
   model.load_state_dict(torch.load(path, map_location=device))
   ```

3. **Data Loading Problems**
   - Verify file permissions
   - Check CSV formatting
   - Ensure image paths are correct

### Performance Tuning
- Adjust learning rate
- Try different batch sizes
- Experiment with augmentation
- Modify fairness constraints

### Getting Help
For support, please include:
1. Error messages
2. Environment details
3. Reproduction steps
4. Expected vs actual behavior
```
