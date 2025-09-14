# UTKFace Age Bias Detection and Mitigation

This project analyzes age-related biases in the UTKFace dataset and provides insights for bias mitigation in facial recognition systems.

## Project Overview

The UTKFace dataset contains face images with labels for age, gender, and ethnicity. This analysis focuses specifically on understanding age distribution patterns and identifying potential biases that could affect machine learning model performance.

## Files

- `understanding.py` - Main analysis script for UTKFace age bias detection
- `requirements.txt` - Python dependencies
- `utkface_age_analysis.csv` - Generated analysis results
- `draft.py` - Initial exploration code

## Features

### Age Distribution Analysis
- Statistical summary of age labels (range, mean, median, std dev)
- Age group breakdown (Children, Teenagers, Young Adults, Middle-aged, Seniors)
- Identification of most/least represented ages

### Bias Detection
- Coefficient of variation analysis
- Under/over-represented age groups identification
- Missing age gaps detection
- Statistical bias indicators

### Visualizations
- Age distribution histograms with density curves
- Box plots showing quartiles and outliers
- Age group bar charts
- Cumulative distribution plots
- Gender-based age distribution comparison

### ML Implications Analysis
- Performance bias predictions
- Generalization concerns
- Ethical considerations
- Bias mitigation recommendations

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the analysis:
   ```bash
   python understanding.py
   ```

3. Enter the path to your UTKFace dataset when prompted

## Dataset Structure

The script expects UTKFace images with filenames in the format:
```
[age]_[gender]_[race]_[date&time].jpg
```

Where:
- `age`: 0-116 years
- `gender`: 0 (male) or 1 (female)
- `race`: 0 (White), 1 (Black), 2 (Asian), 3 (Indian), 4 (Others)

## Key Findings (crop_part1 analysis)

- **Total samples**: 9,778 images
- **Age range**: 1-110 years
- **Strong young-age bias**: 33.3% are children (0-12 years)
- **Overrepresented**: Age 1 (1,112 samples)
- **Missing ages**: 11 age groups (94+ years mostly absent)
- **High variation**: Coefficient of variation = 0.842

## Bias Mitigation Recommendations

### Data Collection
- Collect more samples from underrepresented age groups
- Ensure balanced representation across all target ages
- Consider stratified sampling strategies

### Technical Solutions
- Use class weighting to balance training
- Apply data augmentation techniques
- Consider ensemble methods with age-specific models
- Implement fairness-aware training algorithms

### Evaluation
- Report performance metrics per age group
- Use fairness metrics (demographic parity, equalized odds)
- Conduct bias testing across different age ranges

## Research Applications

This analysis is valuable for:
- Bias detection research in facial recognition
- Fairness-aware machine learning
- Dataset quality assessment
- Ethical AI development
- Age estimation model evaluation

## Note on Dataset Files

Dataset image files are excluded from this repository due to size constraints. To run the analysis, download the UTKFace dataset separately and place it in your local directory.
