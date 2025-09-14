"""
UTKFace Dataset Age Distribution Analysis
=========================================

This script analyzes the age distribution and potential age-related biases 
in the UTKFace dataset. The UTKFace dataset contains face images with 
filenames in the format: [age]_[gender]_[race]_[date&time].jpg

Author: Analysis for understanding age bias in facial recognition datasets
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

def parse_utkface_filenames(dataset_path):
    """
    Parse UTKFace dataset filenames to extract age, gender, and race information.
    
    UTKFace filename format: [age]_[gender]_[race]_[date&time].jpg
    - age: integer from 0 to 116
    - gender: 0 (male) or 1 (female)  
    - race: 0 (White), 1 (Black), 2 (Asian), 3 (Indian), 4 (Others)
    
    Args:
        dataset_path (str): Path to the UTKFace dataset directory
        
    Returns:
        pandas.DataFrame: DataFrame with columns ['filename', 'age', 'gender', 'race']
    """
    
    # Step 1: Get all image files from the dataset directory
    print("🔍 Scanning dataset directory for image files...")
    
    # Create Path object for easier file handling
    dataset_dir = Path(dataset_path)
    
    # Find all .jpg files (UTKFace uses .jpg extension)
    image_files = list(dataset_dir.glob("*.jpg"))
    
    print(f"📁 Found {len(image_files)} image files")
    
    # Step 2: Parse each filename to extract labels
    parsed_data = []
    invalid_files = []
    
    for img_file in image_files:
        filename = img_file.name
        
        try:
            # Split filename by underscore and extract components
            # Example: "20_1_0_20170109142408.jpg" -> ["20", "1", "0", "20170109142408.jpg"]
            parts = filename.split('_')
            
            # Extract age, gender, race from the first three parts
            age = int(parts[0])
            gender = int(parts[1])
            race = int(parts[2])
            
            # Validate the extracted values
            if age < 0 or age > 116:  # Age should be reasonable
                invalid_files.append(filename)
                continue
                
            if gender not in [0, 1]:  # Gender should be 0 or 1
                invalid_files.append(filename)
                continue
                
            if race not in [0, 1, 2, 3, 4]:  # Race should be 0-4
                invalid_files.append(filename)
                continue
            
            # Add valid data to our list
            parsed_data.append({
                'filename': filename,
                'age': age,
                'gender': gender,
                'race': race
            })
            
        except (ValueError, IndexError):
            # Handle files that don't follow the expected naming convention
            invalid_files.append(filename)
    
    print(f"✅ Successfully parsed {len(parsed_data)} files")
    if invalid_files:
        print(f"⚠️  Found {len(invalid_files)} invalid filenames")
    
    # Step 3: Create DataFrame from parsed data
    df = pd.DataFrame(parsed_data)
    
    return df

def analyze_age_distribution(df):
    """
    Analyze the age distribution in the dataset and provide statistical insights.
    
    Args:
        df (pandas.DataFrame): DataFrame with age information
        
    Returns:
        dict: Dictionary containing age distribution statistics
    """
    
    print("\n" + "="*50)
    print("📊 AGE DISTRIBUTION ANALYSIS")
    print("="*50)
    
    # Basic statistics about age distribution
    age_stats = {
        'total_samples': len(df),
        'min_age': df['age'].min(),
        'max_age': df['age'].max(),
        'mean_age': df['age'].mean(),
        'median_age': df['age'].median(),
        'std_age': df['age'].std(),
        'age_range': df['age'].max() - df['age'].min()
    }
    
    print(f"📈 Total number of samples: {age_stats['total_samples']:,}")
    print(f"🔢 Age range: {age_stats['min_age']} to {age_stats['max_age']} years")
    print(f"📊 Mean age: {age_stats['mean_age']:.1f} years")
    print(f"📊 Median age: {age_stats['median_age']:.1f} years")
    print(f"📊 Standard deviation: {age_stats['std_age']:.1f} years")
    
    # Age group analysis
    print(f"\n🎯 AGE GROUP BREAKDOWN:")
    
    # Define age groups for analysis
    age_groups = {
        'Children (0-12)': (0, 12),
        'Teenagers (13-19)': (13, 19),
        'Young Adults (20-35)': (20, 35),
        'Middle-aged (36-55)': (36, 55),
        'Seniors (56+)': (56, 200)
    }
    
    group_counts = {}
    for group_name, (min_age, max_age) in age_groups.items():
        count = len(df[(df['age'] >= min_age) & (df['age'] <= max_age)])
        percentage = (count / len(df)) * 100
        group_counts[group_name] = {'count': count, 'percentage': percentage}
        print(f"   {group_name}: {count:,} samples ({percentage:.1f}%)")
    
    age_stats['age_groups'] = group_counts
    
    # Find most and least represented ages
    age_value_counts = df['age'].value_counts().sort_index()
    most_common_age = age_value_counts.idxmax()
    least_common_age = age_value_counts.idxmin()
    
    print(f"\n🏆 Most represented age: {most_common_age} years ({age_value_counts[most_common_age]} samples)")
    print(f"🔻 Least represented age: {least_common_age} years ({age_value_counts[least_common_age]} samples)")
    
    return age_stats

def create_age_visualizations(df):
    """
    Create comprehensive visualizations for age distribution analysis.
    
    Args:
        df (pandas.DataFrame): DataFrame with age information
    """
    
    print("\n" + "="*50)
    print("📈 CREATING AGE DISTRIBUTION VISUALIZATIONS")
    print("="*50)
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Age Distribution Histogram
    plt.subplot(2, 3, 1)
    plt.hist(df['age'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Age Distribution Histogram', fontsize=14, fontweight='bold')
    plt.xlabel('Age (years)')
    plt.ylabel('Number of Samples')
    plt.grid(True, alpha=0.3)
    
    # Add mean and median lines
    plt.axvline(df['age'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["age"].mean():.1f}')
    plt.axvline(df['age'].median(), color='green', linestyle='--', 
                label=f'Median: {df["age"].median():.1f}')
    plt.legend()
    
    # 2. Age Distribution Density Plot
    plt.subplot(2, 3, 2)
    sns.histplot(data=df, x='age', kde=True, bins=50)
    plt.title('Age Distribution with Density Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Age (years)')
    plt.ylabel('Density')
    
    # 3. Box Plot for Age Distribution
    plt.subplot(2, 3, 3)
    box_plot = plt.boxplot(df['age'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightcoral')
    plt.title('Age Distribution Box Plot', fontsize=14, fontweight='bold')
    plt.ylabel('Age (years)')
    plt.xticks([1], ['All Ages'])
    
    # Add statistics text
    q1 = df['age'].quantile(0.25)
    q3 = df['age'].quantile(0.75)
    iqr = q3 - q1
    plt.text(1.1, df['age'].median(), f'Median: {df["age"].median():.1f}\nQ1: {q1:.1f}\nQ3: {q3:.1f}\nIQR: {iqr:.1f}', 
             verticalalignment='center')
    
    # 4. Age Groups Bar Chart
    plt.subplot(2, 3, 4)
    age_groups = {
        'Children\n(0-12)': (0, 12),
        'Teenagers\n(13-19)': (13, 19),
        'Young Adults\n(20-35)': (20, 35),
        'Middle-aged\n(36-55)': (36, 55),
        'Seniors\n(56+)': (56, 200)
    }
    
    group_names = []
    group_counts = []
    
    for group_name, (min_age, max_age) in age_groups.items():
        count = len(df[(df['age'] >= min_age) & (df['age'] <= max_age)])
        group_names.append(group_name)
        group_counts.append(count)
    
    bars = plt.bar(group_names, group_counts, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC'])
    plt.title('Sample Count by Age Group', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, group_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # 5. Cumulative Distribution
    plt.subplot(2, 3, 5)
    sorted_ages = np.sort(df['age'])
    cumulative_prob = np.arange(1, len(sorted_ages) + 1) / len(sorted_ages)
    plt.plot(sorted_ages, cumulative_prob, linewidth=2, color='purple')
    plt.title('Cumulative Age Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Age (years)')
    plt.ylabel('Cumulative Probability')
    plt.grid(True, alpha=0.3)
    
    # 6. Age Distribution by Gender (if gender data available)
    plt.subplot(2, 3, 6)
    if 'gender' in df.columns:
        # Create separate histograms for male and female
        male_ages = df[df['gender'] == 0]['age']
        female_ages = df[df['gender'] == 1]['age']
        
        plt.hist([male_ages, female_ages], bins=30, alpha=0.7, 
                label=['Male', 'Female'], color=['lightblue', 'pink'])
        plt.title('Age Distribution by Gender', fontsize=14, fontweight='bold')
        plt.xlabel('Age (years)')
        plt.ylabel('Number of Samples')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Gender data not available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Age Distribution by Gender', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Age distribution visualizations created successfully!")

def identify_age_bias(df):
    """
    Identify and analyze age-related biases in the dataset.
    
    Args:
        df (pandas.DataFrame): DataFrame with age information
        
    Returns:
        dict: Dictionary containing bias analysis results
    """
    
    print("\n" + "="*50)
    print("🎯 AGE BIAS ANALYSIS")
    print("="*50)
    
    bias_analysis = {}
    
    # 1. Statistical bias indicators
    age_counts = df['age'].value_counts().sort_index()
    
    # Calculate coefficient of variation (CV) to measure distribution uniformity
    cv = df['age'].std() / df['age'].mean()
    bias_analysis['coefficient_of_variation'] = cv
    
    print(f"📊 Coefficient of Variation: {cv:.3f}")
    if cv > 0.5:
        print("   ⚠️  High variation indicates potential age bias")
    else:
        print("   ✅ Moderate variation suggests reasonable age distribution")
    
    # 2. Identify underrepresented and overrepresented age groups
    mean_samples_per_age = len(df) / (df['age'].max() - df['age'].min() + 1)
    
    underrepresented_ages = []
    overrepresented_ages = []
    
    for age in range(df['age'].min(), df['age'].max() + 1):
        count = age_counts.get(age, 0)
        if count < mean_samples_per_age * 0.5:  # Less than 50% of average
            underrepresented_ages.append((age, count))
        elif count > mean_samples_per_age * 2:  # More than 200% of average
            overrepresented_ages.append((age, count))
    
    bias_analysis['underrepresented_ages'] = underrepresented_ages
    bias_analysis['overrepresented_ages'] = overrepresented_ages
    
    print(f"\n🔻 UNDERREPRESENTED AGES (< 50% of average {mean_samples_per_age:.0f} samples):")
    if underrepresented_ages:
        for age, count in underrepresented_ages[:10]:  # Show top 10
            print(f"   Age {age}: {count} samples")
        if len(underrepresented_ages) > 10:
            print(f"   ... and {len(underrepresented_ages) - 10} more ages")
    else:
        print("   ✅ No significantly underrepresented ages found")
    
    print(f"\n🔺 OVERREPRESENTED AGES (> 200% of average {mean_samples_per_age:.0f} samples):")
    if overrepresented_ages:
        for age, count in overrepresented_ages[:10]:  # Show top 10
            print(f"   Age {age}: {count} samples")
        if len(overrepresented_ages) > 10:
            print(f"   ... and {len(overrepresented_ages) - 10} more ages")
    else:
        print("   ✅ No significantly overrepresented ages found")
    
    # 3. Age gap analysis
    print(f"\n🕳️  AGE GAPS ANALYSIS:")
    missing_ages = []
    for age in range(df['age'].min(), df['age'].max() + 1):
        if age not in age_counts:
            missing_ages.append(age)
    
    bias_analysis['missing_ages'] = missing_ages
    
    if missing_ages:
        print(f"   ⚠️  Found {len(missing_ages)} missing ages: {missing_ages}")
    else:
        print("   ✅ No missing ages in the range")
    
    return bias_analysis

def ml_implications_analysis(bias_analysis, age_stats):
    """
    Analyze the implications of age bias for machine learning models.
    
    Args:
        bias_analysis (dict): Results from age bias analysis
        age_stats (dict): Age distribution statistics
    """
    
    print("\n" + "="*60)
    print("🤖 MACHINE LEARNING IMPLICATIONS OF AGE BIAS")
    print("="*60)
    
    print("📋 POTENTIAL IMPACTS ON ML MODELS:")
    print()
    
    # 1. Performance bias implications
    print("1️⃣  MODEL PERFORMANCE BIAS:")
    if bias_analysis['underrepresented_ages']:
        print("   ⚠️  Models may perform poorly on underrepresented age groups")
        print("   📉 Lower accuracy for ages with fewer training samples")
        print("   🎯 Consider data augmentation or synthetic data generation")
    
    if bias_analysis['overrepresented_ages']:
        print("   ⚠️  Models may be biased toward overrepresented age groups")
        print("   📈 Higher accuracy for well-represented ages")
        print("   ⚖️  Consider class balancing techniques")
    
    print()
    
    # 2. Generalization concerns
    print("2️⃣  GENERALIZATION CONCERNS:")
    cv = bias_analysis['coefficient_of_variation']
    if cv > 0.5:
        print("   ⚠️  High age distribution variation may limit generalization")
        print("   🌍 Model may not work well on populations with different age distributions")
    
    if bias_analysis['missing_ages']:
        print("   ⚠️  Missing age groups create blind spots in model knowledge")
        print("   🕳️  Model cannot make predictions for unrepresented ages")
    
    print()
    
    # 3. Ethical considerations
    print("3️⃣  ETHICAL CONSIDERATIONS:")
    print("   ⚖️  Age bias can lead to discriminatory outcomes")
    print("   👥 Certain age groups may be systematically disadvantaged")
    print("   📜 May violate fairness principles in AI applications")
    
    print()
    
    # 4. Recommendations
    print("4️⃣  RECOMMENDATIONS FOR BIAS MITIGATION:")
    print("   📊 Data Collection:")
    print("      • Collect more samples from underrepresented age groups")
    print("      • Ensure balanced representation across all target ages")
    print("      • Consider stratified sampling strategies")
    print()
    print("   🔧 Technical Solutions:")
    print("      • Use class weighting to balance training")
    print("      • Apply data augmentation techniques")
    print("      • Consider ensemble methods with age-specific models")
    print("      • Implement fairness-aware training algorithms")
    print()
    print("   📈 Evaluation:")
    print("      • Report performance metrics per age group")
    print("      • Use fairness metrics (demographic parity, equalized odds)")
    print("      • Conduct bias testing across different age ranges")

def main():
    """
    Main function to run the complete UTKFace age bias analysis.
    """
    
    print("🎭 UTKFace Dataset Age Bias Analysis")
    print("="*50)
    
    # STEP 1: Set the path to your UTKFace dataset
    # ⚠️  IMPORTANT: Change this path to match your dataset location
    dataset_path = input("📁 Enter the path to your UTKFace dataset directory: ").strip()
    
    # Alternative: Uncomment and modify the line below if you want to hardcode the path
    # dataset_path = r"C:\path\to\your\UTKFace\dataset"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset path '{dataset_path}' does not exist!")
        print("Please check the path and try again.")
        return
    
    try:
        # STEP 2: Parse the dataset filenames
        print(f"\n🔄 Starting analysis of UTKFace dataset at: {dataset_path}")
        df = parse_utkface_filenames(dataset_path)
        
        if df.empty:
            print("❌ No valid UTKFace files found in the specified directory!")
            return
        
        # STEP 3: Analyze age distribution
        age_stats = analyze_age_distribution(df)
        
        # STEP 4: Create visualizations
        create_age_visualizations(df)
        
        # STEP 5: Identify age bias
        bias_analysis = identify_age_bias(df)
        
        # STEP 6: Analyze ML implications
        ml_implications_analysis(bias_analysis, age_stats)
        
        # STEP 7: Save results to CSV for further analysis
        output_file = "utkface_age_analysis.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Analysis results saved to: {output_file}")
        
        print("\n✅ Analysis completed successfully!")
        print("📊 Check the generated plots and statistics above for insights.")
        
    except Exception as e:
        print(f"❌ An error occurred during analysis: {str(e)}")
        print("Please check your dataset path and file format.")

# BEGINNER'S GUIDE TO RUNNING THIS SCRIPT
"""
🚀 HOW TO USE THIS SCRIPT:

1. PREREQUISITES:
   Install required libraries by running in your terminal:
   pip install pandas matplotlib seaborn numpy

2. DATASET SETUP:
   - Download the UTKFace dataset
   - Extract all .jpg files to a single directory
   - Note the full path to this directory

3. RUNNING THE SCRIPT:
   - Run this script: python understanding.py
   - Enter the path to your UTKFace dataset when prompted
   - Wait for the analysis to complete
   - View the generated plots and statistics

4. UNDERSTANDING THE OUTPUT:
   - Statistical summary of age distribution
   - Multiple visualization plots
   - Bias analysis results
   - ML implications and recommendations
   - CSV file with parsed data for further analysis

5. INTERPRETING RESULTS:
   - Look for age groups with very few samples (bias)
   - Check if the distribution matches your target population
   - Consider the ML implications for your specific use case
"""

if __name__ == "__main__":
    main()
