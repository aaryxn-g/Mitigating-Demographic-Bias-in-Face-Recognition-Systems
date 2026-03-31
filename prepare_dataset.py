import os
import csv
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Define paths
base_dir = Path(".")  # Current directory
data_dir = base_dir / "crop_part1"
output_dir = base_dir / "data"

# Create output directories
(output_dir / "train").mkdir(parents=True, exist_ok=True)
(output_dir / "val").mkdir(parents=True, exist_ok=True)
(output_dir / "test").mkdir(parents=True, exist_ok=True)

# Race mapping (from UTKFace documentation)
RACE_MAPPING = {
    '0': 'Caucasian',  # Changed from 'White' to 'Caucasian' to match eval.py
    '1': 'African American',  # Changed from 'Black' to 'African American'
    '2': 'Asian',
    '3': 'Indian',
    '4': 'Others'
}

# Collect all image files
image_files = [f for f in data_dir.glob("*.jpg") if f.is_file()]

# Shuffle the files
random.shuffle(image_files)

# Calculate split sizes
total = len(image_files)
train_size = int(0.7 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size

# Split the data
train_files = image_files[:train_size]
val_files = image_files[train_size:train_size + val_size]
test_files = image_files[train_size + val_size:]

def write_csv(split_name, files):
    csv_path = output_dir / f"{split_name}_labels.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'race'])
        
        for file in files:
            # Extract race from filename (format: age_gender_race_date.jpg)
            try:
                race_code = file.stem.split('_')[2]
                race = RACE_MAPPING.get(race_code, 'Unknown')
                
                # Copy file to the appropriate directory
                dest = output_dir / split_name / file.name
                os.link(file, dest)  # Creates a hard link instead of copying
                
                # Write to CSV
                writer.writerow([file.name, race])
            except (IndexError, KeyError):
                print(f"Skipping file with unexpected name format: {file.name}")
                continue

# Process each split
print(f"Processing train split ({len(train_files)} images)...")
write_csv('train', train_files)

print(f"Processing validation split ({len(val_files)} images)...")
write_csv('val', val_files)

print(f"Processing test split ({len(test_files)} images)...")
write_csv('test', test_files)

print("Dataset preparation complete!")
print(f"Total images: {total}")
print(f"Train: {len(train_files)} ({len(train_files)/total:.1%})")
print(f"Validation: {len(val_files)} ({len(val_files)/total:.1%})")
print(f"Test: {len(test_files)} ({len(test_files)/total:.1%})")
