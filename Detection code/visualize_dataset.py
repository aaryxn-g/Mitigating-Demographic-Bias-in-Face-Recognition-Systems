import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import numpy as np

# Configuration
image_dir = r"C:\Aaryan\College_Stuff\design proj\crop_part1"
race_names = ['Caucasian', 'African American', 'Asian', 'Indian', 'Others']
output_dir = "analysis_results"
os.makedirs(output_dir, exist_ok=True)

# Data Collection
race_data = []
for fname in os.listdir(image_dir):
    if fname.endswith('.jpg'):
        try:
            parts = fname.split('_')
            if len(parts) >= 3:
                race_label = int(parts[2])
                if 0 <= race_label <= 4:  # Validate race label
                    race_data.append({
                        'file': os.path.join(image_dir, fname),
                        'race': race_label
                    })
        except (ValueError, IndexError):
            continue

df = pd.DataFrame(race_data)

# 1. Race Distribution - Enhanced Bar Plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=race_names, y=df['race'].value_counts().sort_index().values,
                palette="viridis")
plt.title('Race Distribution in Dataset', fontsize=14, pad=20)
plt.xlabel('Race', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Add count labels on top of bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black',
                xytext=(0, 5),
                textcoords='offset points')
    
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'race_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

# 2. Race Distribution - Donut Chart
plt.figure(figsize=(8, 8))
race_counts = df['race'].value_counts().sort_index()
plt.pie(race_counts, labels=race_names, autopct='%1.1f%%',
       colors=sns.color_palette('viridis', len(race_names)),
       startangle=90, wedgeprops=dict(width=0.4, edgecolor='w'))

# Draw a circle at the center to make it a donut chart
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Race Distribution (Percentage)', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'race_distribution_donut.png'), dpi=300, bbox_inches='tight')
plt.show()

# 3. Sample Images Visualization
def plot_race_samples(race_id, race_name, n_samples=8):
    race_samples = df[df['race'] == race_id].sample(n_samples, random_state=42)
    
    plt.figure(figsize=(15, 3))
    plt.suptitle(f'Sample Images - {race_name}', y=1.05, fontsize=14)
    
    for idx, (_, row) in enumerate(race_samples.iterrows()):
        try:
            img = Image.open(row['file'])
            plt.subplot(1, n_samples, idx + 1)
            plt.imshow(img)
            plt.axis('off')
        except:
            continue
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'samples_{race_name.lower().replace(" ", "_")}.png'),
               dpi=150, bbox_inches='tight')
    plt.show()

# Generate sample images for each race
for race_id, race_name in enumerate(race_names):
    plot_race_samples(race_id, race_name)

print(f"Analysis complete! Results saved in '{output_dir}' directory.")