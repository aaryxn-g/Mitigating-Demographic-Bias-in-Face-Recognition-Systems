"""
compute_sample_weights_balanced.py
===================================
Fixed sample weighting that prevents class collapse by:
1. Limiting weight ratios to prevent extreme imbalances
2. Using influence scores more carefully
3. Balancing fairness goals with utility maintenance
"""

import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INFLUENCE_CSV = ROOT_DIR / 'Detection code' / 'all_influence' / 'full_dataset_influence_scores.csv'


class BalancedSampleWeighter:
    """
    Compute balanced sample weights that improve fairness without causing class collapse.
    
    Key improvements over previous version:
    - Strict weight ratio limits (max 3:1 instead of 10:1)
    - More conservative upweighting of minorities
    - Prevents over-downweighting of majorities
    - Uses influence scores as refinement, not primary driver
    """
    
    # Stricter weight limits to prevent class collapse
    MIN_WEIGHT = 0.3  # Higher minimum (was 0.1)
    MAX_WEIGHT = 3.0  # Lower maximum (was 10.0)
    
    RACE_GROUPS = {
        'Caucasian': 0,
        'African American': 1,
        'Asian': 2,
        'Indian': 3,
        'Others': 4
    }
    
    # Define groups based on typical UTKFace distribution
    MINORITY_RACES = ['African American', 'Indian']
    MEDIUM_RACES = ['Asian']
    MAJORITY_RACES = ['Caucasian', 'Others']
    
    def __init__(self, influence_csv_path: str, 
                 upweight_factor: float = 1.5,  # Reduced from 2.0
                 downweight_factor: float = 0.8):  # Less aggressive (was 0.5)
        """
        Initialize with more conservative factors.
        
        Args:
            influence_csv_path: Path to influence scores CSV
            upweight_factor: Multiplier for minority groups (1.5 recommended)
            downweight_factor: Multiplier for majority downweighting (0.7-0.9 recommended)
        """
        if not os.path.exists(influence_csv_path):
            raise FileNotFoundError(f"File not found: {influence_csv_path}")
        
        self.influence_df = pd.read_csv(influence_csv_path)
        
        required_cols = {'file_path', 'influence'}
        if not required_cols.issubset(self.influence_df.columns):
            missing = required_cols - set(self.influence_df.columns)
            raise ValueError(f"Missing columns: {missing}")
        
        # Clamp factors to safe ranges
        self.upweight_factor = max(1.0, min(float(upweight_factor), 2.0))
        self.downweight_factor = max(0.5, min(float(downweight_factor), 1.0))
        
        self.idx_to_race = {v: k for k, v in self.RACE_GROUPS.items()}
        
        logger.info(f"Initialized BalancedSampleWeighter with {len(self.influence_df)} samples")
        logger.info(f"Upweight factor: {self.upweight_factor:.2f}, "
                   f"Downweight factor: {self.downweight_factor:.2f}")
    
    def extract_race_label(self, filepath):
        """Extract race from UTKFace filename."""
        filename = os.path.basename(filepath)
        core = filename.split('.')[0]
        parts = core.split('_')
        
        if len(parts) >= 3 and parts[2].isdigit():
            race_idx = int(parts[2])
            return self.idx_to_race.get(race_idx, 'Others')
        
        # Fallback: check path components
        normalized = filepath.replace('\\', '/').split('/')
        for part in normalized:
            for race_name in self.RACE_GROUPS.keys():
                if race_name.lower().replace(' ', '') in part.lower():
                    return race_name
        
        return 'Others'
    
    def _race_from_row(self, row) -> str:
        """Extract race from CSV row."""
        if 'race' not in row:
            return None
        
        value = row['race']
        
        if isinstance(value, str):
            value = value.strip()
            if value in self.RACE_GROUPS:
                return value
            if value.isdigit():
                return self.idx_to_race.get(int(value), None)
        
        if isinstance(value, (int, float)) and not np.isnan(value):
            return self.idx_to_race.get(int(value), None)
        
        return None
    
    def compute_weights(self, output_csv: str = 'fairness_sample_weights_balanced.csv') -> pd.DataFrame:
        """
        Compute balanced sample weights.
        
        Strategy:
        1. Start with class frequency-based weights (inverse frequency)
        2. Apply moderate adjustments for minorities vs majorities
        3. Use influence scores only for fine-tuning
        4. Enforce strict weight limits
        """
        logger.info("Computing balanced sample weights...")
        
        weights_data = []
        race_counts = {race: 0 for race in self.RACE_GROUPS.keys()}
        
        # First pass: count races
        for idx, row in self.influence_df.iterrows():
            file_path = row['file_path']
            race = self._race_from_row(row) or self.extract_race_label(file_path)
            
            if race not in self.RACE_GROUPS:
                race = 'Others'
            
            race_counts[race] += 1
        
        total_samples = sum(race_counts.values())
        logger.info(f"Race distribution: {race_counts}")
        
        # Compute class frequency weights (inverse frequency)
        class_weights = {}
        for race, count in race_counts.items():
            if count > 0:
                # Inverse frequency, but capped
                freq_weight = total_samples / (len(self.RACE_GROUPS) * count)
                # Don't let frequency weights get too extreme
                freq_weight = np.clip(freq_weight, 0.5, 2.0)
                class_weights[race] = freq_weight
            else:
                class_weights[race] = 1.0
        
        logger.info(f"Base class weights: {class_weights}")
        
        # Second pass: compute individual sample weights
        for idx, row in tqdm(self.influence_df.iterrows(), 
                            total=len(self.influence_df),
                            desc="Computing weights"):
            file_path = row['file_path']
            influence = float(row['influence'])
            race = self._race_from_row(row) or self.extract_race_label(file_path)
            
            if race not in self.RACE_GROUPS:
                race = 'Others'
            
            # Start with class frequency weight
            weight = class_weights[race]
            
            # Apply group-specific adjustments (much more moderate)
            if race in self.MINORITY_RACES:
                # Modest boost for true minorities
                # Use influence to refine, not dominate
                influence_boost = 0.2 * abs(influence) if influence > 0 else 0
                weight *= (1.0 + 0.3 * self.upweight_factor + influence_boost)
            
            elif race in self.MEDIUM_RACES:
                # Small boost for medium-sized groups
                influence_boost = 0.1 * abs(influence) if influence > 0 else 0
                weight *= (1.0 + 0.15 * self.upweight_factor + influence_boost)
            
            elif race in self.MAJORITY_RACES:
                # Gentle downweighting for majorities, only for negative influence
                if influence < 0:
                    weight *= (1.0 + 0.2 * influence * (1 - self.downweight_factor))
                else:
                    # Positive influence majorities stay near baseline
                    weight *= 1.0
            
            # Strictly enforce weight limits
            weight = np.clip(weight, self.MIN_WEIGHT, self.MAX_WEIGHT)
            
            weights_data.append({
                'file_path': file_path,
                'influence': influence,
                'race': race,
                'weight': weight
            })
        
        weights_df = pd.DataFrame(weights_data)
        
        # Final normalization to mean=1.0
        weight_mean = weights_df['weight'].mean()
        if weight_mean > 0:
            weights_df['weight'] = weights_df['weight'] / weight_mean
        
        # CRITICAL: Enforce strict ratio cap
        # No race should have mean weight more than 3x any other race
        race_mean_weights = weights_df.groupby('race')['weight'].mean()
        max_ratio = race_mean_weights.max() / race_mean_weights.min()
        
        if max_ratio > 3.0:
            logger.warning(f"⚠ Weight ratio {max_ratio:.2f} exceeds 3:1 limit. Applying ratio cap...")
            
            # Scale down the highest weights to enforce 3:1 ratio
            target_max = race_mean_weights.min() * 3.0
            scale_factor = target_max / race_mean_weights.max()
            
            for race in race_mean_weights.index:
                if race_mean_weights[race] > target_max:
                    mask = weights_df['race'] == race
                    weights_df.loc[mask, 'weight'] *= scale_factor
            
            # Renormalize to mean=1.0
            weights_df['weight'] = weights_df['weight'] / weights_df['weight'].mean()
            
            logger.info(f"✓ Ratio cap applied. New max ratio: {(weights_df.groupby('race')['weight'].mean().max() / weights_df.groupby('race')['weight'].mean().min()):.2f}:1")
        
        # Double-check limits after normalization
        weights_df['weight'] = weights_df['weight'].clip(
            self.MIN_WEIGHT / weight_mean, 
            self.MAX_WEIGHT / weight_mean
        )
        
        # Save results
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)) if os.path.dirname(output_csv) else '.', 
                   exist_ok=True)
        weights_df.to_csv(output_csv, index=False)
        
        # Log statistics per race
        logger.info("\nPer-race weight statistics:")
        for race in self.RACE_GROUPS.keys():
            race_weights = weights_df[weights_df['race'] == race]['weight']
            if len(race_weights) > 0:
                logger.info(f"  {race:20s}: count={len(race_weights):4d}, "
                          f"mean={race_weights.mean():.3f}, "
                          f"min={race_weights.min():.3f}, "
                          f"max={race_weights.max():.3f}")
        
        # Check weight ratio
        mean_weights = weights_df.groupby('race')['weight'].mean()
        max_ratio = mean_weights.max() / mean_weights.min()
        logger.info(f"\nMax/min mean weight ratio: {max_ratio:.2f}:1")
        
        if max_ratio > 5.0:
            logger.warning("⚠ Weight ratio exceeds 5:1. Consider adjusting factors.")
        
        # Save statistics
        stats = self.compute_race_statistics(weights_df)
        stats_path = output_csv.replace('.csv', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"✓ Weights saved to {output_csv}")
        logger.info(f"✓ Stats saved to {stats_path}")
        
        return weights_df
    
    def visualize_weights(self, weights_df: pd.DataFrame,
                         output_plot: str = 'weight_distribution_balanced.png'):
        """Create comprehensive weight visualizations."""
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        race_names = sorted(weights_df['race'].unique())
        colors = plt.cm.Set2(np.linspace(0, 1, len(race_names)))
        
        # 1. Boxplot of weights by race
        ax1 = fig.add_subplot(gs[0, 0])
        sns.boxplot(data=weights_df, x='race', y='weight', ax=ax1, 
                   order=race_names, palette='Set2')
        ax1.set_title('Sample Weight Distribution by Race', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Weight')
        ax1.set_xlabel('')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(1.0, color='r', linestyle='--', alpha=0.7, label='Mean (1.0)')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Influence vs weight scatter
        ax2 = fig.add_subplot(gs[0, 1])
        for race, color in zip(race_names, colors):
            subset = weights_df[weights_df['race'] == race]
            ax2.scatter(subset['influence'], subset['weight'], 
                       alpha=0.5, label=race, s=15, color=color)
        ax2.axhline(1.0, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Influence Score')
        ax2.set_ylabel('Computed Weight')
        ax2.set_title('Influence → Weight Mapping', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(alpha=0.3)
        
        # 3. Mean weights by race
        ax3 = fig.add_subplot(gs[0, 2])
        mean_weights = weights_df.groupby('race')['weight'].mean().sort_values()
        mean_weights.plot(kind='barh', ax=ax3, color='skyblue', edgecolor='black')
        ax3.axvline(1.0, color='r', linestyle='--', alpha=0.7, label='Overall Mean')
        ax3.set_xlabel('Mean Weight')
        ax3.set_title('Mean Weight by Race', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Weight distribution histograms
        ax4 = fig.add_subplot(gs[1, 0])
        for race, color in zip(race_names, colors):
            subset = weights_df[weights_df['race'] == race]
            ax4.hist(subset['weight'], bins=30, alpha=0.5, label=race, color=color)
        ax4.set_xlabel('Weight')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Weight Distribution by Race', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Sample counts by race
        ax5 = fig.add_subplot(gs[1, 1])
        counts = weights_df['race'].value_counts().sort_index()
        counts.plot(kind='bar', ax=ax5, color='coral', edgecolor='black')
        ax5.set_xlabel('Race')
        ax5.set_ylabel('Number of Samples')
        ax5.set_title('Sample Distribution', fontsize=12, fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Weight statistics table
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        stats_text = "Weight Statistics\n" + "="*40 + "\n"
        for race in race_names:
            race_weights = weights_df[weights_df['race'] == race]['weight']
            stats_text += f"\n{race}:\n"
            stats_text += f"  Count: {len(race_weights)}\n"
            stats_text += f"  Mean:  {race_weights.mean():.3f}\n"
            stats_text += f"  Min:   {race_weights.min():.3f}\n"
            stats_text += f"  Max:   {race_weights.max():.3f}\n"
        
        mean_by_race = weights_df.groupby('race')['weight'].mean()
        ratio = mean_by_race.max() / mean_by_race.min()
        stats_text += f"\n{'='*40}\n"
        stats_text += f"Max/Min Ratio: {ratio:.2f}:1\n"
        
        ax6.text(0.05, 0.95, stats_text, fontsize=9, verticalalignment='top',
                fontfamily='monospace', 
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle('Balanced Sample Weight Analysis', fontsize=16, fontweight='bold')
        
        os.makedirs(os.path.dirname(os.path.abspath(output_plot)) if os.path.dirname(output_plot) else '.', 
                   exist_ok=True)
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Visualization saved to {output_plot}")
    
    def compute_race_statistics(self, weights_df: pd.DataFrame) -> dict:
        """Compute detailed statistics."""
        from scipy import stats as sp_stats
        
        def q25(x):
            return x.quantile(0.25)
        
        def q75(x):
            return x.quantile(0.75)
        
        race_stats = weights_df.groupby('race')['weight'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('25%', q25),
            ('50%', 'median'),
            ('75%', q75),
            ('max', 'max')
        ]).round(4).to_dict('index')
        
        overall_stats = {
            'overall_mean': float(weights_df['weight'].mean()),
            'overall_std': float(weights_df['weight'].std()),
            'min_weight': float(weights_df['weight'].min()),
            'max_weight': float(weights_df['weight'].max()),
            'num_samples': int(len(weights_df)),
            'num_races': len(race_stats)
        }
        
        mean_weights = weights_df.groupby('race')['weight'].mean()
        
        fairness_metrics = {
            'max_mean_weight': float(mean_weights.max()),
            'min_mean_weight': float(mean_weights.min()),
            'weight_ratio': float(mean_weights.max() / mean_weights.min()),
            'coefficient_of_variation': float(mean_weights.std() / mean_weights.mean())
        }
        
        return {
            'race_statistics': race_stats,
            'overall_statistics': overall_stats,
            'fairness_metrics': fairness_metrics
        }


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compute balanced sample weights for fairness.'
    )
    parser.add_argument('--influence_csv', type=str, 
                       default=str(DEFAULT_INFLUENCE_CSV),
                       help='Path to influence scores CSV')
    parser.add_argument('--output_weights', type=str, 
                       default='fairness_sample_weights_balanced.csv',
                       help='Output path for weights CSV')
    parser.add_argument('--output_plot', type=str, 
                       default='weight_distribution_balanced.png',
                       help='Output path for visualization')
    parser.add_argument('--upweight_factor', type=float, default=1.5,
                       help='Upweight factor for minorities (1.2-1.8 recommended)')
    parser.add_argument('--downweight_factor', type=float, default=0.8,
                       help='Downweight factor for majorities (0.7-0.9 recommended)')
    
    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_args()
    
    try:
        logger.info(f"Loading influence scores from {args.influence_csv}")
        weighter = BalancedSampleWeighter(
            influence_csv_path=args.influence_csv,
            upweight_factor=args.upweight_factor,
            downweight_factor=args.downweight_factor
        )
        
        weights_df = weighter.compute_weights(output_csv=args.output_weights)
        weighter.visualize_weights(weights_df, output_plot=args.output_plot)
        
        logger.info("\n✓ Weight computation complete!")
        logger.info(f"  Weights: {args.output_weights}")
        logger.info(f"  Visualization: {args.output_plot}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())