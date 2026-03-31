"""Compute influence-based sample weights for fairness-aware retraining."""

import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INFLUENCE_CSV = ROOT_DIR / 'Detection code' / 'all_influence' / 'full_dataset_influence_scores.csv'


class SampleWeighter:
    """Compute sample weights that boost minority groups and dampen harmful majority samples."""
    
    # Constants
    MIN_WEIGHT = 0.1  # Minimum weight to prevent zero weights
    MAX_WEIGHT = 10.0  # Maximum weight to prevent extreme values
    
    RACE_GROUPS = {
        'Caucasian': 0,
        'African American': 1,
        'Asian': 2,
        'Indian': 3,
        'Others': 4
    }
    
    # Define minority and majority groups
    MINORITY_RACES = ['African American', 'Asian', 'Indian']
    MAJORITY_RACES = ['Caucasian', 'Others']
    
    def __init__(self, influence_csv_path: str, upweight_factor: float = 2.0, 
                 downweight_factor: float = 0.5):
        """Initialize the sample weighter with influence scores.
        
        Args:
            influence_csv_path: Path to CSV containing influence scores and metadata
            upweight_factor: Multiplier for minority group samples (>1.0)
            downweight_factor: Multiplier for majority group negative influence (0-1)
            
        Raises:
            FileNotFoundError: If influence_csv_path doesn't exist
            ValueError: If required columns are missing from the CSV
        """
        if not os.path.exists(influence_csv_path):
            raise FileNotFoundError(f"Influence scores file not found: {influence_csv_path}")
            
        try:
            self.influence_df = pd.read_csv(influence_csv_path)
            required_cols = {'file_path', 'influence'}
            if not required_cols.issubset(self.influence_df.columns):
                missing = required_cols - set(self.influence_df.columns)
                raise ValueError(f"Missing required columns: {missing}")
                
            self.upweight_factor = max(1.0, float(upweight_factor))
            self.downweight_factor = min(max(0.0, float(downweight_factor)), 1.0)
            
            # Initialize race groups
            self.race_groups = {
                'Caucasian': 0,
                'African American': 1,
                'Asian': 2,
                'Indian': 3,
                'Others': 4
            }
            self.idx_to_race = {v: k for k, v in self.race_groups.items()}
            
            self.minority_races = ['African American', 'Asian', 'Indian']
            self.majority_races = ['Caucasian', 'Others']
            
            logger.info(f"Initialized SampleWeighter with {len(self.influence_df)} samples")
            logger.info(f"Upweight factor: {self.upweight_factor}, "
                      f"Downweight factor: {self.downweight_factor}")
                      
        except Exception as e:
            logger.error(f"Failed to initialize SampleWeighter: {str(e)}")
            raise
        
    def extract_race_label(self, filepath):
        """
        Extract race label from UTKFace-style filenames (age_gender_race_timestamp.jpg).
        Falls back to path-based heuristic if the filename pattern cannot be parsed.
        """
        filename = os.path.basename(filepath)
        core = filename.split('.')[0]
        parts = core.split('_')
        if len(parts) >= 3 and parts[2].isdigit():
            race_idx = int(parts[2])
            return self.idx_to_race.get(race_idx, 'Others')
        
        # Fallback: check folder names / other hints
        normalized = filepath.replace('\\', '/').split('/')
        for part in normalized:
            for race_name in self.RACE_GROUPS.keys():
                if race_name.lower().replace(' ', '') in part.lower():
                    return race_name
        return 'Others'
    
    def _race_from_row(self, row) -> str:
        if 'race' not in row:
            return None
        value = row['race']
        if isinstance(value, str):
            value = value.strip()
            if value in self.race_groups:
                return value
            if value.isdigit():
                return self.idx_to_race.get(int(value), None)
        if isinstance(value, (int, float)) and not np.isnan(value):
            return self.idx_to_race.get(int(value), None)
        return None
    
    def compute_weights(self, output_csv: str = 'sample_weights.csv',
                       min_weight: float = 0.1, max_weight: float = 10.0) -> pd.DataFrame:
        """Return a DataFrame with file_path, influence, race, and normalized weights."""
        try:
            weights_data = []
            missing_race_count = 0
            
            for idx, row in tqdm(self.influence_df.iterrows(), 
                               total=len(self.influence_df),
                               desc="Computing sample weights"):
                file_path = row['file_path']
                influence = float(row['influence'])
                race = self._race_from_row(row) or self.extract_race_label(file_path)
                
                if race not in self.race_groups:
                    missing_race_count += 1
                    race = 'Others'  # Default to 'Others' for unknown races
                
                # Compute base weight based on race and influence
                if race in self.minority_races:
                    # Amplify minority samples, especially high-influence ones
                    weight = 1.0 + (abs(influence) * self.upweight_factor)
                elif race in self.majority_races and influence < 0:
                    # Downweight negative influence from majority samples
                    weight = 1.0 + (influence * self.downweight_factor)
                else:
                    # Neutral weight for other cases
                    weight = 1.0
                
                # Clip weights to reasonable range
                weight = np.clip(weight, min_weight, max_weight)
                
                weights_data.append({
                    'file_path': file_path,
                    'influence': influence,
                    'race': race,
                    'weight': weight
                })
            
            weights_df = pd.DataFrame(weights_data)
            
            # Normalize weights to have mean = 1.0
            weight_mean = weights_df['weight'].mean()
            if weight_mean > 0:
                weights_df['weight'] = weights_df['weight'] / weight_mean
            
            # Save results
            os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
            weights_df.to_csv(output_csv, index=False)
            
            # Log statistics
            logger.info(f"Computed weights for {len(weights_df)} samples")
            if missing_race_count > 0:
                logger.warning(f"Could not determine race for {missing_race_count} samples")
                
            # Save weight statistics
            stats = weights_df['weight'].describe().to_dict()
            stats['race_distribution'] = weights_df['race'].value_counts().to_dict()
            stats_path = output_csv.replace('.csv', '_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
                
            logger.info(f"Weight stats saved to {stats_path}")
            
            return weights_df
            
        except Exception as e:
            logger.error(f"Error computing weights: {str(e)}")
            raise ValueError(f"Failed to compute weights: {str(e)}")
    
    def visualize_weights(self, weights_df: pd.DataFrame,
                         output_plot: str = 'weight_distribution.png') -> None:
        """Plot per-race distributions plus influence→weight scatter for quick QA."""
        try:
            # Use a modern style that's widely available
            plt.style.use('default')
            sns.set_theme(style="whitegrid")
            fig = plt.figure(figsize=(18, 12))
            gs = fig.add_gridspec(2, 2)
            
            # 1. Boxplot of weight distribution by race
            ax1 = fig.add_subplot(gs[0, 0])
            sns.boxplot(data=weights_df, x='race', y='weight', ax=ax1, 
                       order=sorted(weights_df['race'].unique()))
            ax1.set_title('(a) Sample Weight Distribution by Race', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Weight')
            ax1.set_xlabel('')
            ax1.tick_params(axis='x', rotation=45)
            ax1.axhline(1.0, color='r', linestyle='--', alpha=0.7)
            
            # 2. Scatter plot of influence vs weight
            ax2 = fig.add_subplot(gs[0, 1])
            races = sorted(weights_df['race'].unique())
            colors = plt.cm.tab10(np.linspace(0, 1, len(races)))
            
            for race, color in zip(races, colors):
                subset = weights_df[weights_df['race'] == race]
                ax2.scatter(subset['influence'], subset['weight'], 
                           alpha=0.6, label=race, s=20, color=color)
            
            ax2.axhline(1.0, color='r', linestyle='--', alpha=0.7, label='Baseline (1.0)')
            ax2.set_xlabel('Influence Score')
            ax2.set_ylabel('Computed Weight')
            ax2.set_title('(b) Influence Score → Weight Mapping', fontsize=12, fontweight='bold')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # 3. Histogram of weights by race
            ax3 = fig.add_subplot(gs[1, 0])
            for race, color in zip(races, colors):
                subset = weights_df[weights_df['race'] == race]
                sns.kdeplot(data=subset, x='weight', label=race, ax=ax3, 
                           color=color, fill=True, alpha=0.3, linewidth=1.5)
            
            ax3.set_xlabel('Weight')
            ax3.set_ylabel('Density')
            ax3.set_title('(c) Weight Distribution by Race', fontsize=12, fontweight='bold')
            ax3.legend(title='Race')
            
            # 4. ECDF of weights
            ax4 = fig.add_subplot(gs[1, 1])
            for race, color in zip(races, colors):
                subset = weights_df[weights_df['race'] == race]
                sns.ecdfplot(data=subset, x='weight', label=race, ax=ax4, 
                            color=color, linewidth=2)
            
            ax4.set_xlabel('Weight')
            ax4.set_ylabel('Cumulative Probability')
            ax4.set_title('(d) Cumulative Distribution of Weights', fontsize=12, fontweight='bold')
            
            # Add title and adjust layout
            plt.suptitle('Influence-Based Sample Weight Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save figure
            os.makedirs(os.path.dirname(os.path.abspath(output_plot)), exist_ok=True)
            plt.savefig(output_plot, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved weight visualization to {output_plot}")
            
        except Exception as e:
            logger.error(f"Error generating weight visualizations: {str(e)}")
            raise
    
    def compute_race_statistics(self, weights_df: pd.DataFrame) -> dict:
        """Summarize weight stats per race plus overall fairness indicators."""
        try:
            # Import scipy stats here to avoid dependency if not needed
            from scipy import stats as sp_stats
            
            # Define custom aggregation functions
            def q25(x):
                return x.quantile(0.25)
                
            def q75(x):
                return x.quantile(0.75)
                
            def calc_skew(x):
                return sp_stats.skew(x)
                
            def calc_kurtosis(x):
                return sp_stats.kurtosis(x, fisher=False)  # Fisher=False for Pearson's definition
            
            # Basic statistics by race
            race_stats = weights_df.groupby('race')['weight'].agg([
                ('count', 'count'),
                ('mean', 'mean'),
                ('std', 'std'),
                ('min', 'min'),
                ('25%', q25),
                ('50%', 'median'),
                ('75%', q75),
                ('max', 'max'),
                ('skew', calc_skew),
                ('kurtosis', calc_kurtosis)
            ]).round(4).to_dict('index')
            
            # Overall statistics
            overall_stats = {
                'overall_mean': weights_df['weight'].mean(),
                'overall_std': weights_df['weight'].std(),
                'min_weight': weights_df['weight'].min(),
                'max_weight': weights_df['weight'].max(),
                'num_samples': len(weights_df),
                'num_races': len(race_stats)
            }
            
            # Fairness metrics
            mean_weights = weights_df.groupby('race')['weight'].mean()
            min_mean = mean_weights.min()
            max_mean = mean_weights.max()
            
            fairness_metrics = {
                'max_disparity_ratio': float(max_mean / (min_mean + 1e-10)),
                'min_disparity_ratio': float(min_mean / (max_mean + 1e-10)),
                'mean_disparity': float(mean_weights.std() / mean_weights.mean()) if mean_weights.mean() > 0 else 0,
                'gini_coefficient': self._compute_gini(weights_df['weight'])
            }
            
            # Combine all statistics
            result = {}
            result['race_statistics'] = race_stats
            result['overall_statistics'] = overall_stats
            result['fairness_metrics'] = fairness_metrics
            
            logger.info(
                "Weight summary → samples:%d mean:%.4f gini:%.4f",
                overall_stats['num_samples'],
                overall_stats['overall_mean'],
                fairness_metrics['gini_coefficient']
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error computing statistics: {str(e)}")
            raise
    
    def _compute_gini(self, x: np.ndarray) -> float:
        """Gini coefficient (0 = equal weights, 1 = max inequality)."""
        # Sort and normalize
        x = np.sort(x)
        n = len(x)
        if n == 0 or np.allclose(x, 0):
            return 0.0
            
        # Compute Gini coefficient
        cumx = np.cumsum(x, dtype=float)
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compute influence-based sample weights for fairness.'
    )
    parser.add_argument('--influence_csv', type=str, default=str(DEFAULT_INFLUENCE_CSV),
                      help='Path to input influence scores CSV')
    parser.add_argument('--output_weights', type=str, default='fairness_sample_weights.csv',
                      help='Output path for computed weights (CSV)')
    parser.add_argument('--output_plot', type=str, default='weight_distribution_analysis.png',
                      help='Output path for weight distribution plot')
    parser.add_argument('--upweight_factor', type=float, default=2.0,
                      help='Multiplier for minority group samples')
    parser.add_argument('--downweight_factor', type=float, default=0.5,
                      help='Multiplier for majority group negative influence')
    parser.add_argument('--min_weight', type=float, default=0.1,
                      help='Minimum allowed weight')
    parser.add_argument('--max_weight', type=float, default=10.0,
                      help='Maximum allowed weight')
    parser.add_argument('--log_level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='Logging level')
    
    return parser.parse_args()


def main():
    """Main function for computing influence-based sample weights."""
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Using influence scores from %s", args.influence_csv)
        weighter = SampleWeighter(
            influence_csv_path=args.influence_csv,
            upweight_factor=args.upweight_factor,
            downweight_factor=args.downweight_factor
        )
        
        weights_df = weighter.compute_weights(
            output_csv=args.output_weights,
            min_weight=args.min_weight,
            max_weight=args.max_weight
        )
        
        weighter.visualize_weights(weights_df, output_plot=args.output_plot)
        
        stats = weighter.compute_race_statistics(weights_df)
        
        stats_path = args.output_weights.replace('.csv', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(
            "Done. CSV:%s plot:%s stats:%s",
            args.output_weights,
            args.output_plot,
            stats_path
        )
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    import sys
    main()
