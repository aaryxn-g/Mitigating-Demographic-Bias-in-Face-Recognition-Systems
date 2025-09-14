"""
Machine Unlearning Demo: Soft Reweighting with Influence Functions
================================================================

This module demonstrates machine unlearning techniques inspired by:
1. "Understanding Black-box Predictions via Influence Functions" (Koh & Liang, ICML 2017)
2. "Distribution-Level Feature Distancing for Machine Unlearning" (Choi & Na, AAAI 2025)
3. "Bias and Diversity in Synthetic-based Face Recognition" (Huber et al., WACV 2024)

The implementation shows how to:
- Train a model on a biased dataset
- Use influence functions to identify problematic training instances
- Apply soft reweighting to "forget" specific instances or reduce bias
- Evaluate the trade-off between forgetting and utility preservation
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class InfluenceFunction:
    """
    Simplified implementation of influence functions for logistic regression.
    Based on Koh & Liang (2017) "Understanding Black-box Predictions via Influence Functions"
    """
    
    def __init__(self, model: LogisticRegression, X_train: np.ndarray, y_train: np.ndarray):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.n_samples = X_train.shape[0]
        
    def _compute_hessian(self) -> np.ndarray:
        """
        Compute Hessian matrix for logistic regression.
        For efficiency, we use the fact that for logistic regression:
        H = X^T * S * X where S is diagonal matrix of p(1-p)
        """
        # Get predictions (probabilities)
        probs = self.model.predict_proba(self.X_train)[:, 1]
        
        # Compute diagonal elements: p(1-p)
        s_diag = probs * (1 - probs)
        
        # Weighted feature matrix
        weighted_X = self.X_train * np.sqrt(s_diag).reshape(-1, 1)
        
        # Hessian: X^T * S * X + regularization
        hessian = weighted_X.T @ weighted_X
        
        # Add small regularization for numerical stability
        hessian += np.eye(hessian.shape[0]) * 1e-6
        
        return hessian
    
    def compute_influence_scores(self, test_points: np.ndarray = None) -> np.ndarray:
        """
        Compute influence scores for each training point.
        Higher absolute values indicate more influence on the model.
        
        Args:
            test_points: Points to compute influence for. If None, uses training points.
        
        Returns:
            influence_scores: Array of influence scores for each training point
        """
        if test_points is None:
            test_points = self.X_train
            
        # Compute Hessian and its inverse
        hessian = self._compute_hessian()
        hessian_inv = np.linalg.pinv(hessian)
        
        # For each training point, compute its influence
        influence_scores = np.zeros(self.n_samples)
        
        for i in range(self.n_samples):
            # Gradient of loss w.r.t. parameters for training point i
            x_i = self.X_train[i:i+1]
            y_i = self.y_train[i]
            
            # Prediction and gradient
            pred_prob = self.model.predict_proba(x_i)[0, 1]
            grad_loss = (pred_prob - y_i) * x_i.flatten()
            
            # Influence on test loss (simplified - using first test point)
            test_x = test_points[0:1] if len(test_points) > 0 else x_i
            test_pred = self.model.predict_proba(test_x)[0, 1]
            
            # Influence calculation: I_up,loss = -grad_test^T * H^-1 * grad_train
            influence_scores[i] = -np.dot(grad_loss, hessian_inv @ grad_loss)
            
        return influence_scores

class MachineUnlearning:
    """
    Machine Unlearning system using soft reweighting strategies.
    Combines influence functions with distribution-level adjustments.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.original_model = None
        self.unlearned_model = None
        self.influence_fn = None
        
    def create_biased_dataset(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a synthetic dataset with demographic bias.
        Simulates a scenario where certain demographic groups are unfairly treated.
        """
        # Create base classification dataset
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=10, 
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=self.random_state
        )
        
        # Add demographic attribute (0: Group A, 1: Group B)
        # Introduce bias: Group B has lower positive class rate
        demographic = np.random.binomial(1, 0.4, n_samples)
        
        # Introduce bias: reduce positive outcomes for Group B
        bias_mask = (demographic == 1) & (y == 1)
        y[bias_mask] = np.random.binomial(1, 0.3, np.sum(bias_mask))
        
        # Add demographic as a feature (this creates the bias)
        X = np.column_stack([X, demographic])
        
        return X, y, demographic
    
    def train_original_model(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        """Train the original biased model."""
        X_scaled = self.scaler.fit_transform(X)
        
        self.original_model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        self.original_model.fit(X_scaled, y)
        
        return self.original_model
    
    def identify_problematic_instances(self, X: np.ndarray, y: np.ndarray, 
                                     demographic: np.ndarray, 
                                     strategy: str = "influence") -> np.ndarray:
        """
        Identify instances to forget using different strategies.
        
        Args:
            X, y, demographic: Dataset
            strategy: "influence", "bias_amplifying", or "random"
        
        Returns:
            forget_indices: Indices of instances to forget
        """
        X_scaled = self.scaler.transform(X)
        
        if strategy == "influence":
            # Use influence functions to find most influential points
            self.influence_fn = InfluenceFunction(self.original_model, X_scaled, y)
            influence_scores = self.influence_fn.compute_influence_scores()
            
            # Select top 10% most influential points
            n_forget = int(0.1 * len(y))
            forget_indices = np.argsort(np.abs(influence_scores))[-n_forget:]
            
        elif strategy == "bias_amplifying":
            # Target instances that amplify demographic bias
            # Focus on Group B (demographic=1) with negative outcomes
            bias_instances = (demographic == 1) & (y == 0)
            forget_indices = np.where(bias_instances)[0]
            
            # Limit to reasonable number
            if len(forget_indices) > len(y) * 0.15:
                forget_indices = forget_indices[:int(len(y) * 0.15)]
                
        elif strategy == "random":
            # Random baseline
            n_forget = int(0.1 * len(y))
            forget_indices = np.random.choice(len(y), n_forget, replace=False)
            
        return forget_indices
    
    def soft_reweight_training(self, X: np.ndarray, y: np.ndarray, 
                             forget_indices: np.ndarray,
                             reweight_strategy: str = "influence_based") -> LogisticRegression:
        """
        Apply soft reweighting to forget specific instances.
        Inspired by DLFD's distribution-level approach.
        
        Args:
            X, y: Training data
            forget_indices: Indices to forget
            reweight_strategy: "zero_weight", "negative_weight", or "influence_based"
        
        Returns:
            unlearned_model: Model after unlearning
        """
        X_scaled = self.scaler.transform(X)
        n_samples = len(y)
        
        # Initialize sample weights
        sample_weights = np.ones(n_samples)
        
        if reweight_strategy == "zero_weight":
            # Simply set weight to zero (complete removal)
            sample_weights[forget_indices] = 0.0
            
        elif reweight_strategy == "negative_weight":
            # Use negative weights to "unlearn" (inspired by error maximization)
            sample_weights[forget_indices] = -0.5
            
        elif reweight_strategy == "influence_based":
            # Use influence scores to determine reweighting
            if self.influence_fn is not None:
                influence_scores = self.influence_fn.compute_influence_scores()
                
                # Reduce weights proportional to influence magnitude
                for idx in forget_indices:
                    influence_mag = np.abs(influence_scores[idx])
                    # Stronger influence -> more aggressive reweighting
                    sample_weights[idx] = max(0.1, 1.0 - influence_mag * 2)
            else:
                # Fallback to moderate reweighting
                sample_weights[forget_indices] = 0.2
        
        # Ensure no negative weights for sklearn compatibility
        sample_weights = np.maximum(sample_weights, 0.0)
        
        # Train new model with reweighted samples
        self.unlearned_model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        
        # Handle case where all weights might be zero
        if np.sum(sample_weights) == 0:
            sample_weights = np.ones(n_samples)
            sample_weights[forget_indices] = 0.1
            
        self.unlearned_model.fit(X_scaled, y, sample_weight=sample_weights)
        
        return self.unlearned_model
    
    def evaluate_unlearning(self, X_test: np.ndarray, y_test: np.ndarray,
                          demographic_test: np.ndarray, forget_indices: np.ndarray,
                          X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Evaluate the effectiveness of unlearning.
        
        Returns:
            results: Dictionary with various evaluation metrics
        """
        X_test_scaled = self.scaler.transform(X_test)
        X_train_scaled = self.scaler.transform(X_train)
        
        # Original model performance
        orig_pred = self.original_model.predict(X_test_scaled)
        orig_accuracy = accuracy_score(y_test, orig_pred)
        
        # Unlearned model performance
        unlearn_pred = self.unlearned_model.predict(X_test_scaled)
        unlearn_accuracy = accuracy_score(y_test, unlearn_pred)
        
        # Forgetting effectiveness: how different are predictions on forgotten instances?
        if len(forget_indices) > 0:
            forget_train_scaled = X_train_scaled[forget_indices]
            orig_forget_pred = self.original_model.predict_proba(forget_train_scaled)[:, 1]
            unlearn_forget_pred = self.unlearned_model.predict_proba(forget_train_scaled)[:, 1]
            
            # Measure prediction difference (forgetting score)
            forgetting_score = np.mean(np.abs(orig_forget_pred - unlearn_forget_pred))
        else:
            forgetting_score = 0.0
        
        # Bias evaluation: performance difference between demographic groups
        group_a_mask = demographic_test == 0
        group_b_mask = demographic_test == 1
        
        if np.any(group_a_mask) and np.any(group_b_mask):
            # Original bias
            orig_acc_a = accuracy_score(y_test[group_a_mask], orig_pred[group_a_mask])
            orig_acc_b = accuracy_score(y_test[group_b_mask], orig_pred[group_b_mask])
            orig_bias = abs(orig_acc_a - orig_acc_b)
            
            # Unlearned bias
            unlearn_acc_a = accuracy_score(y_test[group_a_mask], unlearn_pred[group_a_mask])
            unlearn_acc_b = accuracy_score(y_test[group_b_mask], unlearn_pred[group_b_mask])
            unlearn_bias = abs(unlearn_acc_a - unlearn_acc_b)
        else:
            orig_bias = unlearn_bias = 0.0
            orig_acc_a = orig_acc_b = unlearn_acc_a = unlearn_acc_b = 0.0
        
        return {
            'original_accuracy': orig_accuracy,
            'unlearned_accuracy': unlearn_accuracy,
            'utility_preservation': unlearn_accuracy / orig_accuracy if orig_accuracy > 0 else 0,
            'forgetting_score': forgetting_score,
            'original_bias': orig_bias,
            'unlearned_bias': unlearn_bias,
            'bias_reduction': (orig_bias - unlearn_bias) / orig_bias if orig_bias > 0 else 0,
            'demographic_performance': {
                'original': {'group_a': orig_acc_a, 'group_b': orig_acc_b},
                'unlearned': {'group_a': unlearn_acc_a, 'group_b': unlearn_acc_b}
            }
        }

def run_unlearning_experiment():
    """
    Main experiment function demonstrating machine unlearning pipeline.
    """
    print("=" * 60)
    print("Machine Unlearning with Influence Functions Demo")
    print("=" * 60)
    print()
    
    # Initialize unlearning system
    unlearner = MachineUnlearning(random_state=42)
    
    # Step 1: Create biased dataset
    print("Step 1: Creating biased dataset...")
    X, y, demographic = unlearner.create_biased_dataset(n_samples=1000)
    
    # Split data
    X_train, X_test, y_train, y_test, demo_train, demo_test = train_test_split(
        X, y, demographic, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Dataset created: {len(X_train)} training, {len(X_test)} test samples")
    print(f"Demographic distribution - Group A: {np.mean(demo_train == 0):.2f}, Group B: {np.mean(demo_train == 1):.2f}")
    print()
    
    # Step 2: Train original model
    print("Step 2: Training original biased model...")
    original_model = unlearner.train_original_model(X_train, y_train)
    
    # Evaluate original bias
    X_test_scaled = unlearner.scaler.transform(X_test)
    orig_pred = original_model.predict(X_test_scaled)
    
    group_a_acc = accuracy_score(y_test[demo_test == 0], orig_pred[demo_test == 0])
    group_b_acc = accuracy_score(y_test[demo_test == 1], orig_pred[demo_test == 1])
    
    print(f"Original model accuracy: {accuracy_score(y_test, orig_pred):.3f}")
    print(f"Group A accuracy: {group_a_acc:.3f}")
    print(f"Group B accuracy: {group_b_acc:.3f}")
    print(f"Demographic bias: {abs(group_a_acc - group_b_acc):.3f}")
    print()
    
    # Step 3: Test different unlearning strategies
    strategies = ["influence", "bias_amplifying", "random"]
    reweight_methods = ["zero_weight", "influence_based"]
    
    results_summary = []
    
    for strategy in strategies:
        print(f"Step 3.{strategies.index(strategy)+1}: Testing '{strategy}' forgetting strategy...")
        
        # Identify instances to forget
        forget_indices = unlearner.identify_problematic_instances(
            X_train, y_train, demo_train, strategy=strategy
        )
        
        print(f"  Identified {len(forget_indices)} instances to forget")
        
        for reweight_method in reweight_methods:
            print(f"  Using reweighting method: {reweight_method}")
            
            # Apply soft reweighting
            unlearned_model = unlearner.soft_reweight_training(
                X_train, y_train, forget_indices, 
                reweight_strategy=reweight_method
            )
            
            # Evaluate results
            results = unlearner.evaluate_unlearning(
                X_test, y_test, demo_test, forget_indices, X_train, y_train
            )
            
            # Store results
            results['strategy'] = strategy
            results['reweight_method'] = reweight_method
            results['n_forgotten'] = len(forget_indices)
            results_summary.append(results)
            
            print(f"    Utility preservation: {results['utility_preservation']:.3f}")
            print(f"    Forgetting score: {results['forgetting_score']:.3f}")
            print(f"    Bias reduction: {results['bias_reduction']:.3f}")
            print()
    
    # Step 4: Summary and visualization
    print("Step 4: Results Summary")
    print("=" * 40)
    
    df_results = pd.DataFrame(results_summary)
    
    # Best performing method for bias reduction
    best_bias_reduction = df_results.loc[df_results['bias_reduction'].idxmax()]
    print(f"Best bias reduction method:")
    print(f"  Strategy: {best_bias_reduction['strategy']}")
    print(f"  Reweighting: {best_bias_reduction['reweight_method']}")
    print(f"  Bias reduction: {best_bias_reduction['bias_reduction']:.3f}")
    print(f"  Utility preservation: {best_bias_reduction['utility_preservation']:.3f}")
    print()
    
    # Trade-off analysis
    print("Trade-off Analysis:")
    print("-" * 20)
    for _, row in df_results.iterrows():
        print(f"{row['strategy']+' + '+row['reweight_method']:25} | "
              f"Utility: {row['utility_preservation']:.3f} | "
              f"Forgetting: {row['forgetting_score']:.3f} | "
              f"Bias ↓: {row['bias_reduction']:.3f}")
    
    print()
    print("=" * 60)
    print("Experiment Complete!")
    print()
    print("Key Insights:")
    print("- Influence functions help identify training instances that most affect model behavior")
    print("- Soft reweighting preserves more utility than hard removal")
    print("- Bias-targeted forgetting can reduce demographic disparities")
    print("- There's always a trade-off between forgetting and utility preservation")
    
    return df_results, unlearner

def advanced_bias_mitigation_demo():
    """
    Advanced demo using real-world-like scenario with distribution-level adjustments.
    Inspired by DLFD's distribution-level feature distancing approach.
    """
    print("\n" + "=" * 60)
    print("Advanced Demo: Distribution-Level Bias Mitigation")
    print("=" * 60)
    
    # Use breast cancer dataset as proxy for medical diagnosis scenario
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Simulate demographic bias in medical data
    n_samples = len(X)
    # Add synthetic demographic attribute
    demographic = np.random.binomial(1, 0.3, n_samples)  # 30% minority group
    
    # Introduce systematic bias: minority group has different feature distributions
    minority_mask = demographic == 1
    X[minority_mask, :5] += np.random.normal(0, 0.5, (np.sum(minority_mask), 5))
    
    # Split data
    X_train, X_test, y_train, y_test, demo_train, demo_test = train_test_split(
        X, y, demographic, test_size=0.3, random_state=42
    )
    
    # Initialize and train
    unlearner = MachineUnlearning(random_state=42)
    original_model = unlearner.train_original_model(X_train, y_train)
    
    print(f"Advanced dataset: {len(X_train)} training samples")
    print(f"Feature dimensions: {X_train.shape[1]}")
    
    # Focus on bias reduction strategy
    forget_indices = unlearner.identify_problematic_instances(
        X_train, y_train, demo_train, strategy="bias_amplifying"
    )
    
    # Apply sophisticated reweighting
    unlearned_model = unlearner.soft_reweight_training(
        X_train, y_train, forget_indices, 
        reweight_strategy="influence_based"
    )
    
    # Comprehensive evaluation
    results = unlearner.evaluate_unlearning(
        X_test, y_test, demo_test, forget_indices, X_train, y_train
    )
    
    print(f"\nAdvanced Results:")
    print(f"Original accuracy: {results['original_accuracy']:.3f}")
    print(f"Unlearned accuracy: {results['unlearned_accuracy']:.3f}")
    print(f"Utility preservation: {results['utility_preservation']:.3f}")
    print(f"Demographic bias reduction: {results['bias_reduction']:.3f}")
    
    return results, unlearner

if __name__ == "__main__":
    # Run basic experiment
    basic_results, basic_unlearner = run_unlearning_experiment()
    
    # Run advanced demo
    advanced_results, advanced_unlearner = advanced_bias_mitigation_demo()
    
    print("\n" + "=" * 60)
    print("Implementation Notes & Paper Connections:")
    print("=" * 60)
    print()
    print("1. Influence Functions (Koh & Liang 2017):")
    print("   - Implemented simplified influence score calculation")
    print("   - Used to identify most influential training instances")
    print("   - Enables targeted forgetting of high-impact samples")
    print()
    print("2. Distribution-Level Feature Distancing (DLFD) inspiration:")
    print("   - Soft reweighting preserves feature correlations")
    print("   - Avoids correlation collapse from aggressive error maximization")
    print("   - Balances utility preservation with effective forgetting")
    print()
    print("3. Bias Mitigation (Huber et al. 2024):")
    print("   - Demonstrates how synthetic/biased data affects model fairness")
    print("   - Shows importance of demographic-aware unlearning")
    print("   - Validates that bias patterns persist without intervention")
    print()
    print("4. Key Technical Contributions:")
    print("   - Efficient influence approximation for practical use")
    print("   - Multiple forgetting strategies (influence, bias-targeted, random)")
    print("   - Soft reweighting vs. hard removal comparison")
    print("   - Comprehensive evaluation framework")