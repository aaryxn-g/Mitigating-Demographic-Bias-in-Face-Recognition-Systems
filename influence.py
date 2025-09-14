"""
influence.py - UTKFace Age Bias Analysis using Influence Functions

This file provides a comprehensive implementation for analyzing age bias in facial 
recognition models using influence functions. It's designed to be educational and 
beginner-friendly with extensive explanations.

Author: AI Assistant
Purpose: Educational tool for understanding bias in ML models
Dataset: UTKFace (age prediction from facial images)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import warnings
from tqdm import tqdm
import time
import os
from PIL import Image
from pathlib import Path
from torchvision import transforms

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class InfluenceFunctionAnalyzer:
    """
    A comprehensive class for analyzing age bias using influence functions.
    
    WHAT ARE INFLUENCE FUNCTIONS?
    ============================
    Imagine you're a teacher grading a test. You wonder: "If I removed this one 
    student's homework from the pile I used to create the answer key, how would 
    that change the grade I give to another student's test?"
    
    Influence functions answer exactly this question for machine learning models:
    - They tell us how much each training example influences the model's 
      prediction on a test example
    - Positive influence = training sample helps the model make correct predictions
    - Negative influence = training sample hurts the model's performance
    
    WHY DO WE CARE ABOUT INFLUENCE FUNCTIONS FOR BIAS DETECTION?
    ===========================================================
    1. Identify problematic training data that causes biased predictions
    2. Understand which age groups dominate the model's decision-making
    3. Find training samples that unfairly influence predictions for certain ages
    4. Debug model behavior by tracing predictions back to training data
    """
    
    def __init__(self, model, train_loader, test_loader, criterion):
        """
        Initialize the influence function analyzer.
        
        Args:
            model: The trained neural network
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            criterion: Loss function (e.g., MSELoss for age prediction)
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        print("🔍 Influence Function Analyzer initialized!")
        print(f"📊 Training samples: {len(train_loader.dataset)}")
        print(f"🧪 Test samples: {len(test_loader.dataset)}")
    
    def compute_loss_gradient(self, data_loader, sample_idx=None):
        """
        STEP 1: COMPUTE LOSS GRADIENT
        ============================
        
        WHAT IS A GRADIENT?
        ------------------
        Think of a gradient as the "slope" of our loss function. Just like when 
        you're hiking up a hill, the gradient tells us:
        - Which direction makes the loss go up (bad direction)
        - Which direction makes the loss go down (good direction)
        - How steep the slope is (how much the loss changes)
        
        MATHEMATICAL FORMULA:
        ∇θL(z,θ) = ∂L(z,θ)/∂θ
        
        Where:
        - θ (theta) = model parameters (weights and biases)
        - L = loss function (how wrong our predictions are)
        - z = data sample (image + true age)
        - ∇ (nabla) = gradient operator (fancy symbol for "slope")
        
        Args:
            data_loader: DataLoader containing the data
            sample_idx: If provided, compute gradient for specific sample only
            
        Returns:
            gradients: List of gradient tensors for each parameter
        """
        print("\n🧮 STEP 1: Computing Loss Gradients...")
        print("=" * 50)
        
        self.model.train()  # Set to training mode for gradient computation
        
        # Initialize gradient accumulator
        total_gradients = None
        sample_count = 0
        
        # Progress bar for user feedback
        pbar = tqdm(data_loader, desc="Computing gradients")
        
        for batch_idx, (data, targets) in enumerate(pbar):
            # Move data to device (GPU/CPU)
            data, targets = data.to(self.device), targets.to(self.device)
            
            # If we want gradient for specific sample, skip others
            if sample_idx is not None and batch_idx != sample_idx // data_loader.batch_size:
                continue
                
            # Forward pass: compute predictions
            outputs = self.model(data)
            
            # Compute loss: how wrong are our predictions?
            loss = self.criterion(outputs.squeeze(), targets.float())
            
            # IMPORTANT: Zero out any existing gradients
            self.model.zero_grad()
            
            # Backward pass: compute gradients
            # This is where the magic happens - PyTorch automatically computes
            # how much each parameter contributed to the loss
            loss.backward()
            
            # Collect gradients from all model parameters
            batch_gradients = []
            for param in self.model.parameters():
                if param.grad is not None:
                    # Clone the gradient to avoid it being overwritten
                    batch_gradients.append(param.grad.clone())
                else:
                    # Handle case where parameter doesn't have gradient
                    batch_gradients.append(torch.zeros_like(param))
            
            # Accumulate gradients across batches
            if total_gradients is None:
                total_gradients = batch_gradients
            else:
                for i, grad in enumerate(batch_gradients):
                    total_gradients[i] += grad
                    
            sample_count += data.size(0)
            
            # Update progress bar with current loss
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # If computing for specific sample, we're done
            if sample_idx is not None:
                break
        
        # Average gradients across all samples
        if sample_count > 0:
            for grad in total_gradients:
                grad /= sample_count
        
        print(f"✅ Gradients computed for {sample_count} samples")
        
        # Show some example gradient values for educational purposes
        print("\n📋 Example gradient values:")
        for i, grad in enumerate(total_gradients[:2]):  # Show first 2 layers
            print(f"   Layer {i+1}: shape={grad.shape}, mean={grad.mean().item():.6f}, std={grad.std().item():.6f}")
        
        return total_gradients
    
    def compute_hessian_vector_product(self, gradients):
        """
        STEP 2: COMPUTE HESSIAN-VECTOR PRODUCT
        =====================================
        
        WHAT IS A HESSIAN MATRIX?
        ------------------------
        If gradients tell us the "slope" of our loss function, the Hessian tells 
        us the "curvature" - how the slope itself is changing.
        
        Think of it like this:
        - You're driving on a road (the loss landscape)
        - Gradient = how steep the road is right now
        - Hessian = how quickly the steepness is changing (is it getting steeper?)
        
        MATHEMATICAL FORMULA:
        H = ∇²L(θ) = ∂²L/∂θ²
        
        However, computing the full Hessian matrix is extremely expensive!
        Instead, we compute Hessian-vector products efficiently using automatic 
        differentiation.
        
        WHY DO WE NEED THE HESSIAN?
        ---------------------------
        The influence function formula requires H⁻¹ (inverse Hessian).
        This tells us about the model's "sensitivity" - how much small changes 
        in training data affect the final model parameters.
        
        Args:
            gradients: The gradient vectors we computed in Step 1
            
        Returns:
            hessian_vector_product: H * gradients
        """
        print("\n🏔️ STEP 2: Computing Hessian-Vector Product...")
        print("=" * 50)
        
        self.model.train()
        
        # We'll approximate the Hessian using the training data
        hessian_vector_product = [torch.zeros_like(g) for g in gradients]
        sample_count = 0
        
        print("🔄 Computing second-order derivatives...")
        pbar = tqdm(self.train_loader, desc="Hessian computation")
        
        for data, targets in pbar:
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs.squeeze(), targets.float())
            
            # First backward pass - compute gradients
            first_grads = torch.autograd.grad(
                loss, 
                self.model.parameters(), 
                create_graph=True,  # Keep computation graph for second derivatives
                retain_graph=True
            )
            
            # Compute gradient-vector product
            grad_vector_product = sum(
                torch.sum(g1 * g2) for g1, g2 in zip(first_grads, gradients)
            )
            
            # Second backward pass - compute Hessian-vector product
            hvp = torch.autograd.grad(
                grad_vector_product,
                self.model.parameters(),
                retain_graph=False
            )
            
            # Accumulate Hessian-vector products
            for i, h in enumerate(hvp):
                hessian_vector_product[i] += h
                
            sample_count += data.size(0)
            pbar.set_postfix({'Samples': sample_count})
        
        # Average across samples
        for hvp in hessian_vector_product:
            hvp /= sample_count
            
        print(f"✅ Hessian-vector product computed for {sample_count} samples")
        
        # Show some statistics
        print("\n📊 Hessian-vector product statistics:")
        for i, hvp in enumerate(hessian_vector_product[:2]):
            print(f"   Layer {i+1}: mean={hvp.mean().item():.6f}, std={hvp.std().item():.6f}")
        
        return hessian_vector_product
    
    def solve_inverse_hessian_vector_product(self, hessian_vector_product, gradients, 
                                           num_iterations=100, damping=0.01):
        """
        STEP 3: APPROXIMATE INVERSE HESSIAN-VECTOR PRODUCT
        =================================================
        
        WHAT ARE WE DOING HERE?
        ----------------------
        We need to compute H⁻¹ * gradients, where H⁻¹ is the inverse Hessian.
        
        Think of this like solving the equation: H * x = gradients
        We want to find x, which equals H⁻¹ * gradients.
        
        WHY NOT COMPUTE H⁻¹ DIRECTLY?
        ----------------------------
        The Hessian matrix can be HUGE! For a model with 1 million parameters,
        the Hessian would be 1 million × 1 million = 1 trillion numbers!
        That's impossible to store in memory.
        
        SOLUTION: CONJUGATE GRADIENT METHOD
        ----------------------------------
        This is an iterative algorithm that finds H⁻¹ * gradients without 
        explicitly computing H⁻¹. It's like solving a puzzle piece by piece.
        
        Args:
            hessian_vector_product: H * gradients from Step 2
            gradients: Original gradients from Step 1
            num_iterations: How many steps to take (more = more accurate)
            damping: Regularization to make the computation stable
            
        Returns:
            inv_hessian_vector_product: Approximation of H⁻¹ * gradients
        """
        print("\n🔧 STEP 3: Solving Inverse Hessian-Vector Product...")
        print("=" * 50)
        print(f"🔄 Using conjugate gradient method with {num_iterations} iterations")
        print(f"⚖️ Damping factor: {damping}")
        
        # Initialize solution vector
        x = [torch.zeros_like(g) for g in gradients]
        
        # Initial residual: r = b - A*x = gradients - H*x
        # Since x starts at zero: r = gradients
        r = [g.clone() for g in gradients]
        
        # Initial search direction
        p = [g.clone() for g in gradients]
        
        # Initial residual norm squared
        r_norm_sq = sum(torch.sum(ri * ri) for ri in r)
        
        print("🚀 Starting conjugate gradient iterations...")
        
        for iteration in range(num_iterations):
            # Compute A*p (where A is the damped Hessian)
            self.model.zero_grad()
            
            # We need to compute H*p, but we'll approximate it
            Ap = [torch.zeros_like(pi) for pi in p]
            sample_count = 0
            
            # Compute Hessian-vector product with current search direction
            for data, targets in self.train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs.squeeze(), targets.float())
                
                # First gradients
                first_grads = torch.autograd.grad(
                    loss, self.model.parameters(), 
                    create_graph=True, retain_graph=True
                )
                
                # Gradient-vector product with p
                gvp = sum(torch.sum(g * pi) for g, pi in zip(first_grads, p))
                
                # Second gradients (Hessian-vector product)
                hvp = torch.autograd.grad(gvp, self.model.parameters())
                
                for i, h in enumerate(hvp):
                    Ap[i] += h
                    
                sample_count += data.size(0)
            
            # Average and add damping
            for i, ap in enumerate(Ap):
                ap /= sample_count
                ap += damping * p[i]  # Add damping term
            
            # Compute step size: α = r^T * r / p^T * A * p
            pAp = sum(torch.sum(pi * api) for pi, api in zip(p, Ap))
            alpha = r_norm_sq / (pAp + 1e-10)  # Add small epsilon for stability
            
            # Update solution: x = x + α * p
            for i in range(len(x)):
                x[i] += alpha * p[i]
            
            # Update residual: r = r - α * A * p
            for i in range(len(r)):
                r[i] -= alpha * Ap[i]
            
            # Compute new residual norm
            new_r_norm_sq = sum(torch.sum(ri * ri) for ri in r)
            
            # Convergence check
            if iteration % 10 == 0:
                print(f"   Iteration {iteration:3d}: residual norm = {new_r_norm_sq.item():.6e}")
            
            # Update search direction: p = r + β * p
            beta = new_r_norm_sq / (r_norm_sq + 1e-10)
            for i in range(len(p)):
                p[i] = r[i] + beta * p[i]
            
            r_norm_sq = new_r_norm_sq
            
            # Early stopping if converged
            if r_norm_sq.item() < 1e-10:
                print(f"   ✅ Converged at iteration {iteration}")
                break
        
        print(f"✅ Inverse Hessian-vector product approximated")
        print(f"📈 Final residual norm: {r_norm_sq.item():.6e}")
        
        return x
    
    def compute_influence_scores(self, num_test_samples=None, num_train_samples=None):
        """
        STEP 4: COMPUTE INFLUENCE SCORES
        ===============================
        
        THE MAIN EVENT - INFLUENCE FUNCTION FORMULA!
        ============================================
        
        MATHEMATICAL FORMULA:
        IF(z_train, z_test) = -∇L(z_test, θ̂)^T * H^(-1) * ∇L(z_train, θ̂)
        
        Let's break this down in plain English:
        
        1. ∇L(z_test, θ̂): "How does the loss change if we wiggle the model 
           parameters for this test sample?"
        
        2. ∇L(z_train, θ̂): "How does the loss change if we wiggle the model 
           parameters for this training sample?"
        
        3. H^(-1): "The inverse Hessian - captures how 'flexible' the model is"
        
        4. The dot product: "How aligned are these two gradients?"
        
        5. The negative sign: "We want positive influence to mean 'helpful'"
        
        INTERPRETATION:
        - Positive influence score = training sample helps test prediction
        - Negative influence score = training sample hurts test prediction
        - Larger absolute value = stronger influence
        
        Args:
            num_test_samples: Limit number of test samples (for speed)
            num_train_samples: Limit number of training samples (for speed)
            
        Returns:
            influence_scores: Matrix of influences [test_samples × train_samples]
            test_ages: Ages of test samples
            train_ages: Ages of training samples
        """
        print("\n🎯 STEP 4: Computing Influence Scores...")
        print("=" * 50)
        
        # Limit samples for computational efficiency
        if num_test_samples is None:
            num_test_samples = min(100, len(self.test_loader.dataset))
        if num_train_samples is None:
            num_train_samples = min(500, len(self.train_loader.dataset))
            
        print(f"🧪 Analyzing {num_test_samples} test samples")
        print(f"📚 Against {num_train_samples} training samples")
        print(f"🔢 Total influence computations: {num_test_samples * num_train_samples}")
        
        # Store influence scores and metadata
        influence_scores = np.zeros((num_test_samples, num_train_samples))
        test_ages = []
        train_ages = []
        
        # Get test sample data
        test_data_list = []
        test_age_list = []
        for i, (data, targets) in enumerate(self.test_loader):
            if len(test_data_list) >= num_test_samples:
                break
            for j in range(data.size(0)):
                if len(test_data_list) >= num_test_samples:
                    break
                test_data_list.append(data[j:j+1])
                test_age_list.append(targets[j].item())
        
        test_ages = np.array(test_age_list)
        
        # Get training sample data
        train_data_list = []
        train_age_list = []
        for i, (data, targets) in enumerate(self.train_loader):
            if len(train_data_list) >= num_train_samples:
                break
            for j in range(data.size(0)):
                if len(train_data_list) >= num_train_samples:
                    break
                train_data_list.append(data[j:j+1])
                train_age_list.append(targets[j].item())
        
        train_ages = np.array(train_age_list)
        
        print("📊 Age distribution in samples:")
        print(f"   Test ages: min={test_ages.min():.1f}, max={test_ages.max():.1f}, mean={test_ages.mean():.1f}")
        print(f"   Train ages: min={train_ages.min():.1f}, max={train_ages.max():.1f}, mean={train_ages.mean():.1f}")
        
        # Compute influence scores
        print("\n🔄 Computing influence matrix...")
        start_time = time.time()
        
        for test_idx in tqdm(range(num_test_samples), desc="Test samples"):
            # Get test sample
            test_data = test_data_list[test_idx].to(self.device)
            test_target = torch.tensor([test_ages[test_idx]], dtype=torch.float32).to(self.device)
            
            # Compute test gradient
            self.model.zero_grad()
            test_output = self.model(test_data)
            test_loss = self.criterion(test_output.squeeze(), test_target.float())
            test_loss.backward()
            
            test_gradients = []
            for param in self.model.parameters():
                if param.grad is not None:
                    test_gradients.append(param.grad.clone())
                else:
                    test_gradients.append(torch.zeros_like(param))
            
            # Compute inverse Hessian-vector product for test gradients
            hessian_test_product = self.compute_hessian_vector_product(test_gradients)
            inv_hessian_test_product = self.solve_inverse_hessian_vector_product(
                hessian_test_product, test_gradients, num_iterations=50
            )
            
            # Compute influence with each training sample
            for train_idx in range(num_train_samples):
                # Get training sample
                train_data = train_data_list[train_idx].to(self.device)
                train_target = torch.tensor([train_ages[train_idx]], dtype=torch.float32).to(self.device)
                
                # Compute training gradient
                self.model.zero_grad()
                train_output = self.model(train_data)
                train_loss = self.criterion(train_output.squeeze(), train_target.float())
                train_loss.backward()
                
                train_gradients = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        train_gradients.append(param.grad.clone())
                    else:
                        train_gradients.append(torch.zeros_like(param))
                
                # Compute influence score: -test_grad^T * H^(-1) * train_grad
                influence_score = 0
                for test_g, inv_h_g, train_g in zip(test_gradients, inv_hessian_test_product, train_gradients):
                    influence_score += torch.sum(inv_h_g * train_g).item()
                
                influence_scores[test_idx, train_idx] = -influence_score
        
        elapsed_time = time.time() - start_time
        print(f"⏱️ Computation completed in {elapsed_time:.2f} seconds")
        
        # Show statistics
        print("\n📈 Influence Score Statistics:")
        print(f"   Mean: {influence_scores.mean():.6f}")
        print(f"   Std:  {influence_scores.std():.6f}")
        print(f"   Min:  {influence_scores.min():.6f}")
        print(f"   Max:  {influence_scores.max():.6f}")
        
        return influence_scores, test_ages, train_ages
    
    def analyze_age_bias(self, influence_scores, test_ages, train_ages):
        """
        STEP 5: ANALYZE AGE BIAS PATTERNS
        ================================
        
        Now we analyze the influence scores to understand age bias:
        
        1. GROUP BY AGE: How do different age groups influence each other?
        2. IDENTIFY BIAS: Are some ages over-represented in influential samples?
        3. FIND OUTLIERS: Which specific samples have unusual influence patterns?
        
        WHAT TO LOOK FOR:
        - Age groups that consistently have high positive influence (privileged)
        - Age groups that consistently have negative influence (disadvantaged)  
        - Cross-age influence patterns (do young faces help predict young ages?)
        
        Args:
            influence_scores: Matrix from compute_influence_scores
            test_ages: Ages of test samples
            train_ages: Ages of training samples
            
        Returns:
            bias_analysis: Dictionary containing bias analysis results
        """
        print("\n🔍 STEP 5: Analyzing Age Bias Patterns...")
        print("=" * 50)
        
        # Create age bins for analysis
        age_bins = np.arange(0, 101, 10)  # 0-10, 10-20, ..., 90-100
        age_labels = [f"{age_bins[i]}-{age_bins[i+1]}" for i in range(len(age_bins)-1)]
        
        # Bin the ages
        test_age_bins = np.digitize(test_ages, age_bins) - 1
        train_age_bins = np.digitize(train_ages, age_bins) - 1
        
        print(f"📊 Created {len(age_labels)} age bins: {age_labels}")
        
        # Analyze influence by age groups
        bias_analysis = {}
        
        # 1. Average influence by training age group
        print("\n1️⃣ Average influence by training age group:")
        train_age_influence = {}
        for age_bin, label in enumerate(age_labels):
            mask = (train_age_bins == age_bin)
            if np.sum(mask) > 0:
                avg_influence = influence_scores[:, mask].mean()
                train_age_influence[label] = {
                    'avg_influence': avg_influence,
                    'sample_count': np.sum(mask),
                    'std_influence': influence_scores[:, mask].std()
                }
                print(f"   {label:>6} years: avg={avg_influence:8.6f}, "
                      f"samples={np.sum(mask):3d}, std={influence_scores[:, mask].std():8.6f}")
        
        bias_analysis['train_age_influence'] = train_age_influence
        
        # 2. Average influence received by test age group
        print("\n2️⃣ Average influence received by test age group:")
        test_age_influence = {}
        for age_bin, label in enumerate(age_labels):
            mask = (test_age_bins == age_bin)
            if np.sum(mask) > 0:
                avg_influence = influence_scores[mask, :].mean()
                test_age_influence[label] = {
                    'avg_influence': avg_influence,
                    'sample_count': np.sum(mask),
                    'std_influence': influence_scores[mask, :].std()
                }
                print(f"   {label:>6} years: avg={avg_influence:8.6f}, "
                      f"samples={np.sum(mask):3d}, std={influence_scores[mask, :].std():8.6f}")
        
        bias_analysis['test_age_influence'] = test_age_influence
        
        # 3. Cross-age influence matrix
        print("\n3️⃣ Computing cross-age influence matrix...")
        cross_age_matrix = np.zeros((len(age_labels), len(age_labels)))
        cross_age_counts = np.zeros((len(age_labels), len(age_labels)))
        
        for test_bin in range(len(age_labels)):
            for train_bin in range(len(age_labels)):
                test_mask = (test_age_bins == test_bin)
                train_mask = (train_age_bins == train_bin)
                
                if np.sum(test_mask) > 0 and np.sum(train_mask) > 0:
                    cross_influence = influence_scores[np.ix_(test_mask, train_mask)]
                    cross_age_matrix[test_bin, train_bin] = cross_influence.mean()
                    cross_age_counts[test_bin, train_bin] = cross_influence.size
        
        bias_analysis['cross_age_matrix'] = cross_age_matrix
        bias_analysis['cross_age_counts'] = cross_age_counts
        bias_analysis['age_labels'] = age_labels
        
        # 4. Find most influential samples
        print("\n4️⃣ Finding most influential training samples...")
        top_positive_indices = np.unravel_index(
            np.argsort(influence_scores.flatten())[-10:], influence_scores.shape
        )
        top_negative_indices = np.unravel_index(
            np.argsort(influence_scores.flatten())[:10], influence_scores.shape
        )
        
        print("   📈 Top 5 POSITIVE influences:")
        for i in range(5):
            test_idx = top_positive_indices[0][-1-i]
            train_idx = top_positive_indices[1][-1-i]
            score = influence_scores[test_idx, train_idx]
            print(f"      Test age {test_ages[test_idx]:4.1f} ← Train age {train_ages[train_idx]:4.1f} "
                  f"(influence: {score:8.6f})")
        
        print("   📉 Top 5 NEGATIVE influences:")
        for i in range(5):
            test_idx = top_negative_indices[0][i]
            train_idx = top_negative_indices[1][i]
            score = influence_scores[test_idx, train_idx]
            print(f"      Test age {test_ages[test_idx]:4.1f} ← Train age {train_ages[train_idx]:4.1f} "
                  f"(influence: {score:8.6f})")
        
        bias_analysis['top_influences'] = {
            'positive': (top_positive_indices, influence_scores[top_positive_indices]),
            'negative': (top_negative_indices, influence_scores[top_negative_indices])
        }
        
        # 5. Bias detection summary
        print("\n5️⃣ Bias Detection Summary:")
        print("=" * 30)
        
        # Find age groups with highest/lowest average influence
        train_influences = [(label, data['avg_influence']) 
                          for label, data in train_age_influence.items()]
        train_influences.sort(key=lambda x: x[1], reverse=True)
        
        print("🏆 Most POSITIVELY influential training age groups:")
        for label, influence in train_influences[:3]:
            print(f"   {label:>6} years: {influence:8.6f}")
        
        print("⚠️  Most NEGATIVELY influential training age groups:")
        for label, influence in train_influences[-3:]:
            print(f"   {label:>6} years: {influence:8.6f}")
        
        # Calculate bias metrics
        influence_range = max(data['avg_influence'] for data in train_age_influence.values()) - \
                         min(data['avg_influence'] for data in train_age_influence.values())
        
        bias_analysis['bias_metrics'] = {
            'influence_range': influence_range,
            'most_positive_age': train_influences[0][0],
            'most_negative_age': train_influences[-1][0],
            'bias_severity': 'High' if influence_range > 0.001 else 'Moderate' if influence_range > 0.0001 else 'Low'
        }
        
        print(f"📊 Influence range across age groups: {influence_range:.8f}")
        print(f"🎯 Bias severity assessment: {bias_analysis['bias_metrics']['bias_severity']}")
        
        return bias_analysis
    
    def visualize_bias_analysis(self, influence_scores, test_ages, train_ages, bias_analysis):
        """
        STEP 6: VISUALIZE BIAS PATTERNS
        ===============================
        
        Create comprehensive visualizations to understand the bias patterns:
        1. Bar chart of average influence by age group
        2. Heatmap of cross-age influence patterns
        3. Scatter plot of individual influence scores
        4. Distribution plots showing bias patterns
        
        These visualizations help us communicate findings to stakeholders
        and identify specific patterns that might not be obvious in numbers.
        """
        print("\n🎨 STEP 6: Creating Bias Visualization...")
        print("=" * 50)
        
        # Set up the plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Bar chart of training age group influences
        ax1 = plt.subplot(2, 3, 1)
        age_labels = bias_analysis['age_labels']
        train_influences = [bias_analysis['train_age_influence'].get(label, {'avg_influence': 0})['avg_influence'] 
                          for label in age_labels]
        sample_counts = [bias_analysis['train_age_influence'].get(label, {'sample_count': 0})['sample_count'] 
                        for label in age_labels]
        
        bars = ax1.bar(range(len(age_labels)), train_influences, 
                      color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_xlabel('Training Age Groups')
        ax1.set_ylabel('Average Influence Score')
        ax1.set_title('Average Influence by Training Age Group\n(Higher = More Influential)')
        ax1.set_xticks(range(len(age_labels)))
        ax1.set_xticklabels(age_labels, rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add sample count labels on bars
        for i, (bar, count) in enumerate(zip(bars, sample_counts)):
            if count > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(train_influences)*0.01,
                        f'n={count}', ha='center', va='bottom', fontsize=8)
        
        # 2. Cross-age influence heatmap
        ax2 = plt.subplot(2, 3, 2)
        cross_age_matrix = bias_analysis['cross_age_matrix']
        im = ax2.imshow(cross_age_matrix, cmap='RdBu_r', aspect='auto')
        ax2.set_xlabel('Training Age Groups')
        ax2.set_ylabel('Test Age Groups')
        ax2.set_title('Cross-Age Influence Matrix\n(Red=Positive, Blue=Negative)')
        ax2.set_xticks(range(len(age_labels)))
        ax2.set_yticks(range(len(age_labels)))
        ax2.set_xticklabels(age_labels, rotation=45)
        ax2.set_yticklabels(age_labels)
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, shrink=0.8)
        
        # Add text annotations for significant values
        for i in range(len(age_labels)):
            for j in range(len(age_labels)):
                if not np.isnan(cross_age_matrix[i, j]) and abs(cross_age_matrix[i, j]) > 0.0001:
                    ax2.text(j, i, f'{cross_age_matrix[i, j]:.4f}', 
                            ha='center', va='center', fontsize=8, 
                            color='white' if abs(cross_age_matrix[i, j]) > np.nanmax(np.abs(cross_age_matrix))*0.5 else 'black')
        
        # 3. Scatter plot of age vs influence
        ax3 = plt.subplot(2, 3, 3)
        
        # Create scatter data
        train_ages_repeated = np.repeat(train_ages, len(test_ages))
        test_ages_repeated = np.tile(test_ages, len(train_ages))
        influence_flattened = influence_scores.flatten()
        
        # Sample for visualization (too many points otherwise)
        sample_indices = np.random.choice(len(influence_flattened), 
                                        min(2000, len(influence_flattened)), 
                                        replace=False)
        
        scatter = ax3.scatter(train_ages_repeated[sample_indices], 
                            influence_flattened[sample_indices],
                            c=test_ages_repeated[sample_indices], 
                            cmap='viridis', alpha=0.6, s=20)
        ax3.set_xlabel('Training Sample Age')
        ax3.set_ylabel('Influence Score')
        ax3.set_title('Training Age vs Influence Score\n(Color = Test Sample Age)')
        ax3.grid(alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3, shrink=0.8)
        cbar.set_label('Test Sample Age')
        
        # 4. Distribution of influences by age group
        ax4 = plt.subplot(2, 3, 4)
        
        # Box plot of influence distributions
        age_groups = []
        influence_distributions = []
        
        for age_bin, label in enumerate(age_labels):
            train_mask = np.digitize(train_ages, np.arange(0, 101, 10)) - 1 == age_bin
            if np.sum(train_mask) > 0:
                age_influences = influence_scores[:, train_mask].flatten()
                age_groups.append(label)
                influence_distributions.append(age_influences)
        
        bp = ax4.boxplot(influence_distributions, labels=age_groups, patch_artist=True)
        ax4.set_xlabel('Training Age Groups')
        ax4.set_ylabel('Influence Score Distribution')
        ax4.set_title('Influence Score Distributions by Age Group')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', alpha=0.3)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 5. Top influential samples visualization
        ax5 = plt.subplot(2, 3, 5)
        
        top_positive_indices, positive_scores = bias_analysis['top_influences']['positive']
        top_negative_indices, negative_scores = bias_analysis['top_influences']['negative']
        
        # Show top 10 positive and negative influences
        top_test_ages_pos = test_ages[top_positive_indices[0][-10:]]
        top_train_ages_pos = train_ages[top_positive_indices[1][-10:]]
        top_scores_pos = positive_scores[-10:]
        
        top_test_ages_neg = test_ages[top_negative_indices[0][:10]]
        top_train_ages_neg = train_ages[top_negative_indices[1][:10]]
        top_scores_neg = negative_scores[:10]
        
        # Create arrows showing influence direction
        for i in range(min(5, len(top_scores_pos))):  # Show top 5 to avoid clutter
            ax5.annotate('', xy=(top_test_ages_pos[-1-i], top_scores_pos[-1-i]),
                        xytext=(top_train_ages_pos[-1-i], top_scores_pos[-1-i]),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7, lw=2))
            ax5.plot(top_train_ages_pos[-1-i], top_scores_pos[-1-i], 'ro', markersize=8, alpha=0.7)
            ax5.plot(top_test_ages_pos[-1-i], top_scores_pos[-1-i], 'r^', markersize=8, alpha=0.7)
        
        for i in range(min(5, len(top_scores_neg))):
            ax5.annotate('', xy=(top_test_ages_neg[i], top_scores_neg[i]),
                        xytext=(top_train_ages_neg[i], top_scores_neg[i]),
                        arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7, lw=2))
            ax5.plot(top_train_ages_neg[i], top_scores_neg[i], 'bo', markersize=8, alpha=0.7)
            ax5.plot(top_test_ages_neg[i], top_scores_neg[i], 'b^', markersize=8, alpha=0.7)
        
        ax5.set_xlabel('Age')
        ax5.set_ylabel('Influence Score')
        ax5.set_title('Top Influential Sample Pairs\n(Red=Positive, Blue=Negative, O=Train, ^=Test)')
        ax5.grid(alpha=0.3)
        ax5.legend(['Positive Train', 'Positive Test', 'Negative Train', 'Negative Test'], 
                  loc='upper right')
        
        # 6. Bias summary text
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Create summary text
        bias_metrics = bias_analysis['bias_metrics']
        summary_text = f"""
BIAS ANALYSIS SUMMARY
=====================

🎯 Overall Bias Severity: {bias_metrics['bias_severity']}
📊 Influence Range: {bias_metrics['influence_range']:.8f}

🏆 Most Positively Influential Age Group:
   {bias_metrics['most_positive_age']} years

⚠️  Most Negatively Influential Age Group:
   {bias_metrics['most_negative_age']} years

📈 Key Findings:
• {len([k for k, v in bias_analysis['train_age_influence'].items() 
       if v['avg_influence'] > 0])} age groups have positive average influence
• {len([k for k, v in bias_analysis['train_age_influence'].items() 
       if v['avg_influence'] < 0])} age groups have negative average influence
• Influence scores range from {influence_scores.min():.6f} 
  to {influence_scores.max():.6f}

🔍 What This Means:
Positive influence = training samples that help predictions
Negative influence = training samples that hurt predictions
Large range = significant age bias present

💡 Recommendations:
• Investigate {bias_metrics['most_negative_age']} age group data quality
• Consider data augmentation for underrepresented ages
• Implement fairness constraints in training
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('age_bias_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'age_bias_analysis.png'")
        
        return fig

class SimpleAgePredictor(nn.Module):
    """
    A simple neural network for age prediction from facial images.
    
    ARCHITECTURE EXPLANATION:
    ========================
    We use a simple CNN architecture that's easy to understand:
    
    1. Convolutional layers: Extract features from images (edges, textures, etc.)
    2. Pooling layers: Reduce spatial dimensions while keeping important features
    3. Fully connected layers: Make final age prediction based on extracted features
    
    This is intentionally simple for educational purposes. Real-world models 
    would be much more complex (ResNet, EfficientNet, etc.)
    """
    
    def __init__(self, input_channels=3, num_classes=1):
        """
        Initialize the age prediction network.
        
        Args:
            input_channels: Number of input channels (3 for RGB images)
            num_classes: Number of output classes (1 for age regression)
        """
        super(SimpleAgePredictor, self).__init__()
        
        print("🏗️ Building Simple Age Predictor Network...")
        
        # Convolutional feature extractor
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
        )
        
        # Classifier (age predictor)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # Ensure fixed size regardless of input
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)  # Single output for age
        )
        
        print("✅ Network architecture:")
        print("   📊 Feature extractor: Conv2d -> BatchNorm -> ReLU -> MaxPool (×3)")
        print("   🎯 Classifier: AdaptivePool -> Linear -> ReLU -> Dropout (×2) -> Linear")
        print(f"   📈 Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        """Forward pass through the network."""
        features = self.features(x)
        age = self.classifier(features)
        return age

class UTKFaceDataset(Dataset):
    """
    UTKFace Dataset class for loading real face images and extracting age labels.
    
    WHAT IS UTKFace?
    ================
    UTKFace is a large-scale face dataset with age, gender, and ethnicity annotations.
    Each image filename follows the format: [age]_[gender]_[race]_[date&time].jpg
    
    FILENAME FORMAT:
    ===============
    - age: 0 to 116 years
    - gender: 0 (male) or 1 (female)
    - race: 0 (White), 1 (Black), 2 (Asian), 3 (Indian), 4 (Others)
    - date&time: timestamp when the photo was taken
    
    Example: "25_1_0_20170109150557.jpg" = 25-year-old female, White, taken on 2017/01/09
    """
    
    def __init__(self, dataset_path, image_size=64, max_samples=None, transform=None):
        """
        Initialize UTKFace dataset.
        
        Args:
            dataset_path: Path to UTKFace dataset directory
            image_size: Target size for images (will be resized to image_size x image_size)
            max_samples: Maximum number of samples to load (None = load all)
            transform: Optional torchvision transforms
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        if not self.dataset_path.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {dataset_path}")
        self.image_size = image_size
        self.max_samples = max_samples
        
        # Default transforms for preprocessing
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Load and parse all valid image files
        self.image_paths, self.ages, self.genders, self.races = self._load_dataset()
        
        print(f"📊 UTKFace Dataset loaded:")
        print(f"   📁 Dataset path: {self.dataset_path}")
        print(f"   📸 Total images: {len(self.image_paths)}")
        print(f"   🎂 Age range: {min(self.ages):.1f} - {max(self.ages):.1f} years")
        print(f"   📊 Mean age: {np.mean(self.ages):.1f} years")
        
    def _load_dataset(self):
        """
        Load and parse UTKFace dataset files.
        
        Returns:
            image_paths: List of valid image file paths
            ages: List of corresponding ages
            genders: List of corresponding genders
            races: List of corresponding races
        """
        print(f"\n🔍 Scanning UTKFace dataset at: {self.dataset_path}")
        print("=" * 60)
        
        image_paths = []
        ages = []
        genders = []
        races = []
        invalid_files = []
        
        # Find all .jpg files in the dataset directory
        jpg_files = list(self.dataset_path.glob("*.jpg"))
        
        if len(jpg_files) == 0:
            raise ValueError(f"No .jpg files found in dataset directory: {self.dataset_path}")
        
        print(f"📁 Found {len(jpg_files)} .jpg files")
        
        # Parse each filename to extract labels
        for img_path in tqdm(jpg_files, desc="Parsing filenames"):
            try:
                # Extract filename without extension
                filename = img_path.stem
                
                # Split by underscore: [age]_[gender]_[race]_[datetime]
                parts = filename.split('_')
                
                if len(parts) >= 4:
                    age = int(parts[0])
                    gender = int(parts[1])
                    race = int(parts[2])
                    
                    # Validate extracted values
                    if 0 <= age <= 116 and gender in [0, 1] and race in [0, 1, 2, 3, 4]:
                        image_paths.append(img_path)
                        ages.append(np.float32(age))
                        genders.append(gender)
                        races.append(race)
                        
                        # Stop if we've reached max_samples
                        if self.max_samples and len(image_paths) >= self.max_samples:
                            break
                    else:
                        invalid_files.append(filename)
                else:
                    invalid_files.append(filename)
                    
            except (ValueError, IndexError) as e:
                invalid_files.append(img_path.name)
        
        print(f"✅ Successfully parsed {len(image_paths)} valid files")
        if invalid_files:
            print(f"⚠️  Found {len(invalid_files)} invalid filenames")
            if len(invalid_files) <= 5:
                print(f"   Examples: {invalid_files}")
        
        # Show age distribution
        ages_array = np.array(ages)
        print(f"\n📊 Age Distribution Analysis:")
        age_ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 120)]
        for min_age, max_age in age_ranges:
            count = np.sum((ages_array >= min_age) & (ages_array < max_age))
            percentage = count / len(ages_array) * 100 if len(ages_array) > 0 else 0
            print(f"   {min_age:2d}-{max_age:2d} years: {count:5d} samples ({percentage:5.1f}%)")
        
        return image_paths, ages, genders, races
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            image: Preprocessed image tensor
            age: Age label as float
        """
        try:
            # Load image
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Get age label and ensure float32 type
            age = torch.tensor(self.ages[idx], dtype=torch.float32)
            
            return image, age
            
        except Exception as e:
            print(f"⚠️  Error loading image {self.image_paths[idx]}: {e}")
            # Return a dummy sample in case of error
            dummy_image = torch.zeros(3, self.image_size, self.image_size)
            return dummy_image, torch.tensor(0.0, dtype=torch.float32)

def load_utkface_dataset(dataset_path, image_size=64, max_samples=None, batch_size=32, train_split=0.7, val_split=0.15):
    """
    Load UTKFace dataset and create train/validation/test splits.
    
    DATASET LOADING PROCESS:
    =======================
    1. Parse all UTKFace image filenames to extract age labels
    2. Load and preprocess images (resize, normalize)
    3. Split into train/validation/test sets
    4. Create PyTorch DataLoaders for efficient batch processing
    
    BIAS CONSIDERATIONS:
    ==================
    This function loads the REAL UTKFace data, which contains actual bias patterns:
    - Age distribution may be skewed toward certain age groups
    - Image quality may vary across different demographics
    - Sample sizes may be unbalanced across age ranges
    
    Args:
        dataset_path: Path to UTKFace dataset directory
        image_size: Target image size (images will be resized to image_size x image_size)
        max_samples: Maximum number of samples to load (None = load all)
        batch_size: Batch size for DataLoaders
        train_split: Fraction of data for training (default 0.7 = 70%)
        val_split: Fraction of data for validation (default 0.15 = 15%)
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data  
        test_loader: DataLoader for test data
        dataset_info: Dictionary with dataset statistics
    """
    print(f"\n🎭 Loading Real UTKFace Dataset...")
    print("=" * 50)
    print(f"📁 Dataset path: {dataset_path}")
    print(f"📏 Image size: {image_size}x{image_size}")
    print(f"📊 Max samples: {max_samples if max_samples else 'All'}")
    print(f"🔄 Splits: Train={train_split:.1%}, Val={val_split:.1%}, Test={1-train_split-val_split:.1%}")
    
    # Create dataset
    dataset = UTKFaceDataset(dataset_path, image_size=image_size, max_samples=max_samples)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"\n📊 Dataset Split:")
    print(f"   🏃 Training: {train_size:,} samples ({train_size/total_size:.1%})")
    print(f"   🧪 Validation: {val_size:,} samples ({val_size/total_size:.1%})")
    print(f"   🎯 Test: {test_size:,} samples ({test_size/total_size:.1%})")
    
    # Create random splits
    torch.manual_seed(42)  # For reproducible splits
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Collect dataset statistics
    ages_array = np.array(dataset.ages)
    dataset_info = {
        'total_samples': total_size,
        'train_samples': train_size,
        'val_samples': val_size,
        'test_samples': test_size,
        'age_min': ages_array.min(),
        'age_max': ages_array.max(),
        'age_mean': ages_array.mean(),
        'age_std': ages_array.std(),
        'image_size': image_size,
        'batch_size': batch_size
    }
    
    print(f"\n✅ UTKFace Dataset loaded successfully!")
    print(f"📈 Age statistics: min={dataset_info['age_min']:.1f}, max={dataset_info['age_max']:.1f}, mean={dataset_info['age_mean']:.1f}±{dataset_info['age_std']:.1f}")
    
    return train_loader, val_loader, test_loader, dataset_info

# Educational helper functions for beginners

def explain_influence_functions():
    """
    BEGINNER'S GUIDE TO INFLUENCE FUNCTIONS
    =====================================
    
    Call this function to get a comprehensive explanation of influence functions
    in simple terms with intuitive examples.
    """
    print("🎓 INFLUENCE FUNCTIONS EXPLAINED FOR BEGINNERS")
    print("=" * 50)
    
    print("\n🤔 WHAT ARE INFLUENCE FUNCTIONS?")
    print("-" * 35)
    print("Imagine you're studying for a test using a set of practice problems.")
    print("Later, you wonder: 'Which practice problem helped me the most on question 5?'")
    print("Influence functions answer exactly this question for machine learning!")
    print("\n💡 In ML terms:")
    print("• Training data = practice problems")
    print("• Test prediction = your answer on the real test")  
    print("• Influence score = how much each practice problem helped/hurt")
    
    print("\n📊 WHY DO WE CARE?")
    print("-" * 20)
    print("1. 🔍 DEBUG MODELS: Find training data that causes wrong predictions")
    print("2. 🎯 DETECT BIAS: Identify which groups dominate model decisions") 
    print("3. 🧹 CLEAN DATA: Remove problematic training samples")
    print("4. ⚖️ ENSURE FAIRNESS: Make sure all groups are represented fairly")
    
    print("\n🧮 THE MATH (SIMPLIFIED):")
    print("-" * 30)
    print("Influence(train_sample, test_sample) = ")
    print("  -gradient(test) • inverse_hessian • gradient(train)")
    print("\n🔍 Breaking it down:")
    print("• gradient(test): How sensitive the test loss is to model changes")
    print("• gradient(train): How the training sample wants to change the model")
    print("• inverse_hessian: How 'flexible' the model is to changes")
    print("• The dot product: How aligned these changes are")
    
    print("\n📈 INTERPRETING RESULTS:")
    print("-" * 25)
    print("✅ POSITIVE influence: Training sample HELPS test prediction")
    print("❌ NEGATIVE influence: Training sample HURTS test prediction")
    print("🎯 LARGE absolute value: STRONG influence (good or bad)")
    print("🤷 NEAR ZERO: Training sample doesn't affect test prediction much")
    
    print("\n🚀 PRACTICAL APPLICATIONS:")
    print("-" * 28)
    print("• 👥 Fairness: Ensure all demographic groups have positive influence")
    print("• 🔧 Debugging: Find training samples that cause specific errors")  
    print("• 📊 Data Quality: Identify mislabeled or corrupted training data")
    print("• 🎯 Active Learning: Choose most influential samples for annotation")

def explain_gradients_and_hessians():
    """
    BEGINNER'S GUIDE TO GRADIENTS AND HESSIANS
    ==========================================
    
    Explains the mathematical concepts behind influence functions using
    intuitive analogies and visual descriptions.
    """
    print("🧮 GRADIENTS AND HESSIANS EXPLAINED")
    print("=" * 40)
    
    print("\n🏔️ IMAGINE A MOUNTAINOUS LANDSCAPE:")
    print("-" * 38)
    print("Your model's 'loss' creates a landscape where:")
    print("• 🏔️ PEAKS = high loss (bad predictions)")
    print("• 🏞️ VALLEYS = low loss (good predictions)")
    print("• 🎯 Your goal: Find the lowest valley (best model)")
    
    print("\n📈 WHAT IS A GRADIENT?")
    print("-" * 25)
    print("The GRADIENT is like a compass that points:")
    print("• ⬆️ UPHILL: Direction that increases loss (makes model worse)")
    print("• ⬇️ DOWNHILL: Direction that decreases loss (makes model better)")
    print("• 📏 STEEPNESS: How quickly the loss changes")
    print("\n🧮 Mathematically: ∇L = [∂L/∂w₁, ∂L/∂w₂, ...] for all weights")
    print("💡 Think of it as: 'If I change each weight a tiny bit, how does loss change?'")
    
    print("\n🌊 WHAT IS A HESSIAN?")
    print("-" * 22)
    print("The HESSIAN describes the 'curvature' of the loss landscape:")
    print("• 🥣 BOWL-SHAPED: Positive curvature, stable minimum")
    print("• 🥙 SADDLE-SHAPED: Mixed curvature, unstable point")
    print("• 📐 FLAT: Near-zero curvature, plateau region")
    print("\n🧮 Mathematically: H = [∂²L/∂wᵢ∂wⱼ] (second derivatives)")
    print("💡 Think of it as: 'How does the slope itself change?'")
    
    print("\n🎯 WHY DO WE NEED BOTH?")
    print("-" * 24)
    print("• 🧭 GRADIENT: Tells us which direction to move")
    print("• 🏔️ HESSIAN: Tells us how confident we should be in that direction")
    print("• 🔄 TOGETHER: Enable sophisticated optimization and influence analysis")
    
    print("\n⚡ IN INFLUENCE FUNCTIONS:")
    print("-" * 27)
    print("• Gradient: How each sample wants to change the model")
    print("• Hessian inverse: How the model responds to those changes")
    print("• Combined: Net effect of training sample on test prediction")

def demonstrate_small_example():
    """
    SMALL EXAMPLE DEMONSTRATION
    ==========================
    
    Run influence function analysis on a tiny dataset to show
    all the intermediate steps and results clearly.
    """
    print("🔬 SMALL EXAMPLE DEMONSTRATION")
    print("=" * 35)
    print("Running influence analysis on a tiny dataset to show every step...")
    
    # This function is not needed for UTKFace analysis
    print("Small example function removed - use main() for UTKFace analysis")
    return None, None, None
    
    # Split into even smaller sets
    train_size = 60
    test_size = 20
    
    train_data = TensorDataset(small_images[:train_size], small_ages[:train_size])
    test_data = TensorDataset(small_images[train_size:train_size+test_size], 
                             small_ages[train_size:train_size+test_size])
    
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=10, shuffle=False)
    
    # Train very simple model
    print("🏃 Training tiny model...")
    small_model = SimpleAgePredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(small_model.parameters(), lr=0.01)
    
    # Quick training (just 3 epochs)
    for epoch in range(3):
        small_model.train()
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = small_model(data)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
        print(f"   Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    # Run influence analysis on very small subset
    print("\n🔍 Running influence analysis...")
    analyzer = InfluenceFunctionAnalyzer(small_model, train_loader, test_loader, criterion)
    
    # Analyze just 5 test samples against 10 training samples
    influence_scores, test_ages, train_ages = analyzer.compute_influence_scores(
        num_test_samples=5, num_train_samples=10
    )
    
    # Show detailed results
    print("\n📊 DETAILED RESULTS:")
    print("=" * 25)
    print("Influence Matrix (rows=test samples, cols=training samples):")
    print("Test Age →", end="")
    for i in range(len(test_ages)):
        print(f"{test_ages[i]:7.1f}", end="")
    print()
    
    for j in range(len(train_ages)):
        print(f"Train {train_ages[j]:4.1f} │", end="")
        for i in range(len(test_ages)):
            influence = influence_scores[i, j]
            color = "🟢" if influence > 0 else "🔴" if influence < -0.001 else "⚪"
            print(f"{color}{influence:6.3f}", end="")
        print()
    
    print("\n🎯 INTERPRETATION:")
    print("🟢 = Positive influence (helpful)")
    print("🔴 = Negative influence (harmful)")  
    print("⚪ = Neutral influence")
    
    # Find most influential pair
    max_pos_idx = np.unravel_index(np.argmax(influence_scores), influence_scores.shape)
    max_neg_idx = np.unravel_index(np.argmin(influence_scores), influence_scores.shape)
    
    print(f"\n🏆 Most HELPFUL training sample:")
    print(f"   Train age {train_ages[max_pos_idx[1]]:.1f} helps predict test age {test_ages[max_pos_idx[0]]:.1f}")
    print(f"   Influence score: {influence_scores[max_pos_idx]:.6f}")
    
    print(f"\n⚠️  Most HARMFUL training sample:")
    print(f"   Train age {train_ages[max_neg_idx[1]]:.1f} hurts prediction of test age {test_ages[max_neg_idx[0]]:.1f}")
    print(f"   Influence score: {influence_scores[max_neg_idx]:.6f}")
    
    return influence_scores, test_ages, train_ages


def main():
    """
    MAIN FUNCTION: UTKFace Age Bias Analysis using Influence Functions
    ================================================================
    
    This function runs the complete influence function analysis pipeline
    on your real UTKFace dataset to detect age bias patterns.
    """
    print("\n🎯 UTKFace Age Bias Analysis using Influence Functions")
    print("=" * 60)
    
    # ===== STEP 1: LOAD REAL UTKFACE DATASET =====
    print("\n" + "="*20 + " STEP 1: UTKFace DATA LOADING " + "="*20)
    
    # Ask user for dataset path
    print("🎯 UTKFace Dataset Selection:")
    print("1. 📁 UTKFace/ (main dataset - ~23,700 images)")
    print("2. 📁 crop_part1/ (subset - ~9,800 images)")
    print("3. 📁 utkface_aligned_cropped/UTKFace/ (processed version)")
    print("4. 🔧 Custom path")
    
    choice = input("\nEnter your choice (1/2/3/4): ").strip()
    
    if choice == "1":
        dataset_path = r"C:\Aaryan\College_Stuff\design proj\UTKFace"
        max_samples = 5000  # Limit for computational efficiency
    elif choice == "2":
        dataset_path = r"C:\Aaryan\College_Stuff\design proj\crop_part1"
        max_samples = 2000  # Smaller subset
    elif choice == "3":
        dataset_path = r"C:\Aaryan\College_Stuff\design proj\utkface_aligned_cropped\UTKFace"
        max_samples = 3000
    else:
        dataset_path = input("Enter custom dataset path: ").strip()
        max_samples = int(input("Enter max samples to load (or 0 for all): ") or "2000")
        if max_samples == 0:
            max_samples = None
    
    print(f"\n📁 Selected dataset: {dataset_path}")
    print(f"📊 Max samples: {max_samples if max_samples else 'All'}")
    
    # Load real UTKFace dataset
    train_loader, val_loader, test_loader, dataset_info = load_utkface_dataset(
        dataset_path=dataset_path,
        image_size=64,  # Smaller size for faster computation
        max_samples=max_samples,
        batch_size=32,
        train_split=0.7,
        val_split=0.15
    )
    
    print(f"\n📊 Real UTKFace Dataset Loaded:")
    print(f"   🏃 Training: {dataset_info['train_samples']:,} samples")
    print(f"   🧪 Validation: {dataset_info['val_samples']:,} samples")
    print(f"   🎯 Test: {dataset_info['test_samples']:,} samples")
    print(f"   📈 Age range: {dataset_info['age_min']:.1f} - {dataset_info['age_max']:.1f} years")
    
    # ===== STEP 2: TRAIN MODEL =====
    print("\n" + "="*20 + " STEP 2: MODEL TRAINING " + "="*20)
    
    model, training_history = train_simple_model(train_loader, val_loader, num_epochs=8)
    
    # ===== STEP 3: INFLUENCE FUNCTION ANALYSIS =====
    print("\n" + "="*20 + " STEP 3: INFLUENCE ANALYSIS " + "="*20)
    
    # Initialize analyzer
    analyzer = InfluenceFunctionAnalyzer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=nn.MSELoss()
    )
    
    # Ask user for analysis scope
    print("\n🎯 Choose Analysis Scope:")
    print("1. 🔬 Quick analysis (20 test × 100 train samples, ~5 minutes)")
    print("2. 📊 Medium analysis (50 test × 300 train samples, ~15 minutes)")
    print("3. 🚀 Full analysis (100 test × 500 train samples, ~45 minutes)")
    
    scope_choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if scope_choice == "1":
        num_test, num_train = 20, 100
    elif scope_choice == "2":
        num_test, num_train = 50, 300
    else:
        num_test, num_train = 100, 500
    
    print(f"\n🔄 Running influence analysis with {num_test} test × {num_train} train samples...")
    print(f"⏱️  Estimated time: {(num_test * num_train) / 1000 * 2:.1f} minutes")
    
    # Compute influence scores with real UTKFace data
    influence_scores, test_ages, train_ages = analyzer.compute_influence_scores(
        num_test_samples=num_test,
        num_train_samples=num_train
    )
    
    # ===== STEP 4: BIAS ANALYSIS =====
    print("\n" + "="*20 + " STEP 4: BIAS PATTERN ANALYSIS " + "="*20)
    
    bias_analysis = analyzer.analyze_age_bias(influence_scores, test_ages, train_ages)
    
    # ===== STEP 5: VISUALIZATION =====
    print("\n" + "="*20 + " STEP 5: BIAS VISUALIZATION " + "="*20)
    
    visualization_fig = analyzer.visualize_bias_analysis(
        influence_scores, test_ages, train_ages, bias_analysis
    )
    
    # ===== STEP 6: FINAL RECOMMENDATIONS =====
    print("\n" + "="*20 + " STEP 6: RECOMMENDATIONS " + "="*20)
    print("🎯 FINAL ANALYSIS SUMMARY")
    print("=" * 40)
    
    bias_metrics = bias_analysis['bias_metrics']
    
    print("📊 KEY FINDINGS FROM REAL UTKFACE DATA:")
    print(f"   • Overall bias severity: {bias_metrics['bias_severity']}")
    print(f"   • Most positively influential age: {bias_metrics['most_positive_age']}")
    print(f"   • Most negatively influential age: {bias_metrics['most_negative_age']}")
    print(f"   • Influence range: {bias_metrics['influence_range']:.8f}")
    print(f"   • Dataset used: {dataset_path}")
    print(f"   • Total samples analyzed: {dataset_info['total_samples']:,}")
    
    print("\n💡 ACTIONABLE RECOMMENDATIONS:")
    
    if bias_metrics['bias_severity'] == 'High':
        print("   🚨 HIGH BIAS DETECTED - Immediate action required:")
        print("     1. Audit training data for quality issues")
        print("     2. Implement data augmentation for underrepresented ages")
        print("     3. Use fairness-aware training objectives")
        print("     4. Consider demographic parity constraints")
        
    elif bias_metrics['bias_severity'] == 'Moderate':
        print("   ⚠️  MODERATE BIAS - Consider improvements:")
        print("     1. Review data collection procedures")
        print("     2. Balance training set across age groups")
        print("     3. Monitor model performance by demographic")
        print("     4. Implement bias testing in validation pipeline")
        
    else:
        print("   ✅ LOW BIAS - Good job! Continue monitoring:")
        print("     1. Regular bias audits during model updates")
        print("     2. Maintain diverse training data")
        print("     3. Track performance metrics by age group")
        print("     4. Consider edge cases and rare age groups")
    
    print(f"\n📋 TECHNICAL DETAILS:")
    print(f"   • Total influence computations: {len(test_ages)} × {len(train_ages)} = {len(test_ages) * len(train_ages):,}")
    print(f"   • Real UTKFace images processed: {dataset_info['total_samples']:,}")
    print(f"   • Image resolution: {dataset_info['image_size']}×{dataset_info['image_size']} pixels")
    print(f"   • Age range in dataset: {dataset_info['age_min']:.1f} - {dataset_info['age_max']:.1f} years")
    print(f"   • Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   • Device used: {device}")
    
    print("\n🚀 NEXT STEPS FOR YOUR UTKFACE ANALYSIS:")
    print("=" * 40)
    print("1. 📊 DATA AUDIT:")
    print("   → Examine UTKFace samples with most negative influence")
    print("   → Check for data quality issues in your dataset")
    print("   → Compare results across different UTKFace subsets (UTKFace/ vs crop_part1/)")
    print("   → Verify age distribution matches your application needs")
    
    print("2. 🔧 MODEL IMPROVEMENT:")
    print("   → Remove or fix problematic training samples")
    print("   → Add more data for underrepresented age groups")
    print("   → Consider demographic-aware loss functions")
    
    print("3. 📈 MONITORING:")
    print("   → Implement bias testing in your ML pipeline")
    print("   → Track performance metrics by age group")
    print("   → Regular influence function analysis on new data")
    
    print("4. 🎯 EVALUATION:")
    print("   → Test model on diverse age groups")
    print("   → Measure fairness metrics (demographic parity, etc.)")
    print("   → Compare performance across different populations")
    
    print("\n" + "="*60)
    print("🎉 UTKFACE INFLUENCE ANALYSIS COMPLETE!")
    print("📁 Results saved as 'age_bias_analysis.png'")
    print("📊 Real bias patterns detected in your UTKFace dataset")
    print("🔬 Use these insights to build fairer age prediction models!")
    print("💾 Consider running analysis on different UTKFace subsets for comparison")
    print("=" * 60)
    
    return {
        'model': model,
        'influence_scores': influence_scores,
        'test_ages': test_ages,
        'train_ages': train_ages,
        'bias_analysis': bias_analysis,
        'training_history': training_history,
        'visualization': visualization_fig,
        'dataset_info': dataset_info
    }

def train_simple_model(train_loader, val_loader, num_epochs=10):
    """
    Train a simple age prediction model.
    
    This function demonstrates the complete training process and creates
    a model that we can then analyze for bias using influence functions.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        
    Returns:
        model: Trained neural network
        training_history: Dictionary with training metrics
    """
    print(f"\n🏃 Training Age Prediction Model for {num_epochs} epochs...")
    print("=" * 50)
    
    # Initialize model, loss function, and optimizer
    model = SimpleAgePredictor().to(device)
    criterion = nn.MSELoss()  # Mean Squared Error for age regression
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    
    # Track training history
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': []
    }
    
    print("🚀 Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_samples = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for data, targets in train_pbar:
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs.squeeze(), targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            batch_loss = loss.item()
            batch_mae = torch.mean(torch.abs(outputs.squeeze() - targets)).item()
            
            train_loss += batch_loss * data.size(0)
            train_mae += batch_mae * data.size(0)
            train_samples += data.size(0)
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'MAE': f'{batch_mae:.2f}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_samples = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for data, targets in val_pbar:
                data, targets = data.to(device), targets.to(device)
                
                outputs = model(data)
                loss = criterion(outputs.squeeze(), targets)
                
                batch_mae = torch.mean(torch.abs(outputs.squeeze() - targets)).item()
                
                val_loss += loss.item() * data.size(0)
                val_mae += batch_mae * data.size(0)
                val_samples += data.size(0)
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'MAE': f'{batch_mae:.2f}'
                })
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / train_samples
        epoch_val_loss = val_loss / val_samples
        epoch_train_mae = train_mae / train_samples
        epoch_val_mae = val_mae / val_samples
        
        # Update learning rate
        scheduler.step()
        
        # Store history
        training_history['train_loss'].append(epoch_train_loss)
        training_history['val_loss'].append(epoch_val_loss)
        training_history['train_mae'].append(epoch_train_mae)
        training_history['val_mae'].append(epoch_val_mae)
        
        # Print epoch summary
        print(f"📊 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {epoch_train_loss:.4f}, Train MAE: {epoch_train_mae:.2f} years")
        print(f"   Val Loss:   {epoch_val_loss:.4f}, Val MAE:   {epoch_val_mae:.2f} years")
        print(f"   Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print()
    
    print("✅ Training completed!")
    print(f"📈 Final validation MAE: {training_history['val_mae'][-1]:.2f} years")
    
    return model, training_history

if __name__ == "__main__":
    print("🎯 UTKFace Age Bias Detection using Influence Functions")
    print("=" * 60)
    print("🔬 This analysis will detect real bias patterns in your UTKFace dataset")
    print("📊 Using influence functions to identify problematic training samples")
    print("=" * 60)
    
    # Run the main analysis
    results = main()
    
    print("\n🎉 Analysis complete!")
    print("📁 Check 'age_bias_analysis.png' for detailed visualizations")
    print("💡 Use the recommendations above to improve your model fairness!")