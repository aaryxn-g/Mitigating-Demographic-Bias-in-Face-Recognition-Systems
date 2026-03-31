"""
train_fair_loss.py
==================
Phase 2: Trains ResNet-18 with BOTH sample weights AND fairness-aware loss.

Key additions over Phase 1:
1. FairnessLoss class that penalizes accuracy variance across races
2. Logging of fairness penalty at each epoch
3. Model checkpointing based on best fairness score (lowest group disparity)

Research Basis:
- Kotwal & Marcel (2025): In-processing bias mitigation via fairness constraints
- Variance minimization: Standard approach for equal opportunity enforcement
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset, random_split, Subset
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt


class SimpleFairnessLoss(nn.Module):
    """
    A clean, minimal fairness-aware loss that penalizes accuracy variance across groups.
    
    Key features:
    - Uses only Phase 1 sample weights (no double reweighting)
    - Simple variance minimization of per-group accuracies
    - Fixed lambda_fairness parameter (tune between 0.5-1.0)
    
    Args:
        lambda_fairness: Weight for fairness penalty (0.5-1.0 recommended)
        num_races: Number of demographic groups (5 in UTKFace)
    """
    def __init__(self, lambda_fairness=0.4, num_races=5):
        super().__init__()
        self.lambda_fairness = lambda_fairness
        self.num_races = num_races
    
    def forward(self, logits, targets, race_labels, sample_weights=None):
        """
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            race_labels: Demographic group for each sample [batch_size]
            sample_weights: Sample weights from Phase 1 [batch_size]
            
        Returns:
            total_loss: Combined loss with fairness penalty
            ce_loss: Standard cross-entropy loss
            fairness_penalty: Fairness penalty term (variance of per-group accuracies)
        """
        # Standard CE loss (weighted by Phase 1 weights only)
        if sample_weights is not None:
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            ce_loss = (ce_loss * sample_weights).mean()
        else:
            ce_loss = F.cross_entropy(logits, targets)
        
        # Fairness: per-race accuracy variance
        preds = logits.argmax(dim=1)
        race_accs = []
        
        for race_id in range(self.num_races):
            mask = (race_labels == race_id)
            if mask.sum() > 0:  # Only consider groups present in batch
                acc = (preds[mask] == targets[mask]).float().mean()
                race_accs.append(acc)
        
        # Calculate variance of accuracies across groups
        if len(race_accs) > 0:
            race_accs = torch.stack(race_accs)
            fairness_penalty = race_accs.var()
        else:
            fairness_penalty = torch.tensor(0.0, device=logits.device)
        
        # Total loss with fairness penalty
        total_loss = ce_loss + self.lambda_fairness * fairness_penalty
        
        return total_loss, ce_loss, fairness_penalty


class FairLossTrainer:
    """
    Trainer with fairness-aware loss and comprehensive logging.
    """
    
    RACE_NAMES = ['Caucasian', 'African American', 'Asian', 'Indian', 'Others']
    
    def __init__(self, weights_csv, device='cuda', batch_size=32, 
                 learning_rate=0.0005, lambda_fairness=0.1, num_epochs=18):
        self.device = device
        self.batch_size = batch_size
        self.lr = learning_rate
        self.lambda_fairness = lambda_fairness
        self.num_epochs = num_epochs
        
        # Load sample weights from Phase 1
        self.weights_df = pd.read_csv(weights_csv)
        
        # Initialize model with better initialization for the final layer
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Add dropout for regularization
            nn.Linear(num_ftrs, 5)  # 5 races
        )
        # Initialize the final layer with Kaiming initialization
        nn.init.kaiming_normal_(self.model.fc[1].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.model.fc[1].bias, 0.0)
        self.model = self.model.to(self.device)
        
        # Fairness-aware loss
        self.criterion = SimpleFairnessLoss(
            lambda_fairness=lambda_fairness,
            num_races=5
        )
        
        # Use AdamW optimizer with different learning rates for different layers
        param_groups = [
            {'params': [p for n, p in self.model.named_parameters() if 'fc' not in n], 'lr': self.lr * 0.1},
            {'params': self.model.fc.parameters(), 'lr': self.lr}
        ]
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=1e-4,  # L2 regularization
            eps=1e-8,  # Better numerical stability
            amsgrad=True  # Use AMSGrad variant for better convergence
        )
        
        # Learning rate scheduler with warmup and cosine decay
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=num_epochs, 
            eta_min=1e-6  # Minimum learning rate
        )
        
        # Gradient clipping value
        self.max_grad_norm = 1.0
        
        print(f"✓ Initialized Fair Loss Trainer")
        print(f"  - Device: {device}")
        print(f"  - Lambda Fairness: {lambda_fairness}")
        print(f"  - Sample Weighting: {len(self.weights_df)} samples")
    
    def train_epoch(self, train_loader, epoch):
        """
        Train one epoch with fairness-aware loss.
        """
        self.model.train()
        
        epoch_ce_loss = 0
        epoch_fairness_penalty = 0
        epoch_total_loss = 0
        
        all_preds, all_labels, all_races = [], [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            # Unpack the batch (images, labels, races, weights)
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)  # Race labels
            races = batch[1].to(self.device)    # Using the same as race labels since we're predicting race
            weights = batch[2].to(self.device)  # Sample weights
            
            # Forward pass with fairness loss
            self.optimizer.zero_grad()
            
            # Forward pass with mixup augmentation
            if np.random.rand() < 0.5:  # 50% chance to apply mixup
                alpha = 0.2  # Mixup hyperparameter
                lam = np.random.beta(alpha, alpha)
                batch_size = images.size(0)
                index = torch.randperm(batch_size).to(images.device)
                
                mixed_images = lam * images + (1 - lam) * images[index, :]
                outputs = self.model(mixed_images)
                
                # Compute mixup loss
                loss1, ce_loss1, _ = self.criterion(outputs, labels, races, weights)
                loss2, ce_loss2, _ = self.criterion(outputs, labels[index], races[index], weights[index] if weights is not None else None)
                total_loss = lam * loss1 + (1 - lam) * loss2
                ce_loss = lam * ce_loss1 + (1 - lam) * ce_loss2
                
                # Compute fairness penalty separately on original data
                with torch.no_grad():
                    outputs_orig = self.model(images)
                    _, _, fairness_penalty = self.criterion(outputs_orig, labels, races, weights)
            else:
                # Standard forward pass
                outputs = self.model(images)
                total_loss, ce_loss, fairness_penalty = self.criterion(outputs, labels, races, weights)
            
            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.max_grad_norm,
                norm_type=2.0
            )
            
            # Check for exploding/vanishing gradients
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: Invalid loss value: {total_loss}")
                # Skip this batch if loss is invalid
                continue
                
            self.optimizer.step()
            
            # Accumulate losses
            epoch_total_loss += total_loss.item()
            epoch_ce_loss += ce_loss.item()
            epoch_fairness_penalty += fairness_penalty.item()
            
            # Collect predictions for metrics
            with torch.no_grad():
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_races.extend(races.cpu().numpy())
                
                pbar.set_postfix({
                    'total_loss': f'{total_loss.item():.4f}',
                    'ce_loss': f'{ce_loss.item():.4f}',
                    'fairness': f'{fairness_penalty.item():.4f}'
                })
        
        # Average losses
        n_batches = len(train_loader)
        return {
            'total_loss': epoch_total_loss / n_batches,
            'ce_loss': epoch_ce_loss / n_batches,
            'fairness_penalty': epoch_fairness_penalty / n_batches,
            'preds': all_preds,
            'labels': all_labels,
            'races': all_races
        }
    
    def compute_fairness_metrics(self, preds, labels, races):
        """
        Compute comprehensive fairness metrics.
        """
        preds = np.array(preds)
        labels = np.array(labels)
        races = np.array(races)
        
        race_accuracies = {}
        for race_id, race_name in enumerate(self.RACE_NAMES):
            mask = races == race_id
            if mask.sum() > 0:
                acc = (preds[mask] == labels[mask]).mean()
                race_accuracies[race_name] = float(acc)
        
        # Variance (primary fairness metric)
        acc_values = list(race_accuracies.values())
        accuracy_variance = np.var(acc_values)
        accuracy_std = np.std(acc_values)
        
        # Max-min gap
        max_min_gap = max(acc_values) - min(acc_values)
        
        return {
            'per_race_accuracy': race_accuracies,
            'accuracy_variance': float(accuracy_variance),
            'accuracy_std': float(accuracy_std),
            'max_min_gap': float(max_min_gap)
        }
    
    def train(self, train_loader, val_loader, num_epochs=25, output_dir='phase2_outputs'):
        """
        Full training loop with fairness tracking.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        history = {
            'total_loss': [],
            'ce_loss': [],
            'fairness_penalty': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        best_fairness_score = float('inf')  # Lower is better
        
        for epoch in range(num_epochs):
            # Train
            train_results = self.train_epoch(train_loader, epoch)
            
            # Compute fairness metrics
            train_metrics = self.compute_fairness_metrics(
                train_results['preds'],
                train_results['labels'],
                train_results['races']
            )
            
            # Log
            history['total_loss'].append(train_results['total_loss'])
            history['ce_loss'].append(train_results['ce_loss'])
            history['fairness_penalty'].append(train_results['fairness_penalty'])
            history['train_metrics'].append(train_metrics)
            
            print(f"\n[Epoch {epoch+1}/{num_epochs}]")
            print(f"  Total Loss: {train_results['total_loss']:.4f}")
            print(f"  CE Loss: {train_results['ce_loss']:.4f}")
            print(f"  Fairness Penalty: {train_results['fairness_penalty']:.4f}")
            print(f"  Per-Race Accuracy: {train_metrics['per_race_accuracy']}")
            print(f"  Accuracy Std Dev: {train_metrics['accuracy_std']:.4f}")
            print(f"  Max-Min Gap: {train_metrics['max_min_gap']:.4f}")
            
            # Save best model based on fairness
            current_fairness_score = train_metrics['accuracy_variance']
            if current_fairness_score < best_fairness_score:
                best_fairness_score = current_fairness_score
                torch.save(self.model.state_dict(), 
                          f'{output_dir}/best_fair_model.pth')
                print(f"  ✓ New best fairness model saved (variance: {best_fairness_score:.6f})")
            
            self.scheduler.step()
        
        # Save final model and history
        torch.save(self.model.state_dict(), f'{output_dir}/final_fair_model.pth')
        with open(f'{output_dir}/training_history_phase2.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot training curves
        self.plot_training_curves(history, output_dir)
        
        print(f"\n✓ PHASE 2 TRAINING COMPLETE")
        print(f"  Best Fairness Score: {best_fairness_score:.6f}")
        print(f"  Models saved to: {output_dir}/")
        
        return self.model, history
    
    def plot_training_curves(self, history, output_dir):
        """
        Visualize training progress and fairness improvements.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = range(1, len(history['total_loss']) + 1)
        
        # Plot 1: Loss components
        axes[0, 0].plot(epochs, history['ce_loss'], label='CE Loss', color='blue')
        axes[0, 0].plot(epochs, history['fairness_penalty'], label='Fairness Penalty', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Components Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Plot 2: Fairness metrics
        variances = [m['accuracy_variance'] for m in history['train_metrics']]
        stds = [m['accuracy_std'] for m in history['train_metrics']]
        gaps = [m['max_min_gap'] for m in history['train_metrics']]
        
        axes[0, 1].plot(epochs, variances, label='Accuracy Variance', color='red')
        axes[0, 1].plot(epochs, stds, label='Accuracy Std Dev', color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Disparity')
        axes[0, 1].set_title('Fairness Metrics (Lower = Better)')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: Max-Min Gap
        axes[1, 0].plot(epochs, gaps, color='purple', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy Gap')
        axes[1, 0].set_title('Max-Min Accuracy Gap Across Races')
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 4: Per-race accuracy trends
        for race_name in self.RACE_NAMES:
            race_accs = [m['per_race_accuracy'].get(race_name, 0) 
                        for m in history['train_metrics']]
            axes[1, 1].plot(epochs, race_accs, label=race_name, marker='o', markersize=3)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Per-Race Accuracy Convergence')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/phase2_training_curves.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Training curves saved to {output_dir}/phase2_training_curves.png")
        plt.close()


class UTKFaceDataset(Dataset):
    """Custom dataset for UTKFace with race labels."""
    
    def __init__(self, root_dir, weights_df, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            weights_df (DataFrame): DataFrame containing file_path and weight information.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Create a mapping from filename to row in weights_df
        self.weights_df = weights_df
        self.weights_df['filename'] = self.weights_df['file_path'].apply(lambda x: os.path.basename(x))
        
        # Get all image files in the directory
        self.image_files = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        # Filter to only include files that exist in our weights_df
        self.valid_files = [
            f for f in self.image_files
            if f in self.weights_df['filename'].values
        ]
        
        print(f"Found {len(self.valid_files)} images with weights out of {len(self.image_files)} total images")
        
        # Create race to index mapping
        self.race_to_idx = {race: i for i, race in enumerate(['Caucasian', 'African American', 'Asian', 'Indian', 'Others'])}
        
        # Cache metadata so we can fetch weights without reloading images
        self.samples = []
        for img_name in self.valid_files:
            row = self.weights_df[self.weights_df['filename'] == img_name].iloc[0]
            self.samples.append({
                'path': os.path.join(self.root_dir, img_name),
                'filename': img_name,
                'race_label': self.race_to_idx.get(row['race'], 4),
                'weight': float(row['weight'])
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, sample['race_label'], sample['race_label'], sample['weight']
    
    def get_sample_weight(self, idx):
        return self.samples[idx]['weight']


def get_data_loaders(weights_csv, data_dir, batch_size=32, val_split=0.2):
    """Create training and validation data loaders."""
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load weights dataframe
    weights_df = pd.read_csv(weights_csv)
    
    # Create dataset
    full_dataset = UTKFaceDataset(
        root_dir=data_dir,
        weights_df=weights_df,
        transform=transform
    )
    
    # Split into train and validation sets
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Changed from 4 to 0 to avoid multiprocessing issues
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,  # Changed from 4 to 0 to avoid multiprocessing issues
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train with Fairness Loss')
    parser.add_argument('--weights', type=str, default='fairness_sample_weights.csv',
                        help='Path to the weights CSV file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=18,
                        help='Number of epochs to train (default: 18)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--lambda_fairness', type=float, default=0.1,
                        help='Weight for fairness loss term')
    parser.add_argument('--data_dir', type=str, default='../crop_part1',
                        help='Directory containing the dataset (default: ../crop_part1)')
    args = parser.parse_args()
    
    # Set device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    # Initialize trainer
    trainer = FairLossTrainer(
        weights_csv=args.weights,
        device=DEVICE,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lambda_fairness=args.lambda_fairness,
        num_epochs=args.epochs
    )
    
    # Load data
    print("Loading dataset...")
    try:
        train_loader, val_loader = get_data_loaders(
            weights_csv=args.weights,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            val_split=0.2
        )
        print(f"Loaded {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples")
        
        # Train the model
        print(f"Starting training for {args.epochs} epochs...")
        model, history = trainer.train(train_loader, val_loader, num_epochs=args.epochs)
        
        # Save the model
        output_dir = 'checkpoints'
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'fair_loss_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
