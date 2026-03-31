"""
train_fair_loss_fixed.py
=========================
Fixed implementation of fairness-aware training with:
1. Improved fairness loss with dynamic weighting
2. Better sample weight balancing
3. Comprehensive validation metrics
4. Early stopping based on fairness-utility trade-off
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt


class ImprovedFairnessLoss(nn.Module):
    """
    Improved fairness loss with dynamic lambda adjustment.
    
    Key improvements:
    - Uses standard deviation instead of variance (more interpretable)
    - Dynamic lambda that increases if fairness isn't improving
    - Min-max gap penalty in addition to variance
    - Weighted combination of multiple fairness metrics
    """
    def __init__(self, initial_lambda=0.5, num_races=5):
        super().__init__()
        self.lambda_fairness = initial_lambda
        self.num_races = num_races
        self.epoch_count = 0
        self.prev_fairness = None
        
    def forward(self, logits, targets, race_labels, sample_weights=None):
        """
        Compute combined loss with fairness penalty.
        
        Returns:
            total_loss: Combined loss
            ce_loss: Cross-entropy component
            fairness_penalty: Fairness component
            metrics: Dict with detailed fairness metrics
        """
        # Standard CE loss with sample weights
        if sample_weights is not None:
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            ce_loss = (ce_loss * sample_weights).mean()
        else:
            ce_loss = F.cross_entropy(logits, targets)
        
        # Compute per-race accuracies
        preds = logits.argmax(dim=1)
        race_accs = []
        race_counts = []
        
        for race_id in range(self.num_races):
            mask = (race_labels == race_id)
            count = mask.sum().item()
            if count > 0:
                acc = (preds[mask] == targets[mask]).float().mean()
                race_accs.append(acc)
                race_counts.append(count)
        
        if len(race_accs) < 2:
            # Can't compute fairness with less than 2 groups
            return ce_loss, ce_loss, torch.tensor(0.0, device=logits.device), {}
        
        race_accs = torch.stack(race_accs)
        
        # Multiple fairness penalties
        std_penalty = race_accs.std()  # Standard deviation of accuracies
        gap_penalty = race_accs.max() - race_accs.min()  # Max-min gap
        
        # Combined fairness penalty (weighted average)
        fairness_penalty = 0.7 * std_penalty + 0.3 * gap_penalty
        
        # Total loss
        total_loss = ce_loss + self.lambda_fairness * fairness_penalty
        
        metrics = {
            'std': std_penalty.item(),
            'gap': gap_penalty.item(),
            'mean_acc': race_accs.mean().item(),
            'min_acc': race_accs.min().item(),
            'max_acc': race_accs.max().item()
        }
        
        return total_loss, ce_loss, fairness_penalty, metrics
    
    def update_lambda(self, current_fairness, epoch):
        """
        Dynamically adjust lambda based on fairness improvement.
        Increase lambda if fairness isn't improving.
        """
        if self.prev_fairness is not None:
            improvement = self.prev_fairness - current_fairness
            
            if improvement < 0.001:  # Not improving enough
                self.lambda_fairness = min(2.0, self.lambda_fairness * 1.1)
                print(f"  Increasing lambda to {self.lambda_fairness:.4f} (slow fairness progress)")
        
        self.prev_fairness = current_fairness
        self.epoch_count = epoch


class FairLossTrainer:
    """Improved trainer with better validation and early stopping."""
    
    RACE_NAMES = ['Caucasian', 'African American', 'Asian', 'Indian', 'Others']
    
    def __init__(self, weights_csv, device='cuda', batch_size=32, 
                 learning_rate=0.0005, initial_lambda=0.5, num_epochs=25):
        self.device = device
        self.batch_size = batch_size
        self.lr = learning_rate
        self.num_epochs = num_epochs
        
        # Load sample weights
        self.weights_df = pd.read_csv(weights_csv)
        
        # Normalize weights to prevent extreme values
        # Cap weights at 3x mean to avoid class collapse
        weights = self.weights_df['weight'].values
        mean_weight = weights.mean()
        max_allowed = mean_weight * 3.0
        weights = np.clip(weights, mean_weight * 0.3, max_allowed)
        self.weights_df['weight'] = weights / weights.mean()  # Renormalize
        
        print(f"Adjusted sample weights: min={weights.min():.3f}, "
              f"max={weights.max():.3f}, mean={weights.mean():.3f}")
        
        # Initialize model
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        
        # Simple architecture that matches what we'll evaluate
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),  # Lower dropout
            nn.Linear(num_ftrs, 5)
        )
        
        # Better initialization
        nn.init.xavier_normal_(self.model.fc[1].weight)
        nn.init.constant_(self.model.fc[1].bias, 0.0)
        
        self.model = self.model.to(self.device)
        
        # Improved fairness loss
        self.criterion = ImprovedFairnessLoss(
            initial_lambda=initial_lambda,
            num_races=5
        )
        
        # Optimizer with layer-wise learning rates
        param_groups = [
            {'params': [p for n, p in self.model.named_parameters() if 'fc' not in n], 
             'lr': self.lr * 0.1},  # Lower LR for pretrained layers
            {'params': self.model.fc.parameters(), 'lr': self.lr}
        ]
        
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=5e-4,
            eps=1e-8
        )
        
        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=num_epochs,
            eta_min=1e-6
        )
        
        print(f"✓ Initialized Improved Fair Loss Trainer")
        print(f"  - Device: {device}")
        print(f"  - Initial Lambda: {initial_lambda}")
        print(f"  - Sample Weighting: {len(self.weights_df)} samples")
    
    def train_epoch(self, train_loader, epoch):
        """Train one epoch."""
        self.model.train()
        
        epoch_ce_loss = 0
        epoch_fairness_penalty = 0
        epoch_total_loss = 0
        
        all_preds, all_labels, all_races = [], [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            races = batch[1].to(self.device)  
            weights = batch[2].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute loss with fairness penalty
            total_loss, ce_loss, fairness_penalty, metrics = self.criterion(
                outputs, labels, races, weights
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            epoch_total_loss += total_loss.item()
            epoch_ce_loss += ce_loss.item()
            epoch_fairness_penalty += fairness_penalty.item()
            
            # Collect predictions
            with torch.no_grad():
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_races.extend(races.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'ce': f'{ce_loss.item():.4f}',
                'fair': f'{fairness_penalty.item():.4f}'
            })
        
        n_batches = len(train_loader)
        avg_fairness = epoch_fairness_penalty / n_batches
        
        # Update lambda based on progress
        self.criterion.update_lambda(avg_fairness, epoch)
        
        return {
            'total_loss': epoch_total_loss / n_batches,
            'ce_loss': epoch_ce_loss / n_batches,
            'fairness_penalty': avg_fairness,
            'preds': all_preds,
            'labels': all_labels,
            'races': all_races
        }
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        
        all_preds, all_labels, all_races = [], [], []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                races = batch[1].to(self.device)
                
                outputs = self.model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_races.extend(races.cpu().numpy())
        
        return {
            'preds': all_preds,
            'labels': all_labels,
            'races': all_races
        }
    
    def compute_fairness_metrics(self, preds, labels, races):
        """Compute comprehensive fairness metrics."""
        preds = np.array(preds)
        labels = np.array(labels)
        races = np.array(races)
        
        race_accuracies = {}
        for race_id, race_name in enumerate(self.RACE_NAMES):
            mask = races == race_id
            if mask.sum() > 0:
                acc = (preds[mask] == labels[mask]).mean()
                race_accuracies[race_name] = float(acc)
            else:
                race_accuracies[race_name] = 0.0
        
        acc_values = [v for v in race_accuracies.values() if v > 0]
        
        if len(acc_values) > 0:
            accuracy_variance = np.var(acc_values)
            accuracy_std = np.std(acc_values)
            max_min_gap = max(acc_values) - min(acc_values)
            mean_accuracy = np.mean(acc_values)
        else:
            accuracy_variance = 0
            accuracy_std = 0
            max_min_gap = 0
            mean_accuracy = 0
        
        # Fairness-utility trade-off score (higher is better)
        # Balances mean accuracy with fairness (low std)
        tradeoff_score = mean_accuracy - 0.5 * accuracy_std
        
        return {
            'per_race_accuracy': race_accuracies,
            'accuracy_variance': float(accuracy_variance),
            'accuracy_std': float(accuracy_std),
            'max_min_gap': float(max_min_gap),
            'mean_accuracy': float(mean_accuracy),
            'tradeoff_score': float(tradeoff_score)
        }
    
    def train(self, train_loader, val_loader, output_dir='phase2_outputs'):
        """Full training loop with validation and early stopping."""
        os.makedirs(output_dir, exist_ok=True)
        
        history = {
            'total_loss': [],
            'ce_loss': [],
            'fairness_penalty': [],
            'lambda_values': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        best_tradeoff_score = -float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Train
            train_results = self.train_epoch(train_loader, epoch)
            train_metrics = self.compute_fairness_metrics(
                train_results['preds'],
                train_results['labels'],
                train_results['races']
            )
            
            # Validate
            val_results = self.validate(val_loader)
            val_metrics = self.compute_fairness_metrics(
                val_results['preds'],
                val_results['labels'],
                val_results['races']
            )
            
            # Log
            history['total_loss'].append(train_results['total_loss'])
            history['ce_loss'].append(train_results['ce_loss'])
            history['fairness_penalty'].append(train_results['fairness_penalty'])
            history['lambda_values'].append(self.criterion.lambda_fairness)
            history['train_metrics'].append(train_metrics)
            history['val_metrics'].append(val_metrics)
            
            print(f"\n[Epoch {epoch+1}/{self.num_epochs}]")
            print(f"  Training:")
            print(f"    Total Loss: {train_results['total_loss']:.4f}")
            print(f"    CE Loss: {train_results['ce_loss']:.4f}")
            print(f"    Fairness Penalty: {train_results['fairness_penalty']:.4f}")
            print(f"    Lambda: {self.criterion.lambda_fairness:.4f}")
            print(f"    Mean Accuracy: {train_metrics['mean_accuracy']:.4f}")
            print(f"    Std Dev: {train_metrics['accuracy_std']:.4f}")
            print(f"    Max-Min Gap: {train_metrics['max_min_gap']:.4f}")
            
            print(f"  Validation:")
            print(f"    Mean Accuracy: {val_metrics['mean_accuracy']:.4f}")
            print(f"    Std Dev: {val_metrics['accuracy_std']:.4f}")
            print(f"    Max-Min Gap: {val_metrics['max_min_gap']:.4f}")
            print(f"    Trade-off Score: {val_metrics['tradeoff_score']:.4f}")
            
            # Save best model based on fairness-utility trade-off
            if val_metrics['tradeoff_score'] > best_tradeoff_score:
                best_tradeoff_score = val_metrics['tradeoff_score']
                torch.save(self.model.state_dict(), 
                          f'{output_dir}/best_fair_model.pth')
                print(f"  ✓ New best model (trade-off: {best_tradeoff_score:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping triggered (no improvement for {patience} epochs)")
                break
            
            self.scheduler.step()
        
        # Save final model and history
        torch.save(self.model.state_dict(), f'{output_dir}/final_fair_model.pth')
        with open(f'{output_dir}/training_history_phase2.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        self.plot_training_curves(history, output_dir)
        
        print(f"\n✓ PHASE 2 TRAINING COMPLETE")
        print(f"  Best Trade-off Score: {best_tradeoff_score:.4f}")
        print(f"  Models saved to: {output_dir}/")
        
        return self.model, history
    
    def plot_training_curves(self, history, output_dir):
        """Visualize training progress."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        epochs = range(1, len(history['total_loss']) + 1)
        
        # Loss components
        axes[0, 0].plot(epochs, history['ce_loss'], label='CE Loss')
        axes[0, 0].plot(epochs, history['fairness_penalty'], label='Fairness Penalty')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Components')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Lambda evolution
        axes[0, 1].plot(epochs, history['lambda_values'], color='purple')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Lambda')
        axes[0, 1].set_title('Fairness Weight (Lambda) Evolution')
        axes[0, 1].grid(alpha=0.3)
        
        # Fairness metrics
        train_stds = [m['accuracy_std'] for m in history['train_metrics']]
        val_stds = [m['accuracy_std'] for m in history['val_metrics']]
        axes[0, 2].plot(epochs, train_stds, label='Train')
        axes[0, 2].plot(epochs, val_stds, label='Validation')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy Std Dev')
        axes[0, 2].set_title('Fairness (Lower = Better)')
        axes[0, 2].legend()
        axes[0, 2].grid(alpha=0.3)
        
        # Mean accuracy
        train_means = [m['mean_accuracy'] for m in history['train_metrics']]
        val_means = [m['mean_accuracy'] for m in history['val_metrics']]
        axes[1, 0].plot(epochs, train_means, label='Train')
        axes[1, 0].plot(epochs, val_means, label='Validation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Mean Accuracy')
        axes[1, 0].set_title('Mean Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Max-min gap
        train_gaps = [m['max_min_gap'] for m in history['train_metrics']]
        val_gaps = [m['max_min_gap'] for m in history['val_metrics']]
        axes[1, 1].plot(epochs, train_gaps, label='Train')
        axes[1, 1].plot(epochs, val_gaps, label='Validation')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Max-Min Gap')
        axes[1, 1].set_title('Max-Min Accuracy Gap')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        # Trade-off score
        train_tradeoffs = [m['tradeoff_score'] for m in history['train_metrics']]
        val_tradeoffs = [m['tradeoff_score'] for m in history['val_metrics']]
        axes[1, 2].plot(epochs, train_tradeoffs, label='Train')
        axes[1, 2].plot(epochs, val_tradeoffs, label='Validation')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Trade-off Score')
        axes[1, 2].set_title('Fairness-Utility Trade-off (Higher = Better)')
        axes[1, 2].legend()
        axes[1, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/training_curves.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Training curves saved")
        plt.close()


class UTKFaceDataset(Dataset):
    """Custom dataset for UTKFace with balanced weights."""
    
    def __init__(self, root_dir, weights_df, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.weights_df = weights_df
        self.weights_df['filename'] = self.weights_df['file_path'].apply(
            lambda x: os.path.basename(x)
        )
        
        self.image_files = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        self.valid_files = [
            f for f in self.image_files
            if f in self.weights_df['filename'].values
        ]
        
        print(f"Found {len(self.valid_files)} images with weights")
        
        self.race_to_idx = {
            'Caucasian': 0,
            'African American': 1,
            'Asian': 2,
            'Indian': 3,
            'Others': 4
        }
        
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
        img = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Return: image, label, race_id, weight
        return img, sample['race_label'], sample['race_label'], sample['weight']


def get_data_loaders(weights_csv, data_dir, batch_size=32, val_split=0.2):
    """Create training and validation data loaders."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    weights_df = pd.read_csv(weights_csv)
    
    full_dataset = UTKFaceDataset(
        root_dir=data_dir,
        weights_df=weights_df,
        transform=transform
    )
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='fairness_sample_weights.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lambda_fairness', type=float, default=0.5)
    parser.add_argument('--data_dir', type=str, default='../crop_part1')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    trainer = FairLossTrainer(
        weights_csv=args.weights,
        device=device,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        initial_lambda=args.lambda_fairness,
        num_epochs=args.epochs
    )
    
    print("Loading dataset...")
    train_loader, val_loader = get_data_loaders(
        weights_csv=args.weights,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=0.2
    )
    
    print(f"\nStarting training for {args.epochs} epochs...")
    model, history = trainer.train(train_loader, val_loader)
    
    print("\n✓ Training complete!")
