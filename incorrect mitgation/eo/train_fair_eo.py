"""
Train ResNet-18 with Equalized Odds Loss for Fair Race Classification

This script implements Phase 2 bias mitigation using proper fairness constraints.

Key improvements over previous attempts:
1. Uses Equalized Odds Loss (minimizes FPR variance) instead of accuracy variance
2. Conservative weight capping to avoid class collapse
3. Dynamic lambda scheduling for gradual fairness emphasis
4. Comprehensive logging and checkpointing
5. Per-race accuracy tracking to detect collapse early
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
import argparse


class EqualizedOddsLoss(nn.Module):
    """
    Equalized Odds Loss - minimizes FPR variance across groups.
    """
    def __init__(self, num_races=5, lambda_fairness=0.1, use_fpr=True, device='cuda'):
        super().__init__()
        self.num_races = num_races
        self.lambda_fairness = lambda_fairness
        self.use_fpr = use_fpr
        self.device = device
    
    def forward(self, logits, targets, race_labels, sample_weights=None):
        """Compute CE loss + FPR variance penalty."""
        preds = logits.argmax(dim=1)
        
        # CE loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        if sample_weights is not None:
            ce_loss = (ce_loss * sample_weights).mean()
        else:
            ce_loss = ce_loss.mean()
        
        # Compute FPR per race
        race_fprs = []
        for race_id in range(self.num_races):
            mask = (race_labels == race_id)
            if mask.sum() > 0:
                preds_race = preds[mask]
                targets_race = targets[mask]
                
                fp = ((preds_race == race_id) & (targets_race != race_id)).float()
                tn = ((preds_race != race_id) & (targets_race != race_id)).float()
                
                fpr = fp.sum() / (fp.sum() + tn.sum() + 1e-8)
                race_fprs.append(fpr)
        
        if len(race_fprs) > 1:
            race_fprs = torch.stack(race_fprs)
            fairness_penalty = race_fprs.var()
        else:
            fairness_penalty = torch.tensor(0.0, device=self.device)
        
        total_loss = ce_loss + self.lambda_fairness * fairness_penalty
        
        metrics = {
            'fpr_variance': fairness_penalty.item(),
            'mean_fpr': race_fprs.mean().item() if len(race_fprs) > 0 else 0.0
        }
        
        return total_loss, ce_loss, fairness_penalty, metrics


class EqualizedOddsWithTPRFNRLoss(nn.Module):
    """
    Full Equalized Odds: minimizes both FPR and TPR variance.
    """
    def __init__(self, num_races=5, lambda_fairness=0.1, lambda_tpr=0.5, lambda_fpr=0.5, device='cuda'):
        super().__init__()
        self.num_races = num_races
        self.lambda_fairness = lambda_fairness
        self.lambda_tpr = lambda_tpr
        self.lambda_fpr = lambda_fpr
        self.device = device
    
    def forward(self, logits, targets, race_labels, sample_weights=None):
        """Compute CE loss + FPR/TPR variance penalty."""
        preds = logits.argmax(dim=1)
        
        # CE loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        if sample_weights is not None:
            ce_loss = (ce_loss * sample_weights).mean()
        else:
            ce_loss = ce_loss.mean()
        
        # Compute FPR and TPR per race
        race_fprs = []
        race_tprs = []
        
        for race_id in range(self.num_races):
            mask = (race_labels == race_id)
            if mask.sum() > 0:
                preds_race = preds[mask]
                targets_race = targets[mask]
                
                fp = ((preds_race == race_id) & (targets_race != race_id)).float()
                tn = ((preds_race != race_id) & (targets_race != race_id)).float()
                tp = ((preds_race == race_id) & (targets_race == race_id)).float()
                fn = ((preds_race != race_id) & (targets_race == race_id)).float()
                
                fpr = fp.sum() / (fp.sum() + tn.sum() + 1e-8)
                tpr = tp.sum() / (tp.sum() + fn.sum() + 1e-8)
                
                race_fprs.append(fpr)
                race_tprs.append(tpr)
        
        if len(race_fprs) > 1:
            race_fprs = torch.stack(race_fprs)
            race_tprs = torch.stack(race_tprs)
            
            fpr_penalty = race_fprs.var()
            tpr_penalty = race_tprs.var()
            
            fairness_penalty = self.lambda_fpr * fpr_penalty + self.lambda_tpr * tpr_penalty
        else:
            fairness_penalty = torch.tensor(0.0, device=self.device)
            fpr_penalty = torch.tensor(0.0, device=self.device)
            tpr_penalty = torch.tensor(0.0, device=self.device)
        
        total_loss = ce_loss + self.lambda_fairness * fairness_penalty
        
        metrics = {
            'fpr_variance': fpr_penalty.item() if len(race_fprs) > 1 else 0.0,
            'tpr_variance': tpr_penalty.item() if len(race_tprs) > 1 else 0.0
        }
        
        return total_loss, ce_loss, fairness_penalty, metrics


class UTKFaceDataset(Dataset):
    """Dataset for UTKFace with race labels and sample weights."""
    
    def __init__(self, root_dir, weights_df, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Create filename to row mapping
        self.weights_df = weights_df.copy()
        self.weights_df['filename'] = self.weights_df['file_path'].apply(
            lambda x: os.path.basename(x)
        )
        
        # Get all image files
        self.image_files = [
            f for f in os.listdir(root_dir) 
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        
        # Filter to only images in weights_df
        valid_files = self.weights_df['filename'].values
        self.image_files = [f for f in self.image_files if f in valid_files]
        
        # Race mapping
        self.race_to_idx = {
            'Caucasian': 0,
            'African American': 1,
            'Asian': 2,
            'Indian': 3,
            'Others': 4
        }
        
        print(f"✓ Loaded {len(self.image_files)} images with weights")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        
        # Get metadata from weights_df
        row = self.weights_df[self.weights_df['filename'] == filename].iloc[0]
        race_label = self.race_to_idx.get(row['race'], 4)
        weight = float(row['weight'])
        
        # Load image
        img_path = os.path.join(self.root_dir, filename)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, race_label, race_label, weight


class FairTrainerWithEqualizedOdds:
    """Trainer with Equalized Odds fairness loss."""
    
    RACE_NAMES = ['Caucasian', 'African American', 'Asian', 'Indian', 'Others']
    
    def __init__(self, weights_csv, device='cuda', batch_size=32, 
                 learning_rate=0.0005, lambda_fairness=0.1, 
                 num_epochs=25, use_tpr_fpr=False):
        """
        Args:
            weights_csv: Path to weights CSV from Phase 1
            device: 'cuda' or 'cpu'
            batch_size: Batch size for training
            learning_rate: Learning rate
            lambda_fairness: Initial fairness weight (will be scheduled)
            num_epochs: Number of epochs
            use_tpr_fpr: If True, use TPR+FPR loss. If False, use FPR only.
        """
        
        self.device = device
        self.batch_size = batch_size
        self.lr = learning_rate
        self.lambda_fairness = lambda_fairness
        self.num_epochs = num_epochs
        self.use_tpr_fpr = use_tpr_fpr
        
        # Load weights
        self.weights_df = pd.read_csv(weights_csv)
        
        # Initialize model
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 5)
        )
        
        # Proper weight initialization
        nn.init.kaiming_normal_(self.model.fc[1].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.model.fc[1].bias, 0.0)
        
        self.model = self.model.to(self.device)
        self.model.eval()  # Start in eval mode
        
        # Fairness loss
        if use_tpr_fpr:
            self.criterion = EqualizedOddsWithTPRFNRLoss(
                num_races=5,
                lambda_fairness=lambda_fairness,
                lambda_tpr=0.5,
                lambda_fpr=0.5,
                device=device
            )
        else:
            self.criterion = EqualizedOddsLoss(
                num_races=5,
                lambda_fairness=lambda_fairness,
                use_fpr=True,
                device=device
            )
        
        # Optimizer with differential learning rates
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() if 'fc' not in n],
                'lr': self.lr * 0.1
            },
            {
                'params': self.model.fc.parameters(),
                'lr': self.lr
            }
        ]
        
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=1e-4,
            eps=1e-8,
            amsgrad=True
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
        
        print(f"✓ Initialized Fair Trainer (Equalized Odds)")
        print(f"  - Device: {device}")
        print(f"  - Lambda Fairness: {lambda_fairness}")
        print(f"  - Loss Type: {'TPR+FPR' if use_tpr_fpr else 'FPR only'}")
    
    def train_epoch(self, train_loader, epoch):
        """Train one epoch."""
        
        self.model.train()
        
        epoch_total_loss = 0
        epoch_ce_loss = 0
        epoch_fairness_penalty = 0
        
        all_preds, all_labels, all_races = [], [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            races = batch[1].to(self.device)  # Race is same as label
            weights = batch[2].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute loss
            total_loss, ce_loss, fairness_penalty, metrics = self.criterion(
                outputs, labels, races, weights
            )
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
                norm_type=2.0
            )
            
            # Check for NaN
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: Invalid loss at batch. Skipping.")
                continue
            
            self.optimizer.step()
            
            # Accumulate
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
                'total': f'{total_loss.item():.4f}',
                'ce': f'{ce_loss.item():.4f}',
                'fair': f'{fairness_penalty.item():.4f}'
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
        """Compute per-race accuracy and fairness metrics."""
        
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
        
        # Fairness metrics
        acc_values = list(race_accuracies.values())
        accuracy_variance = np.var(acc_values)
        accuracy_std = np.std(acc_values)
        max_min_gap = max(acc_values) - min(acc_values)
        
        return {
            'per_race_accuracy': race_accuracies,
            'accuracy_variance': float(accuracy_variance),
            'accuracy_std': float(accuracy_std),
            'max_min_gap': float(max_min_gap)
        }
    
    def dynamic_lambda_schedule(self, epoch, total_epochs):
        """
        Schedule lambda fairness weight over time.
        
        Start low (focus on accuracy), gradually increase (focus on fairness).
        """
        # Cosine annealing: start at 0.5x, increase to 1.5x
        progress = epoch / total_epochs
        schedule = 0.5 + 0.5 * (1 - np.cos(np.pi * progress))
        
        new_lambda = self.lambda_fairness * schedule
        self.criterion.lambda_fairness = new_lambda
        
        return new_lambda
    
    def train(self, train_loader, val_loader, output_dir='phase2_fair_outputs'):
        """Full training loop."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        history = {
            'total_loss': [],
            'ce_loss': [],
            'fairness_penalty': [],
            'train_metrics': [],
            'val_metrics': [],
            'lambda_schedule': []
        }
        
        best_fairness_score = float('inf')
        
        for epoch in range(self.num_epochs):
            
            # Dynamic lambda scheduling
            current_lambda = self.dynamic_lambda_schedule(epoch, self.num_epochs)
            history['lambda_schedule'].append(current_lambda)
            
            # Train
            train_results = self.train_epoch(train_loader, epoch)
            
            # Compute fairness metrics
            train_metrics = self.compute_fairness_metrics(
                train_results['preds'],
                train_results['labels'],
                train_results['races']
            )
            
            # Log
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"  Total Loss: {train_results['total_loss']:.4f}")
            print(f"  CE Loss: {train_results['ce_loss']:.4f}")
            print(f"  Fairness Penalty: {train_results['fairness_penalty']:.4f}")
            print(f"  Lambda: {current_lambda:.4f}")
            print(f"  Per-Race Accuracy: {train_metrics['per_race_accuracy']}")
            print(f"  Accuracy Std: {train_metrics['accuracy_std']:.4f}")
            print(f"  Max-Min Gap: {train_metrics['max_min_gap']:.4f}")
            
            # Check for collapse
            min_acc = min(train_metrics['per_race_accuracy'].values())
            if min_acc < 0.01:
                print(f"  ⚠️  WARNING: Minimum accuracy {min_acc:.4f} is near zero!")
            
            # Store history
            history['total_loss'].append(train_results['total_loss'])
            history['ce_loss'].append(train_results['ce_loss'])
            history['fairness_penalty'].append(train_results['fairness_penalty'])
            history['train_metrics'].append(train_metrics)
            
            # Checkpointing
            fairness_score = train_metrics['accuracy_std']
            if fairness_score < best_fairness_score:
                best_fairness_score = fairness_score
                torch.save(
                    self.model.state_dict(),
                    os.path.join(output_dir, 'best_fair_model_eo.pth')
                )
                print(f"  ✓ New best fairness model saved (std={fairness_score:.4f})")
            
            # Step scheduler
            self.scheduler.step()
        
        # Save final model and history
        torch.save(
            self.model.state_dict(),
            os.path.join(output_dir, 'final_fair_model_eo.pth')
        )
        
        with open(os.path.join(output_dir, 'training_history_eo.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n✓ Training complete!")
        print(f"  Models saved to {output_dir}")
        print(f"  Best fairness score: {best_fairness_score:.4f}")
        
        # Plot
        self.plot_training_curves(history, output_dir)
        
        return self.model, history
    
    def plot_training_curves(self, history, output_dir):
        """Visualize training progress."""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        epochs = range(1, len(history['total_loss']) + 1)
        
        # Plot 1: Loss components
        axes[0, 0].plot(epochs, history['ce_loss'], label='CE Loss', color='blue')
        axes[0, 0].plot(epochs, history['fairness_penalty'], label='Fairness Penalty', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Components Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Plot 2: Lambda schedule
        axes[0, 1].plot(epochs, history['lambda_schedule'], color='purple', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Lambda')
        axes[0, 1].set_title('Fairness Weight Evolution')
        axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: Fairness metrics
        stds = [m['accuracy_std'] for m in history['train_metrics']]
        gaps = [m['max_min_gap'] for m in history['train_metrics']]
        
        axes[0, 2].plot(epochs, stds, label='Std Dev', color='orange')
        axes[0, 2].plot(epochs, gaps, label='Max-Min Gap', color='purple')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Disparity')
        axes[0, 2].set_title('Fairness Metrics (Lower = Better)')
        axes[0, 2].legend()
        axes[0, 2].grid(alpha=0.3)
        
        # Plot 4: Per-race accuracy
        for race_name in self.RACE_NAMES:
            accs = [m['per_race_accuracy'][race_name] for m in history['train_metrics']]
            axes[1, 0].plot(epochs, accs, label=race_name, marker='o', markersize=3)
        
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Per-Race Accuracy Convergence')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 5: Mean accuracy
        means = [np.mean(list(m['per_race_accuracy'].values())) 
                 for m in history['train_metrics']]
        axes[1, 1].plot(epochs, means, color='green', linewidth=2, marker='s')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Mean Accuracy')
        axes[1, 1].set_title('Overall Accuracy')
        axes[1, 1].grid(alpha=0.3)
        
        # Plot 6: Utility-Fairness Trade-off
        tradeoffs = [
            np.mean(list(m['per_race_accuracy'].values())) - 0.5 * m['accuracy_std']
            for m in history['train_metrics']
        ]
        axes[1, 2].plot(epochs, tradeoffs, color='cyan', linewidth=2, marker='^')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Trade-off Score')
        axes[1, 2].set_title('Utility-Fairness Trade-off (Higher = Better)')
        axes[1, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves_eo.png'), dpi=150)
        print(f"✓ Training curves saved to {output_dir}/training_curves_eo.png")
        plt.close()


def get_data_loaders(weights_csv, data_dir, batch_size=32, val_split=0.2):
    """Create train and validation dataloaders."""
    
    weights_df = pd.read_csv(weights_csv)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    full_dataset = UTKFaceDataset(data_dir, weights_df, transform)
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
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
    
    print(f"✓ Data Loaders Created")
    print(f"  - Train: {len(train_dataset)} samples")
    print(f"  - Val: {len(val_dataset)} samples")
    
    return train_loader, val_loader


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Train with Equalized Odds Loss for Fair Face Recognition'
    )
    parser.add_argument('--weights', type=str, default='fairness_sample_weights_ultraconservative.csv',
                       help='Path to weights CSV from Phase 1')
    parser.add_argument('--data_dir', type=str, default='/path/to/crop_part1',
                       help='Directory with training images')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lambda_fairness', type=float, default=0.1,
                       help='Initial fairness weight (will be scheduled)')
    parser.add_argument('--use_tpr_fpr', action='store_true',
                       help='Use TPR+FPR loss instead of FPR only')
    parser.add_argument('--output_dir', type=str, default='phase2_fair_outputs_eo')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("PHASE 2: FAIR TRAINING WITH EQUALIZED ODDS LOSS")
    print(f"{'='*80}\n")
    
    # Create trainer
    trainer = FairTrainerWithEqualizedOdds(
        weights_csv=args.weights,
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lambda_fairness=args.lambda_fairness,
        num_epochs=args.epochs,
        use_tpr_fpr=args.use_tpr_fpr
    )
    
    # Load data
    train_loader, val_loader = get_data_loaders(
        args.weights,
        args.data_dir,
        batch_size=args.batch_size,
        val_split=0.2
    )
    
    # Train
    model, history = trainer.train(
        train_loader,
        val_loader,
        output_dir=args.output_dir
    )
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}\n")
