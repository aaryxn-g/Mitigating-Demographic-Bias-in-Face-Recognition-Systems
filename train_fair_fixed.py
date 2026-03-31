"""
train_fair_fixed.py
===================
FIXED fairness training that prevents class collapse.

Key improvements:
1. NO sample weights in loss (all samples treated equally)
2. Balanced sampling via WeightedRandomSampler (see minorities more often)
3. Minimum accuracy constraint (prevents any race from collapsing)
4. Gentle fairness penalty (lambda=0.05, not aggressive)
5. Equalized Odds loss (FPR/TPR variance)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse


class ConstrainedEqualizedOddsLoss(nn.Module):
    """
    Equalized Odds with minimum accuracy constraint.
    
    Key features:
    - Minimizes FPR/TPR variance across groups
    - Adds penalty if any race drops below min_accuracy threshold
    - NO sample weights used in loss computation
    """
    
    def __init__(self, num_races=5, lambda_fairness=0.05, min_accuracy=0.5, device='cuda'):
        super().__init__()
        self.num_races = num_races
        self.lambda_fairness = lambda_fairness
        self.min_accuracy = min_accuracy
        self.device = device
    
    def forward(self, logits, targets, race_labels):
        """
        Compute loss WITHOUT sample weights.
        
        Args:
            logits: Model predictions
            targets: True labels
            race_labels: Race for each sample
            
        Returns:
            total_loss, ce_loss, fairness_penalty, metrics
        """
        preds = logits.argmax(dim=1)
        
        # CRITICAL: NO sample weights - all samples treated equally
        ce_loss = F.cross_entropy(logits, targets)
        
        # Compute per-race metrics
        race_fprs = []
        race_tprs = []
        race_accs = []
        
        for race_id in range(self.num_races):
            mask = (race_labels == race_id)
            if mask.sum() > 0:
                preds_race = preds[mask]
                targets_race = targets[mask]
                
                # Accuracy
                acc = (preds_race == targets_race).float().mean()
                race_accs.append(acc)
                
                # FPR and TPR
                fp = ((preds_race == race_id) & (targets_race != race_id)).float()
                tn = ((preds_race != race_id) & (targets_race != race_id)).float()
                tp = ((preds_race == race_id) & (targets_race == race_id)).float()
                fn = ((preds_race != race_id) & (targets_race == race_id)).float()
                
                fpr = fp.sum() / (fp.sum() + tn.sum() + 1e-8)
                tpr = tp.sum() / (tp.sum() + fn.sum() + 1e-8)
                
                race_fprs.append(fpr)
                race_tprs.append(tpr)
        
        # Equalized Odds penalty (FPR + TPR variance)
        if len(race_fprs) > 1:
            race_fprs = torch.stack(race_fprs)
            race_tprs = torch.stack(race_tprs)
            race_accs_tensor = torch.stack(race_accs)
            
            fpr_penalty = race_fprs.var()
            tpr_penalty = race_tprs.var()
            fairness_penalty = 0.5 * fpr_penalty + 0.5 * tpr_penalty
            
            # CRITICAL: Minimum accuracy constraint
            # If any race drops below min_accuracy, add large penalty
            min_acc = race_accs_tensor.min()
            min_acc_penalty = torch.relu(self.min_accuracy - min_acc)
            
        else:
            fairness_penalty = torch.tensor(0.0, device=self.device)
            min_acc_penalty = torch.tensor(0.0, device=self.device)
            min_acc = torch.tensor(1.0, device=self.device)
        
        # Total loss: CE + gentle fairness + strong min accuracy constraint
        total_loss = ce_loss + self.lambda_fairness * fairness_penalty + 10.0 * min_acc_penalty
        
        metrics = {
            'fpr_variance': fpr_penalty.item() if len(race_fprs) > 1 else 0.0,
            'tpr_variance': tpr_penalty.item() if len(race_tprs) > 1 else 0.0,
            'min_accuracy': min_acc.item(),
            'min_acc_penalty': min_acc_penalty.item()
        }
        
        return total_loss, ce_loss, fairness_penalty, metrics


class UTKFaceBalancedDataset(Dataset):
    """Dataset WITHOUT sample weights - just returns images and labels."""
    
    def __init__(self, root_dir, labels_csv, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load labels
        self.labels_df = pd.read_csv(labels_csv)
        
        self.race_to_idx = {
            'Caucasian': 0,
            'African American': 1,
            'Asian': 2,
            'Indian': 3,
            'Others': 4
        }
        
        # Build samples list
        self.samples = []
        for idx, row in self.labels_df.iterrows():
            filename = row['filename']
            race = row['race']
            race_label = self.race_to_idx.get(race, 4)
            
            img_path = os.path.join(self.root_dir, filename)
            if os.path.exists(img_path):
                self.samples.append({
                    'path': img_path,
                    'filename': filename,
                    'race_label': race_label,
                    'race_name': race
                })
        
        print(f"✓ Loaded {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Return: image, label, race_id (NO weights!)
        return img, sample['race_label'], sample['race_label']
    
    def get_race_labels(self):
        """Return list of race labels for sampling."""
        return [s['race_label'] for s in self.samples]


class FairTrainerFixed:
    """Fixed trainer with balanced sampling and constrained loss."""
    
    RACE_NAMES = ['Caucasian', 'African American', 'Asian', 'Indian', 'Others']
    
    def __init__(self, device='cuda', batch_size=32, learning_rate=0.0005,
                 lambda_fairness=0.05, min_accuracy=0.5, num_epochs=25):
        self.device = device
        self.batch_size = batch_size
        self.lr = learning_rate
        self.num_epochs = num_epochs
        
        # Initialize model
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 5)
        )
        
        nn.init.xavier_normal_(self.model.fc[1].weight)
        nn.init.constant_(self.model.fc[1].bias, 0.0)
        
        self.model = self.model.to(self.device)
        
        # Constrained Equalized Odds loss
        self.criterion = ConstrainedEqualizedOddsLoss(
            num_races=5,
            lambda_fairness=lambda_fairness,
            min_accuracy=min_accuracy,
            device=device
        )
        
        # Optimizer
        param_groups = [
            {'params': [p for n, p in self.model.named_parameters() if 'fc' not in n], 
             'lr': self.lr * 0.1},
            {'params': self.model.fc.parameters(), 'lr': self.lr}
        ]
        
        self.optimizer = optim.AdamW(param_groups, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-6
        )
        
        print(f"✓ Initialized Fixed Fair Trainer")
        print(f"  - Device: {device}")
        print(f"  - Lambda Fairness: {lambda_fairness} (gentle)")
        print(f"  - Min Accuracy Constraint: {min_accuracy}")
        print(f"  - NO sample weights in loss")
        print(f"  - Using balanced sampling")
    
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
            races = batch[2].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute loss (NO sample weights!)
            total_loss, ce_loss, fairness_penalty, metrics = self.criterion(
                outputs, labels, races
            )
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
                'loss': f'{total_loss.item():.4f}',
                'ce': f'{ce_loss.item():.4f}',
                'fair': f'{fairness_penalty.item():.4f}',
                'min_acc': f'{metrics["min_accuracy"]:.3f}'
            })
        
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
        
        acc_values = list(race_accuracies.values())
        
        return {
            'per_race_accuracy': race_accuracies,
            'accuracy_variance': float(np.var(acc_values)),
            'accuracy_std': float(np.std(acc_values)),
            'max_min_gap': float(max(acc_values) - min(acc_values)),
            'mean_accuracy': float(np.mean(acc_values)),
            'min_accuracy': float(min(acc_values))
        }
    
    def train(self, train_loader, output_dir='phase2_fixed_outputs'):
        """Full training loop."""
        os.makedirs(output_dir, exist_ok=True)
        
        history = {
            'total_loss': [],
            'ce_loss': [],
            'fairness_penalty': [],
            'train_metrics': []
        }
        
        best_min_accuracy = 0.0
        
        for epoch in range(self.num_epochs):
            # Train
            train_results = self.train_epoch(train_loader, epoch)
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
            
            print(f"\n[Epoch {epoch+1}/{self.num_epochs}]")
            print(f"  Total Loss: {train_results['total_loss']:.4f}")
            print(f"  CE Loss: {train_results['ce_loss']:.4f}")
            print(f"  Fairness Penalty: {train_results['fairness_penalty']:.4f}")
            print(f"  Per-Race Accuracy: {train_metrics['per_race_accuracy']}")
            print(f"  Mean Accuracy: {train_metrics['mean_accuracy']:.4f}")
            print(f"  Min Accuracy: {train_metrics['min_accuracy']:.4f}")
            print(f"  Std Dev: {train_metrics['accuracy_std']:.4f}")
            print(f"  Max-Min Gap: {train_metrics['max_min_gap']:.4f}")
            
            # Save best model based on minimum accuracy
            if train_metrics['min_accuracy'] > best_min_accuracy:
                best_min_accuracy = train_metrics['min_accuracy']
                torch.save(self.model.state_dict(), 
                          f'{output_dir}/best_fair_model_fixed.pth')
                print(f"  ✓ New best model (min_acc: {best_min_accuracy:.4f})")
            
            self.scheduler.step()
        
        # Save final model and history
        torch.save(self.model.state_dict(), f'{output_dir}/final_fair_model_fixed.pth')
        with open(f'{output_dir}/training_history_fixed.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n✓ FIXED FAIRNESS TRAINING COMPLETE")
        print(f"  Best Min Accuracy: {best_min_accuracy:.4f}")
        print(f"  Models saved to: {output_dir}/")
        
        return self.model, history


def get_balanced_data_loader(data_dir, labels_csv, batch_size=32):
    """
    Create data loader with balanced sampling.
    
    Key: Uses WeightedRandomSampler to oversample minorities.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = UTKFaceBalancedDataset(data_dir, labels_csv, transform)
    
    # Compute race counts
    race_labels = dataset.get_race_labels()
    race_counts = {}
    for race_label in race_labels:
        race_counts[race_label] = race_counts.get(race_label, 0) + 1
    
    print(f"\nRace distribution:")
    for race_id, count in sorted(race_counts.items()):
        race_name = dataset.race_to_idx
        race_name = [k for k, v in dataset.race_to_idx.items() if v == race_id][0]
        print(f"  {race_name}: {count} samples")
    
    # Compute sampling weights (inverse frequency)
    sample_weights = [1.0 / race_counts[race_label] for race_label in race_labels]
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use balanced sampler
        num_workers=0,
        pin_memory=True
    )
    
    print(f"✓ Created balanced data loader with {len(dataset)} samples")
    print(f"  Batch size: {batch_size}")
    print(f"  Using WeightedRandomSampler (minorities oversampled)")
    
    return train_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FIXED fairness training without class collapse'
    )
    parser.add_argument('--data_dir', type=str, default='crop_part1')
    parser.add_argument('--labels_csv', type=str, default='data/train_labels.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lambda_fairness', type=float, default=0.05,
                       help='Fairness weight (keep small, 0.05-0.1)')
    parser.add_argument('--min_accuracy', type=float, default=0.5,
                       help='Minimum accuracy constraint per race')
    parser.add_argument('--output_dir', type=str, default='phase2_fixed_outputs')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("FIXED FAIRNESS TRAINING")
    print(f"{'='*80}")
    print("Key improvements:")
    print("  1. NO sample weights in loss")
    print("  2. Balanced sampling (WeightedRandomSampler)")
    print("  3. Minimum accuracy constraint")
    print("  4. Gentle fairness penalty (lambda=0.05)")
    print(f"{'='*80}\n")
    
    # Create trainer
    trainer = FairTrainerFixed(
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lambda_fairness=args.lambda_fairness,
        min_accuracy=args.min_accuracy,
        num_epochs=args.epochs
    )
    
    # Load data with balanced sampling
    train_loader = get_balanced_data_loader(
        args.data_dir,
        args.labels_csv,
        batch_size=args.batch_size
    )
    
    # Train
    model, history = trainer.train(train_loader, output_dir=args.output_dir)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}\n")
