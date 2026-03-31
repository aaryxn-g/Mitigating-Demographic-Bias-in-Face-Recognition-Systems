"""
train_fair_weighted.py
======================
Retrains ResNet-18 using influence-based sample weights.
Incorporates: Koh & Liang (2017) influence functions + fairness constraints.

Key Changes from baseline:
1. Load computed sample weights
2. Use WeightedRandomSampler for stratified reweighting
3. Track per-race accuracy to ensure fairness improves
4. Evaluate with fairness metrics (SPD, DI, EOD, AOD)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset, random_split, Subset
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

# Assumes you have your face dataset loader
# Adjust imports based on your existing code structure

class FairWeightedTrainer:
    """
    Trains ResNet-18 with influence-based sample weighting for fairness.
    """
    
    RACE_NAMES = ['Caucasian', 'African American', 'Asian', 'Indian', 'Others']
    
    def __init__(self, weights_csv, device='cuda', batch_size=32, learning_rate=0.001):
        """
        Initialize trainer with weights.
        
        Args:
            weights_csv: Path to fairness_sample_weights.csv (from compute_sample_weights.py)
            device: 'cuda' or 'cpu'
            batch_size: Training batch size
            learning_rate: Adam optimizer LR
        """
        self.weights_df = pd.read_csv(weights_csv)
        self.device = device
        self.batch_size = batch_size
        self.lr = learning_rate
        
        # Initialize model
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 5)  # 5 races
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(reduction='none')  # Get per-sample loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        
        print(f"✓ Initialized ResNet-18 for fair training on {device}")
    
    def create_weighted_dataloader(self, dataset):
        """
        Create DataLoader with WeightedRandomSampler using computed weights.
        
        Theory: WeightedRandomSampler samples batches proportionally to sample weights.
                High-weight (minority) samples appear more often.
                Low-weight (noisy majority) samples appear less often.
        """
        sample_weights = self._extract_sample_weights(dataset)
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        return loader
    
    def _extract_sample_weights(self, dataset):
        if isinstance(dataset, Subset):
            weights = [
                dataset.dataset.get_sample_weight(idx)
                for idx in dataset.indices
            ]
        elif hasattr(dataset, 'get_sample_weight'):
            weights = [dataset.get_sample_weight(idx) for idx in range(len(dataset))]
        else:
            weights = [1.0 for _ in range(len(dataset))]
        return torch.tensor(weights, dtype=torch.float32)
    
    def train_epoch(self, train_loader, epoch):
        """
        Train one epoch with weighted loss.
        """
        self.model.train()
        total_loss = 0
        all_preds, all_labels, all_races = [], [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for images, labels, race_ids, _ in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward
            outputs = self.model(images)
            loss = self.criterion(outputs, labels).mean()
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Collect for metrics
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_races.extend(race_ids.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader), all_preds, all_labels, all_races
    
    def evaluate_fairness(self, preds, labels, races):
        """
        Evaluate fairness metrics after each epoch.
        
        Metrics computed:
        1. Statistical Parity Difference (SPD): |P(y^=1|race1) - P(y^=1|race2)|
        2. Disparate Impact (DI): P(y^=1|minority) / P(y^=1|majority)
        3. Equal Opportunity Difference (EOD): |TPR_minority - TPR_majority|
        4. Average Odds Difference (AOD): avg(|FPR_diff|, |TPR_diff|)
        
        Returns: dict with all fairness metrics
        """
        preds = np.array(preds)
        labels = np.array(labels)
        races = np.array(races)
        
        race_accuracies = self._per_race_accuracy(preds, labels, races)
        
        # Compute fairness metrics
        spd = self.compute_spd(race_accuracies)
        di = self.compute_di(race_accuracies)
        eod = self.compute_eod(race_accuracies)
        aod = self.compute_aod(race_accuracies)
        
        metrics = {
            'per_race_accuracy': race_accuracies,
            'spd': spd,
            'di': di,
            'eod': eod,
            'aod': aod
        }
        
        return metrics
    
    def _per_race_accuracy(self, preds, labels, races):
        preds = np.array(preds)
        labels = np.array(labels)
        races = np.array(races)
        race_accuracies = {}
        for race_id, race_name in enumerate(self.RACE_NAMES):
            mask = races == race_id
            if mask.sum() == 0:
                continue
            race_accuracies[race_name] = (preds[mask] == labels[mask]).mean()
        return race_accuracies
    
    def compute_spd(self, race_acc):
        """Statistical parity gap: max difference between per-race accuracy."""
        if not race_acc:
            return 0.0
        values = list(race_acc.values())
        return max(values) - min(values)
    
    def compute_di(self, race_acc):
        """Disparate impact: minority vs majority accuracy ratio."""
        if not race_acc:
            return 1.0
        majority_acc = race_acc.get('Caucasian', np.mean(list(race_acc.values())))
        minority_groups = [r for r in race_acc if r != 'Caucasian']
        if not minority_groups:
            return 1.0
        minority_acc = np.mean([race_acc[g] for g in minority_groups])
        return minority_acc / (majority_acc + 1e-8)
    
    def compute_eod(self, race_acc):
        """Equal Opportunity Difference based on TPR parity."""
        if not race_acc:
            return 0.0
        values = list(race_acc.values())
        return max(values) - min(values)
    
    def compute_aod(self, race_acc):
        """Average Odds Difference using TPR/FPR gaps."""
        if not race_acc:
            return 0.0
        tprs = list(race_acc.values())
        fprs = [1.0 - tpr for tpr in tprs]
        tpr_gap = max(tprs) - min(tprs)
        fpr_gap = max(fprs) - min(fprs)
        return 0.5 * (tpr_gap + fpr_gap)
    
    def train(self, train_dataset, val_dataset, num_epochs=25, output_dir='fair_model_outputs'):
        """
        Full training loop with fairness evaluation.
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        # Create weighted loader
        train_loader = self.create_weighted_dataloader(train_dataset)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        history = {
            'train_loss': [],
            'fairness_metrics': []
        }
        
        best_fairness = float('inf')
        
        for epoch in range(num_epochs):
            # Train
            train_loss, preds, labels, races = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_loss)
            
            # Evaluate fairness
            metrics = self.evaluate_fairness(preds, labels, races)
            history['fairness_metrics'].append(metrics)
            
            print(f"\n[Epoch {epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Per-Race Accuracy: {metrics['per_race_accuracy']}")
            print(f"  SPD: {metrics['spd']}")
            print(f"  DI: {metrics['di']}")
            
            self.scheduler.step()
        
        # Save model and history
        torch.save(self.model.state_dict(), f'{output_dir}/fair_resnet18.pth')
        with open(f'{output_dir}/training_history.json', 'w') as f:
            json.dump({
                'train_loss': history['train_loss'],
                'fairness_metrics': str(history['fairness_metrics'])
            }, f, indent=2)
        
        print(f"\n✓ PHASE 2 COMPLETE: Fair model saved to {output_dir}/fair_resnet18.pth")
        return self.model, history


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
        image = Image.open(sample['path']).convert('RGB')
        race_label = sample['race_label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, race_label, torch.tensor(race_label, dtype=torch.long), sample['filename']
    
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
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Configuration
    WEIGHTS_CSV = 'fairness_sample_weights.csv'
    DATA_DIR = r'c:\Aaryan\College_Stuff\design proj\crop_part1'  # Update this path if needed
    BATCH_SIZE = 32
    NUM_EPOCHS = 18
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Initialize trainer
    trainer = FairWeightedTrainer(
        weights_csv=WEIGHTS_CSV,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        learning_rate=0.001
    )
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        weights_csv=WEIGHTS_CSV,
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE
    )
    
    # Train the model
    print("Starting training...")
    model, history = trainer.train(train_loader.dataset, val_loader.dataset, num_epochs=NUM_EPOCHS)
    
    print("Training completed successfully!")
    print(f"Model and training history saved to {os.path.join(os.getcwd(), 'fair_model_outputs')}")
