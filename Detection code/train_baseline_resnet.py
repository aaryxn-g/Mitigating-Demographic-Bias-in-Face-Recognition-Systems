from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import transforms
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
from sklearn.model_selection import train_test_split

def stratified_split(dataset, val_size=0.2, test_size=0.1):
    """Create stratified splits based on race labels."""
    # Extract race labels from the dataset
    targets = [dataset.samples[i]['race'] for i in range(len(dataset))]
    
    # First split: separate test set
    train_val_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        stratify=targets,
        random_state=42
    )
    
    # Second split: separate validation set from remaining data
    train_val_targets = [targets[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (1 - test_size),  # Adjust for initial test split
        stratify=train_val_targets,
        random_state=42
    )
    
    return train_idx, val_idx, test_idx

def get_data_loaders(data_dir, batch_size=32, val_split=0.2, test_split=0.1, subsample_ratio=0.05):  # Changed default to 5%
    """Create train, validation, and test data loaders with stratified subsampling and splitting.
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for data loaders
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        subsample_ratio: Fraction of the full dataset to use (for faster training)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create full dataset
    full_dataset = UTKFaceRaceDataset(root_dir=data_dir, transform=transform)
    
    # Perform stratified subsampling
    if subsample_ratio < 1.0:
        print(f"\nPerforming stratified subsampling (keeping {subsample_ratio*100:.0f}% of data)...")
        # Get all indices and corresponding race labels
        all_indices = list(range(len(full_dataset)))
        race_labels = [full_dataset.samples[i]['race'] for i in all_indices]
        
        # Perform stratified split to get the subsampled indices
        subsampled_indices, _ = train_test_split(
            all_indices,
            test_size=1 - subsample_ratio,
            stratify=race_labels,
            random_state=42
        )
        
        # Create a subset with the sampled indices
        dataset = Subset(full_dataset, subsampled_indices)
        
        # Extract race labels for the subsampled dataset
        targets = [full_dataset.samples[i]['race'] for i in subsampled_indices]
    else:
        dataset = full_dataset
        targets = [sample['race'] for sample in full_dataset.samples]
    
    # Get stratified splits for train/val/test
    train_val_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=test_split,
        stratify=targets,
        random_state=42
    )
    
    # Get targets for train_val split
    train_val_targets = [targets[i] for i in train_val_idx]
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_split / (1 - test_split),  # Adjust for initial test split
        stratify=train_val_targets,
        random_state=42
    )
    
    # Create subset datasets
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)
    
    # Log dataset statistics
    print("\nDataset Statistics:")
    print(f"Original dataset size: {len(full_dataset)}")
    if subsample_ratio < 1.0:
        print(f"Subsampled dataset size: {len(dataset)} ({subsample_ratio*100:.0f}% of original)")
    print(f"Training samples: {len(train_set)}")
    print(f"Validation samples: {len(val_set)}")
    print(f"Test samples: {len(test_set)}")
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, 
                          shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader

class RaceMetricsLogger:
    """Helper class to log race-wise metrics during training."""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.metrics = []
        
    def log_epoch(self, epoch, phase, metrics_dict):
        """Log metrics for a single epoch."""
        # Add to metrics history
        entry = {'epoch': epoch, 'phase': phase}
        entry.update(metrics_dict)
        self.metrics.append(entry)
        
        # Log to TensorBoard
        for metric_name, value in metrics_dict.items():
            if 'race' in metric_name:
                race = metric_name.split('_')[-1]
                self.writer.add_scalar(f'Race/{race}/{metric_name}', value, epoch)
            else:
                self.writer.add_scalar(f'{phase}/{metric_name}', value, epoch)
        
        # Save to CSV
        df = pd.DataFrame(self.metrics)
        df.to_csv(os.path.join(self.log_dir, 'training_metrics.csv'), index=False)
    
    def close(self):
        self.writer.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    best_acc = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Setup logging
    log_dir = f"logs/resnet_race_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = RaceMetricsLogger(log_dir)
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        race_correct = [0] * 5
        race_total = [0] * 5
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
            
            # Update metrics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Update per-race metrics
            for race in range(5):
                mask = (labels == race)
                race_correct[race] += (preds[mask] == labels[mask]).sum().item()
                race_total[race] += mask.sum().item()
        
        # Calculate training metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        # Log training metrics
        train_metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_acc.item()
        }
        # Add per-race accuracy
        for race in range(5):
            acc = race_correct[race] / race_total[race] if race_total[race] > 0 else 0
            train_metrics[f'race_{race}_acc'] = acc
        
        logger.log_epoch(epoch, 'train', train_metrics)
        
        # Validation phase
        val_loss, val_acc, race_acc = evaluate_model(model, val_loader, criterion, device)
        
        # Log validation metrics
        val_metrics = {
            'loss': val_loss,
            'accuracy': val_acc.item()
        }
        # Add per-race accuracy
        for race, acc in enumerate(race_acc):
            val_metrics[f'race_{race}_acc'] = acc
        
        logger.log_epoch(epoch, 'val', val_metrics)
        
        # Print epoch summary
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        print('Per-race Val Acc:')
        race_names = ['Caucasian', 'African American', 'Asian', 'Indian', 'Others']
        for name, acc in zip(race_names, race_acc):
            print(f'  {name}: {acc:.4f}')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
            print(f'New best model saved with val acc: {best_acc:.4f}')
    
    logger.close()
    return model

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    race_correct = [0] * 5
    race_total = [0] * 5
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Calculate per-race metrics
            for race in range(5):
                mask = (labels == race)
                race_correct[race] += (preds[mask] == labels[mask]).sum().item()
                race_total[race] += mask.sum().item()
    
    # Calculate metrics
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)
    
    # Calculate per-race accuracy
    race_acc = [c / t if t > 0 else 0 for c, t in zip(race_correct, race_total)]
    
    return epoch_loss, epoch_acc, race_acc

class UTKFaceRaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        for fname in os.listdir(self.root_dir):
            if fname.endswith('.jpg'):
                try:
                    parts = fname.split('_')
                    if len(parts) >= 3:
                        race_label = int(parts[2])
                        if 0 <= race_label <= 4:  # 5 race categories
                            samples.append({
                                'image_path': os.path.join(self.root_dir, fname),
                                'race': race_label
                            })
                except (ValueError, IndexError):
                    continue
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        race = sample['race']
        
        if self.transform:
            image = self.transform(image)
            
        return image, race

def main():
    # Configuration
    data_dir = r"C:\Aaryan\College_Stuff\design proj\crop_part1"
    batch_size = 32
    num_epochs = 25
    learning_rate = 0.001
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size)
    
    # Initialize model
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)  # 5 race categories
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    print("Starting training...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loss, test_acc, race_acc = evaluate_model(model, test_loader, criterion, device)
    
    # Print results
    race_names = ['Caucasian', 'African American', 'Asian', 'Indian', 'Others']
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print("\nPer-race Accuracy:")
    for name, acc in zip(race_names, race_acc):
        print(f"{name}: {acc:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), 'final_race_model.pth')
    print("\nTraining complete! Model saved as 'final_race_model.pth'")

if __name__ == "__main__":
    main()