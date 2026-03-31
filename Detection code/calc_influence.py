import os
import sys
import time
import logging
import traceback
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
from torch.autograd import grad
from torch.utils.data import DataLoader, Dataset, Subset as TorchSubset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

# Initialize logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('influence_calculation.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class Subset(TorchSubset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        if hasattr(dataset, 'samples'):
            self.samples = [dataset.samples[i] for i in indices]
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'samples'):
            self.samples = [dataset.dataset.samples[dataset.indices[i]] for i in indices]

def flatten_grads(grads: List[torch.Tensor]) -> torch.Tensor:
    if not grads:
        raise ValueError("Cannot flatten empty gradient list")
        
    grads = [g for g in grads if g is not None and isinstance(g, torch.Tensor)]
    if not grads:
        raise ValueError("All gradients are None")
        
    device = grads[0].device
    grads = [g.to(device) for g in grads]
    
    return torch.cat([g.contiguous().view(-1) for g in grads])

def hvp(loss: torch.Tensor, 
      model_params: List[torch.Tensor], 
      vector: List[torch.Tensor]) -> List[torch.Tensor]:
    if len(vector) != len(model_params):
        raise ValueError("vector must have the same length as model_params")
    
    grad_params = torch.autograd.grad(
        loss, 
        model_params, 
        create_graph=True,
        allow_unused=True
    )
    
    grad_flat = flatten_grads(grad_params)
    vector_flat = flatten_grads(vector)
    
    grad_vector = torch.dot(grad_flat, vector_flat)
    
    hvp_flat = torch.autograd.grad(
        grad_vector, 
        model_params, 
        retain_graph=True,
        allow_unused=True
    )
    
    return [h if h is not None else torch.zeros_like(p) 
            for p, h in zip(model_params, hvp_flat)]

def recursive_inverse_hvp(
    model: nn.Module,
    criterion: callable,
    val_loader: torch.utils.data.DataLoader,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    damping: float = 0.01,  # Increased damping for stability
    scale: float = 0.01,    # Reduced scale for stable updates
    recursion_depth: int = 10,  # Start with fewer iterations
    verbose: bool = True
) -> List[torch.Tensor]:
    # Set model to train mode to enable gradient computation
    model.train()
    print("\nComputing validation gradients...")
    val_grads = None
    val_count = 0
    
    for batch in tqdm(val_loader, desc="Processing validation"):
        try:
            if len(batch) == 2:  # (inputs, targets)
                inputs, targets = batch
            else:  # Handle case where batch is a dictionary
                inputs, targets = batch['image'], batch['race']
                
            inputs = inputs.to(device).requires_grad_(True)
            targets = targets.to(device)
            
            # Forward pass with gradient computation
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets).mean()
            
            # Compute gradients
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=[p for p in model.parameters() if p.requires_grad],
                create_graph=True,
                allow_unused=True
            )
            
            # Filter out None gradients and ensure they're on the correct device
            grads = [g.detach().to(device) for g in grads if g is not None]
            if not grads:
                print("Warning: No gradients were computed for validation batch")
                continue
            
            if val_grads is None:
                val_grads = grads
            else:
                val_grads = [g_accum + g_batch for g_accum, g_batch in zip(val_grads, grads)]
            
            val_count += 1
            if val_count >= 5:
                break
                
        except Exception as e:
            print(f"Error processing validation batch: {str(e)}")
            continue
    
    if val_count == 0:
        raise ValueError("No valid validation batches processed")
    
    val_grads = [g / val_count for g in val_grads]
    print(f"Computed average validation gradient over {val_count} batches")
    
    ihvp = [0.01 * torch.randn_like(g, device=device) for g in val_grads]
    
    train_iter = iter(train_loader)
    
    print(f"\nStarting IHVP estimation with {recursion_depth} iterations...")
    
    pbar = tqdm(range(recursion_depth), desc="IHVP Estimation")
    for i in pbar:
        try:
            try:
                batch = next(train_iter)
                if len(batch) == 2:  # (inputs, targets)
                    inputs, targets = batch
                else:  # Handle case where batch is a dictionary
                    inputs, targets = batch['image'], batch['race']
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
                if len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs, targets = batch['image'], batch['race']
            
            inputs = inputs.to(device).requires_grad_(True)
            targets = targets.to(device)
            
            # Forward pass with gradient computation
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets).mean()
            
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=[p for p in model.parameters() if p.requires_grad],
                create_graph=True,
                allow_unused=True
            )
            
            # Filter out None gradients and ensure they're on the correct device
            grads = [g.to(device) if g is not None else None for g in grads]
            grads = [g for g in grads if g is not None]
            
            if not grads:
                print("Warning: No gradients were computed for training batch")
                continue
            
            # Compute H @ ihvp using the hvp function
            def hvp_fn(v):
                # Compute gradient of <grads, v>
                grad_v = torch.autograd.grad(
                    grads,
                    [p for p in model.parameters() if p.requires_grad],
                    grad_outputs=v,
                    retain_graph=True,
                    allow_unused=True
                )
                return [h.to(device) if h is not None else torch.zeros_like(p, device=device)
                       for h, p in zip(grad_v, [p for p in model.parameters() if p.requires_grad])]
            
            with torch.cuda.amp.autocast(enabled=False):
                H_ihvp = hvp_fn(ihvp)
            
            with torch.no_grad():
                for j in range(len(ihvp)):
                    try:
                        update = val_grads[j] - H_ihvp[j]
                        
                        if torch.isnan(update).any() or torch.isinf(update).any():
                            print(f"Warning: Invalid values in update at step {i}, parameter {j}")
                            continue
                            
                        update = torch.clamp(update, -1e3, 1e3)
                        
                        ihvp[j] = (1 - damping) * ihvp[j] + scale * update
                        
                        if torch.isnan(ihvp[j]).any() or torch.isinf(ihvp[j]).any():
                            print(f"Warning: Invalid values in ihvp at step {i}, parameter {j}")
                            ihvp[j] = torch.zeros_like(ihvp[j])
                            
                    except Exception as e:
                        print(f"Error updating parameter {j} at step {i}: {str(e)}")
                        continue
            
            if verbose and (i + 1) % max(1, recursion_depth // 10) == 0:  # Log every 10%
                ihvp_norm = torch.norm(torch.cat([p.view(-1) for p in ihvp])).item()
                pbar.set_description(f"IHVP [{i+1}/{recursion_depth}] | Norm: {ihvp_norm:.2e}")
        
        except Exception as e:
            print(f"Error in recursion step {i}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("IHVP computation completed")
    return ihvp

def get_original_dataset_and_index(dataset, idx):
    try:
        # Handle tensor indices
        if torch.is_tensor(idx):
            idx = idx.item()
            
        # If this is a Subset, recursively unwrap it
        if hasattr(dataset, 'dataset'):
            # Get the original indices if available, otherwise assume sequential
            if hasattr(dataset, 'indices'):
                if idx >= len(dataset.indices):
                    raise IndexError(f"Index {idx} is out of bounds for dataset with {len(dataset.indices)} samples")
                original_idx = dataset.indices[idx]
            else:
                original_idx = idx
                
            # Recursively get the base dataset and original index
            return get_original_dataset_and_index(dataset.dataset, original_idx)
        
        # If this is a ConcatDataset, find which sub-dataset contains the index
        if hasattr(dataset, 'datasets'):
            for i, sub_dataset in enumerate(dataset.datasets):
                if idx < len(sub_dataset):
                    return get_original_dataset_and_index(sub_dataset, idx)
                idx -= len(sub_dataset)
            raise IndexError(f"Index {idx} is out of bounds for ConcatDataset")
        
        # Handle UTKFaceRaceDataset sample structure
        if hasattr(dataset, 'samples'):
            if idx >= len(dataset.samples):
                raise IndexError(f"Index {idx} is out of bounds for dataset with {len(dataset.samples)} samples")
            sample = dataset.samples[idx]
            if isinstance(sample, dict):
                # For UTKFaceRaceDataset, we'll use the index as is since samples are stored as dicts
                return dataset, idx
        
        # Base case: this is the original dataset
        if hasattr(dataset, '__len__') and idx >= len(dataset):
            raise IndexError(f"Index {idx} is out of bounds for dataset with {len(dataset)} samples")
        
        return dataset, idx
        
    except Exception as e:
        # Provide more context in the error message
        dataset_size = len(dataset) if hasattr(dataset, '__len__') else 'unknown'
        dataset_type = type(dataset).__name__
        raise IndexError(
            f"Error getting original index for index {idx}. "
            f"Dataset type: {dataset_type}, size: {dataset_size}. "
            f"Error: {str(e)}"
        ) from e

def compute_influences(
    model: nn.Module,
    criterion: callable,
    train_loader: torch.utils.data.DataLoader,
    ihvp: List[torch.Tensor],
    device: torch.device,
    batch_size: int = 32,
    max_errors: int = 10  # Maximum number of errors to allow before stopping
) -> Tuple[List[int], List[float], List[dict]]:
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*50)
    logger.info("STARTING INFLUENCE COMPUTATION")
    logger.info("="*50)
    
    # Initialize variables
    all_indices = []
    all_influences = []
    metadata = []
    error_count = 0
    processed_count = 0
    total_samples = len(train_loader.dataset)
    
    logger.info(f"Starting influence computation for {total_samples} samples...")
    logger.info(f"Using device: {device}")
    logger.info(f"Model in training mode: {model.training}")
    
    # Ensure model is in evaluation mode (no dropout/batch norm)
    was_training = model.training
    model.eval()
    
    # Log model mode and device
    logger.info(f"Model device: {next(model.parameters()).device}")
    logger.info(f"IHVP device: {ihvp[0].device if ihvp else 'None'}")
    logger.info(f"Model in training mode: {model.training}")
    
    # Validate IHVP
    if not ihvp or any(v is None for v in ihvp):
        raise ValueError("Invalid IHVP: contains None values")
    
    # Flatten IHVP once
    try:
        ihvp_flat = torch.cat([i.contiguous().view(-1).to(device) for i in ihvp])
    except Exception as e:
        logger.error(f"Error flattening IHVP: {str(e)}")
        raise
    
    # Log data loader info
    logger.info(f"Data loader length: {len(train_loader)} batches")
    logger.info(f"Batch size: {train_loader.batch_size}")
    logger.info(f"Number of workers: {train_loader.num_workers}")
    logger.info(f"Dataset length: {len(train_loader.dataset)}")
    
    # Process each batch
    logger.info("Starting batch processing...")
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Computing influences")):
        if error_count >= max_errors:
            logger.warning(f"Stopping early due to {error_count} errors")
            break
        
        try:
            inputs, targets = batch
            logger.info(f"Processing batch {batch_idx} - inputs shape: {inputs.shape}, targets shape: {targets.shape}")
            
            if inputs.nelement() == 0 or targets.nelement() == 0:
                logger.warning(f"Empty batch {batch_idx} - inputs: {inputs.shape}, targets: {targets.shape}")
                continue
                
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            
            # Process each sample in the batch
            for i in range(batch_size):
                try:
                    # Calculate global sample index
                    sample_idx = batch_idx * batch_size + i
                    if sample_idx >= total_samples:
                        logger.warning(f"Sample index {sample_idx} exceeds dataset size {total_samples}")
                        continue
                    
                    # Get single sample
                    input_sample = inputs[i].unsqueeze(0).requires_grad_(True)
                    target_sample = targets[i].unsqueeze(0)
                    
                    # Forward pass with gradient computation
                    model.zero_grad()
                    try:
                        with torch.set_grad_enabled(True):
                            output = model(input_sample)
                            
                            # Calculate loss
                            loss = criterion(output, target_sample)
                            
                            # Compute gradients
                            grads = torch.autograd.grad(
                                outputs=loss,
                                inputs=[p for p in model.parameters() if p.requires_grad],
                                create_graph=True,
                                allow_unused=True
                            )
                    except Exception as e:
                        logger.error(f"Error in forward/backward pass for sample {sample_idx}: {str(e)}")
                        raise
                    
                    # Filter and validate gradients
                    grads = [g for g in grads if g is not None]
                    if not grads:
                        raise ValueError("No gradients were computed (all parameters have requires_grad=False)")
                    
                    # Flatten gradients
                    grad_flat = torch.cat([g.contiguous().view(-1) for g in grads])
                    
                    # Ensure same device as IHVP
                    grad_flat = grad_flat.to(device)
                    
                    # Compute influence score
                    try:
                        with torch.no_grad():
                            influence = -torch.dot(grad_flat, ihvp_flat).item()  # Added negative sign for correct influence calculation
                    except Exception as e:
                        logger.error(f"Error computing influence for sample {sample_idx}: {str(e)}")
                        logger.error(f"Gradient shape: {grad_flat.shape}, IHVP shape: {ihvp_flat.shape}")
                        logger.error(f"Gradient device: {grad_flat.device}, IHVP device: {ihvp_flat.device}")
                        raise
                    
                    # Get original dataset index
                    try:
                        base_dataset, true_idx = get_original_dataset_and_index(train_loader.dataset, sample_idx)
                        
                        # Store results
                        all_influences.append(influence)
                        all_indices.append(true_idx)
                        
                        # Log progress
                        processed_count += 1
                        if processed_count % 100 == 0:
                            logger.info(f"Processed {processed_count}/{total_samples} samples | "
                                      f"Current influence: {influence:.4f}")
                            
                    except Exception as idx_error:
                        error_count += 1
                        logger.warning(f"Error processing sample {sample_idx}: {str(idx_error)}")
                        all_influences.append(0.0)
                        all_indices.append(sample_idx)
                        
                except Exception as e:
                    error_count += 1
                    logger.warning(f"Error computing influence for sample {sample_idx}: {str(e)}")
                    all_influences.append(0.0)
                    all_indices.append(sample_idx)
                    
                    if error_count >= max_errors:
                        logger.warning(f"Reached maximum allowed errors ({max_errors}). Stopping...")
                        break
                        
        except Exception as batch_error:
            error_count += 1
            logger.error(f"Error processing batch {batch_idx}: {str(batch_error)}")
            logger.error(traceback.format_exc())
            
            if error_count >= max_errors:
                logger.warning(f"Reached maximum allowed errors ({max_errors}). Stopping...")
                break
    
    # Restore model's original training state
    model.train(was_training)
    
    # Log completion
    logger.info("\n" + "="*50)
    logger.info("COMPLETED INFLUENCE COMPUTATION")
    logger.info("="*50)
    logger.info(f"Processed {processed_count} samples with {error_count} errors")
    logger.info(f"All indices: {all_indices}")
    logger.info(f"All influences: {all_influences}")
    logger.info(f"Influence stats - Min: {np.min(all_influences):.4f}, "
               f"Mean: {np.mean(all_influences):.4f}, "
               f"Max: {np.max(all_influences):.4f}")
    logger.info("="*50 + "\n")
    
    # Now collect metadata for all samples
    for idx in range(len(train_loader.dataset)):
        try:
            base_dataset, true_idx = get_original_dataset_and_index(train_loader.dataset, idx)
            
            # Initialize with default values
            file_path = f'sample_{idx}'
            race = -1
            
            # Try to get metadata from UTKFaceRaceDataset
            if hasattr(base_dataset, 'samples') and len(base_dataset.samples) > true_idx:
                try:
                    sample = base_dataset.samples[true_idx]
                    if isinstance(sample, dict):
                        # Handle UTKFaceRaceDataset format
                        file_path = sample.get('image_path', f'sample_{true_idx}')
                        race = sample.get('race', -1)
                    else:
                        # Fallback for other dataset formats
                        file_path = str(sample[0]) if isinstance(sample, (list, tuple)) else f'sample_{true_idx}'
                        race = int(sample[1]) if isinstance(sample, (list, tuple)) and len(sample) > 1 else -1
                except Exception as e:
                    print(f"Warning: Could not get metadata for sample {idx} (true_idx={true_idx}): {str(e)}")
            # Fallback for datasets with separate targets
            elif hasattr(base_dataset, 'targets') and len(base_dataset.targets) > true_idx:
                try:
                    race = int(base_dataset.targets[true_idx])
                except (IndexError, ValueError, TypeError) as e:
                    print(f"Warning: Could not get race from targets for sample {idx}: {str(e)}")
            
            # Get influence score if available
            influence_score = all_influences[idx] if idx < len(all_influences) else 0.0
            
            metadata.append({
                'index': idx,
                'original_index': true_idx,
                'file_path': file_path,
                'race': race,
                'influence_score': influence_score
            })
            
        except Exception as e:
            print(f"Error getting metadata for index {idx}: {str(e)}")
            metadata.append({
                'index': idx,
                'original_index': -1,
                'file_path': f'error_{idx}',
                'race': -1,
                'influence_score': all_influences[idx] if idx < len(all_influences) else 0.0
            })
    
    return all_indices, all_influences, metadata

def analyze_bias(influence_df: pd.DataFrame, output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Basic statistics
        print("\nInfluence Score Statistics:")
        print(f"Mean influence: {influence_df['influence'].mean():.6f}")
        print(f"Std influence: {influence_df['influence'].std():.6f}")
        print(f"Min influence: {influence_df['influence'].min():.6f}")
        print(f"Max influence: {influence_df['influence'].max():.6f}")
        
        # Group by race and compute statistics
        if 'race' in influence_df.columns:
            race_stats = influence_df.groupby('race')['influence'].agg(
                ['count', 'mean', 'std', 'min', 'max']
            ).reset_index()
            
            print("\nInfluence by Race:")
            print(race_stats)
            
            # Save to CSV
            race_stats.to_csv(os.path.join(output_dir, 'influence_by_race.csv'), index=False)
            
            # Plot influence distribution by race
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='race', y='influence', data=influence_df)
                plt.title('Influence Score Distribution by Race')
                plt.xlabel('Race')
                plt.ylabel('Influence')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'influence_by_race.png'))
                plt.close()
            except ImportError:
                print("Matplotlib/seaborn not available for plotting")
        
        # Save top and bottom influential samples
        if 'file_path' in influence_df.columns:
            top_influential = influence_df.nlargest(10, 'influence')
            bottom_influential = influence_df.nsmallest(10, 'influence')
            top_influential.to_csv(os.path.join(output_dir, 'top_influential.csv'), index=False)
            bottom_influential.to_csv(os.path.join(output_dir, 'bottom_influential.csv'), index=False)
            print("\nTop influential samples:")
            print(top_influential[['file_path', 'race', 'influence']])
            
            # Save influence statistics by race
            stats = influence_df.groupby('race')['influence'].agg(['mean', 'std', 'count'])
            stats.to_csv(os.path.join(output_dir, 'influence_stats_by_race.csv'))
            
            # Top influential samples by race
            top_by_race = influence_df.groupby('race').apply(
                lambda x: x.nlargest(10, 'influence')
            ).reset_index(drop=True)
            top_by_race.to_csv(os.path.join(output_dir, 'top_influential_by_race.csv'), index=False)
            
            # Negative influence (potentially harmful) samples
            negative_influence = influence_df[influence_df['influence'] < 0]
            if not negative_influence.empty:
                negative_by_race = negative_influence.groupby('race').apply(
                    lambda x: x.nsmallest(10, 'influence')
                ).reset_index(drop=True)
                negative_by_race.to_csv(os.path.join(output_dir, 'negative_influence_by_race.csv'), index=False)
        
        print(f"\nAnalysis results saved to {os.path.abspath(output_dir)}")
        
    except Exception as e:
        print(f"Error in bias analysis: {str(e)}")
        raise

class UTKFaceRaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()
        
        self._print_dataset_stats()
        
    def _load_samples(self):
        """Load dataset samples with validation."""
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
            
        samples = []
        skipped_files = 0
        
        for fname in os.listdir(self.root_dir):
            if not (fname.endswith('.jpg') or fname.endswith('.png') or '.chip' in fname):
                continue
                
            try:
                # Remove any file extensions (e.g., .chip.jpg)
                base = fname.split('.')[0]
                parts = base.split('_')
                
                # UTKFace format: [age]_[gender]_[race]_[date&time].jpg
                if len(parts) >= 4 and parts[2].isdigit():
                    race = int(parts[2])
                    if 0 <= race <= 4:  # Only include valid race labels (0-4)
                        samples.append({
                            'image_path': os.path.join(self.root_dir, fname),
                            'race': race,
                            'age': int(parts[0]) if parts[0].isdigit() else -1,
                            'gender': int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1,
                            'filename': fname
                        })
                    else:
                        print(f"Skipping {fname}: race label {race} out of range (0-4)")
                        skipped_files += 1
                else:
                    print(f"Skipping {fname}: invalid filename format")
                    skipped_files += 1
                    
            except Exception as e:
                print(f"Error processing {fname}: {str(e)}")
                skipped_files += 1
        
        if not samples:
            raise ValueError(f"No valid samples found in {self.root_dir}. Check the directory and file formats.")
            
        if skipped_files > 0:
            print(f"Skipped {skipped_files} files due to errors or invalid format")
            
        return samples
    
    def _print_dataset_stats(self):
        """Print dataset statistics."""
        if not self.samples:
            return
            
        race_counts = {}
        for sample in self.samples:
            race = sample['race']
            race_counts[race] = race_counts.get(race, 0) + 1
            
        print("\n" + "="*50)
        print(f"Loaded {len(self.samples)} samples from {self.root_dir}")
        print("-"*50)
        print("Race distribution:")
        for race, count in sorted(race_counts.items()):
            print(f"  Race {race}: {count} samples")
        print("="*50 + "\n")
        
        # Print first few samples for validation
        print("Sample validation (first 5 samples):")
        for i in range(min(5, len(self.samples))):
            sample = self.samples[i]
            print(f"  Sample {i}: {os.path.basename(sample['image_path'])} - Race: {sample['race']}, "
                  f"Age: {sample.get('age', 'N/A')}, Gender: {sample.get('gender', 'N/A')}")
        print()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        if not isinstance(idx, int):
            raise TypeError(f"Index must be an integer, got {type(idx)}")
            
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} is out of bounds for dataset with {len(self.samples)} samples")
            
        sample = self.samples[idx]
        
        try:
            # Load image
            if not os.path.exists(sample['image_path']):
                raise FileNotFoundError(f"Image not found: {sample['image_path']}")
                
            image = Image.open(sample['image_path']).convert('RGB')
            race = sample['race']
            
            # Apply transformations if specified
            if self.transform:
                image = self.transform(image)
                
            return image, race
            
        except Exception as e:
            print(f"Error loading image {sample.get('filename', 'unknown')}: {str(e)}")
            
            # Return a zero tensor with the correct shape as a fallback
            if self.transform:
                # Assuming 3-channel image with standard ImageNet normalization
                dummy_img = torch.zeros(3, 224, 224)  # Default to 224x224 which is common for CNNs
                return dummy_img, -1  # -1 indicates an invalid sample
            return None, -1
    
    def get_race_distribution(self):
        """Get the distribution of race labels in the dataset."""
        race_counts = {}
        for sample in self.samples:
            race = sample['race']
            race_counts[race] = race_counts.get(race, 0) + 1
        return race_counts

def get_data_loaders(data_dir, batch_size=32, val_split=0.2, test_split=0.1, subsample_ratio=0.05):
    """Create train, validation, and test data loaders with stratified subsampling and splitting."""
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
def setup_logging():
    """Set up logging configuration"""
    # Only log WARNING and above to console, INFO and above to file
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    
    file_handler = logging.FileHandler('influence_calculation.log')
    file_handler.setLevel(logging.INFO)
    
    # Simple format for console, more detailed for file
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    console.setFormatter(console_format)
    file_handler.setFormatter(file_format)
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add our handlers
    logger.addHandler(console)
    logger.addHandler(file_handler)
    
    return logger

def main():
    """Main function to run the influence calculation pipeline."""
    # Set up logging
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Starting influence calculation script")
    logger.info("-" * 80)
    
    try:
        torch.manual_seed(42)
        logger.info(f"Set random seed to 42")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        logger.info(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Device Count: {torch.cuda.device_count()}")
            logger.info(f"CUDA Current Device: {torch.cuda.current_device()}")
    
        logger.info("Initializing ResNet18 model...")
        # Use the new weights parameter instead of pretrained
        model = models.resnet18(weights=None)  # Initialize with random weights
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)  # 5 race categories
        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Load the trained model weights if available
        model_path = "final_race_model.pth"
        model_abs_path = os.path.abspath(model_path)
        logger.info(f"Looking for model weights at: {model_abs_path}")
        
        if os.path.exists(model_path):
            try:
                logger.info("Loading model weights...")
                # Load the entire checkpoint first
                checkpoint = torch.load(model_path, map_location=device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    # If checkpoint contains a 'state_dict' key
                    model.load_state_dict(checkpoint['state_dict'])
                elif isinstance(checkpoint, dict):
                    # If checkpoint is already a state dict
                    model.load_state_dict(checkpoint)
                else:
                    # If it's a direct model state dict
                    model.load_state_dict(checkpoint)
                    
                logger.info(f"Successfully loaded model from {model_path}")
                
                # Log model architecture
                logger.info("Model architecture:")
                logger.info(model)
                
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                logger.error(traceback.format_exc())
                logger.warning("Continuing with random weights")
        else:
            logger.warning(f"Model file {model_path} not found. Using random weights.")
            logger.warning("This may lead to poor influence calculation results.")
            
        model = model.to(device)
        logger.info(f"Model moved to {device}")
        
        # Set model to train mode to enable gradient computation
        model.train()
        logger.info("Model set to training mode")
        
        # Ensure all parameters require gradients
        for name, param in model.named_parameters():
            param.requires_grad = True
        logger.info("All model parameters set to require gradients")
    
        # Model setup is complete
        
        # Data loading and preprocessing
        logger.info("\n" + "="*50)
        logger.info("DATA LOADING")
        logger.info("="*50)
        
        # Set up data directory - look in current directory
        data_dir = "crop_part1"  # Changed to look in current directory
        data_path = os.path.join(os.getcwd(), data_dir)
        
        if not os.path.exists(data_path):
            logger.error(f"Data directory not found at {data_path}")
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Directory contents: {os.listdir('.')}")
            raise FileNotFoundError(f"Data directory not found at {data_path}")
            
        logger.info(f"Using data directory: {data_path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Directory contents: {os.listdir('.')}")
        
        # Count number of files in data directory
        try:
            num_files = len([f for f in os.listdir(data_path) if f.endswith('.jpg')])
            logger.info(f"Found {num_files} image files in {data_path}")
            
            if num_files == 0:
                logger.error(f"No image files found in the data directory: {data_path}")
                raise ValueError(f"No image files found in the data directory: {data_path}")
                
        except Exception as e:
            logger.error(f"Error reading data directory: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Create data loaders for IHVP computation (using train/val split)
        logger.info("Creating data loaders for IHVP computation...")
        logger.info("Parameters: batch_size=32, val_split=0.1, test_split=0.1, subsample_ratio=1.0")
        
        try:
            train_loader, val_loader, test_loader = get_data_loaders(
                data_dir, 
                batch_size=32,
                val_split=0.1, 
                test_split=0.1,
                subsample_ratio=1.0
            )
            
            # Log dataset sizes
            logger.info(f"Training set size: {len(train_loader.dataset)} samples")
            logger.info(f"Validation set size: {len(val_loader.dataset)} samples")
            logger.info(f"Test set size: {len(test_loader.dataset)} samples")
            
            if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
                raise ValueError("One or more datasets are empty. Please check your data loading pipeline.")
                
        except Exception as e:
            logger.error(f"Error creating data loaders: {str(e)}")
            logger.error(traceback.format_exc())
            raise   
        logger.info("Using CrossEntropyLoss as the loss function")
        criterion = nn.CrossEntropyLoss()
        logger.info("Loss function initialized")
    
        logger.info("\n" + "="*50)
        logger.info("IHVP COMPUTATION")
        logger.info("="*50)
        
        logger.info("Starting IHVP computation with optimized parameters:")
        logger.info(f"- Damping: 0.01")
        logger.info(f"- Scale: 0.01")
        logger.info(f"- Recursion depth: 10")
        logger.info(f"- Device: {device}")
        
        # Track computation time
        start_time = time.time()
        
        try:
            ihvp = recursive_inverse_hvp(
                model=model,
                criterion=criterion,
                val_loader=val_loader,
                train_loader=train_loader,
                device=device,
                damping=0.01,
                scale=0.01,
                recursion_depth=10,
                verbose=True
            )
            
            elapsed = time.time() - start_time
            logger.info(f"IHVP computation completed in {elapsed/60:.2f} minutes")
            
            if ihvp is not None:
                ihvp_norms = [torch.norm(p).item() for p in ihvp]
                logger.info(f"IHVP parameter norms (min/mean/max): "
                          f"{min(ihvp_norms):.4f} / "
                          f"{sum(ihvp_norms)/len(ihvp_norms):.4f} / "
                          f"{max(ihvp_norms):.4f}")
            else:
                logger.error("IHVP computation returned None")
                raise ValueError("IHVP computation failed - returned None")
            
            results_dir = "all_influence"
            os.makedirs(results_dir, exist_ok=True)
            ihvp_path = os.path.join(results_dir, "ihvp.pt")
            
            try:
                torch.save(ihvp, ihvp_path)
                logger.info(f"Saved IHVP to {os.path.abspath(ihvp_path)}")
                
                if os.path.exists(ihvp_path):
                    file_size = os.path.getsize(ihvp_path) / (1024 * 1024)  # in MB
                    logger.info(f"IHVP file size: {file_size:.2f} MB")
                else:
                    logger.warning(f"IHVP file not found at {ihvp_path} after saving")
                    
            except Exception as e:
                logger.error(f"Error saving IHVP: {str(e)}")
                logger.error(traceback.format_exc())
                raise
        
            logger.info("\n" + "="*50)
            logger.info("FULL DATASET INFLUENCE COMPUTATION")
            logger.info("="*50)
            
            # Get the full dataset (without any splits)
            full_dataset = UTKFaceRaceDataset(
                root_dir=data_dir,
                transform=transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            )
            
            # Create a single DataLoader for the full dataset
            full_loader = DataLoader(
                full_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            logger.info(f"Processing full dataset with {len(full_dataset)} samples")
            logger.info("Starting influence computation on full dataset...")
            
            # Track computation time
            inf_start_time = time.time()
            
            # Compute influences on the full dataset
            indices, influence_scores, metadata = compute_influences(
                model=model,
                criterion=criterion,
                train_loader=full_loader,  # Use full dataset loader
                ihvp=ihvp,
                device=device,
                batch_size=32,
                max_errors=10
            )
            if influence_scores:
                    scores = np.array(influence_scores)
                    logger.info(f"Computed {len(scores)} influence scores")
                    logger.info(f"Influence scores - Min: {scores.min():.4f}, "
                              f"Mean: {scores.mean():.4f}, "
                              f"Max: {scores.max():.4f}")
                    logger.info(f"Positive influences: {np.sum(scores > 0)} "
                              f"({np.mean(scores > 0)*100:.2f}%)")
            else:
                    logger.warning("No influence scores were computed")
                    return
                    
                # Save Results
            logger.info("\n" + "="*50)
            logger.info("SAVING RESULTS")
            logger.info("="*50)
                
            try:
                    # Ensure all arrays are the same length
                    min_length = min(len(indices), len(influence_scores), len(metadata))
                    if min_length == 0:
                        raise ValueError("No valid data to save")
                        
                    # Truncate all arrays to the minimum length
                    indices = indices[:min_length]
                    influence_scores = influence_scores[:min_length]
                    metadata = metadata[:min_length]
                    
                    logger.info(f"Saving {min_length} valid samples")
                    
                    # Prepare results dictionary
                    results = {
                        'indices': indices,
                        'influence_scores': influence_scores,
                        'metadata': metadata,
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'parameters': {
                            'damping': 0.01,
                            'scale': 0.01,
                            'recursion_depth': 10,
                            'batch_size': 4
                        }
                    }
                    
                    # Save results to a .pt file
                    results_path = os.path.join(results_dir, "influence_results.pt")
                    torch.save(results, results_path)
                    logger.info(f"Saved influence results to {os.path.abspath(results_path)}")
                    
                    # Create DataFrame with proper error handling
                    try:
                        df_data = {
                            'index': indices,
                            'influence': influence_scores
                        }
                        
                        # Add metadata fields if they exist
                        meta_fields = ['race', 'age', 'gender', 'file_path']
                        for field in meta_fields:
                            field_values = []
                            for m in metadata:
                                try:
                                    field_values.append(m.get(field, -1))
                                except (AttributeError, KeyError):
                                    field_values.append(-1)
                            df_data[field] = field_values
                        
                        df = pd.DataFrame(df_data)
                        
                        # Save full dataset
                        csv_path = os.path.join(results_dir, "full_dataset_influence_scores.csv")
                        df.to_csv(csv_path, index=False)
                        logger.info(f"Saved full dataset influence scores to {os.path.abspath(csv_path)}")
                        
                        # Save sample
                        sample_size = min(1000, len(df))
                        if sample_size > 0:
                            sample_csv_path = os.path.join(results_dir, "sample_influence_scores.csv")
                            df.sample(sample_size).to_csv(sample_csv_path, index=False)
                            logger.info(f"Saved sample influence scores to {os.path.abspath(sample_csv_path)}")
                        
                        # Run bias analysis if we have data
                        if len(df) > 0:
                            logger.info("\n" + "="*50)
                            logger.info("BIAS ANALYSIS")
                            logger.info("="*50)
                            
                            try:
                                analyze_bias(df, output_dir=results_dir)
                                logger.info("Bias analysis completed successfully")
                            except Exception as e:
                                logger.error(f"Error during bias analysis: {str(e)}")
                                logger.error(traceback.format_exc())
                        
                        logger.info("\n" + "="*50)
                        logger.info("SCRIPT COMPLETED SUCCESSFULLY")
                        logger.info("="*50)
                        
                    except Exception as e:
                        logger.error(f"Error creating CSV files: {str(e)}")
                        logger.error(traceback.format_exc())
                        raise
                        
            except Exception as e:
                    logger.error(f"Error saving results: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise
                
        except KeyboardInterrupt:
            logger.warning("\nScript was interrupted by user. Partial results may be available in the results/ directory.")
            raise
    except Exception as e:
        logger.error(f"\nAn unexpected error occurred: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("\nScript execution completed")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()