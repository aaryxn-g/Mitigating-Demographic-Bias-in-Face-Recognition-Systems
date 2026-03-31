"""
run_advanced_eval.py
====================
Run advanced fairness evaluation on the fixed model.
"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from advanced_fairness_eval import ComprehensiveFairnessEvaluator
from train_fair_fixed import UTKFaceBalancedDataset

def main():
    print("="*80)
    print("ADVANCED FAIRNESS EVALUATION")
    print("="*80)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # 1) Build test loader (same as eval_comprehensive.py)
    print("\nLoading test dataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = UTKFaceBalancedDataset(
        root_dir='data/test',
        labels_csv='data/test_labels.csv',
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✓ Test set loaded: {len(test_dataset)} samples")
    
    # 2) Create evaluator
    print("\nInitializing comprehensive fairness evaluator...")
    evaluator = ComprehensiveFairnessEvaluator(output_dir='advanced_eval')
    
    # 3) Evaluate mitigated model
    print("\nRunning advanced fairness evaluation...")
    print("This will compute:")
    print("  - Equalized Odds (FPR/TPR parity)")
    print("  - Demographic Parity")
    print("  - Per-race Precision/Recall/F1")
    print("  - Calibration metrics")
    print("  - ROC-AUC per race")
    print("  - Confusion matrices")
    print("  - Statistical parity tests")
    print()
    
    metrics = evaluator.evaluate(
        model_path='phase2_fixed_outputs/best_fair_model_fixed.pth',
        test_loader=test_loader,
        device=device
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nResults saved to: advanced_eval/")
    print("  - fairness_report.json (comprehensive metrics)")
    print("  - fairness_visualizations.png (plots)")
    print()
    
    return metrics

if __name__ == '__main__':
    metrics = main()
    print("Done! Metrics JSON + PNG plots are in advanced_eval/")
