# Based on HisToGene https://github.com/maxpmx/HisToGene/tree/main

#!/usr/bin/env python3
# coding:utf-8 

"""
HisToGene Pathway Prediction Training Script
Modified from gene prediction to pathway prediction using AUCell/UCell scores
"""

import random
import os
import argparse
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from dataset_pathway import ViT_HER2ST_Pathway, SKIN_Pathway
from predict_pathway import model_predict_pathway
from utils import *
from vis_model import HisToGene

# Command line argument support

parser = argparse.ArgumentParser(description='HisToGene pathway prediction training')
parser.add_argument('--fold', type=int, required=True, 
                    help='Fold number for cross-validation (which sample to use as test set)')
parser.add_argument('--pathway', type=str, required=True,
                    choices=['GO_0034340', 'GO_0060337', 'GO_0071357',
                             'KEGG_P53_SIGNALING_PATHWAY', 'KEGG_BREAST_CANCER', 'KEGG_APOPTOSIS',
                             'BREAST_CANCER_LUMINAL_A', 'HER2_AMPLICON_SIGNATURE', 'HALLMARK_MYC_TARGETS_V1'],
                    help='Pathway to train on')
parser.add_argument('--score', type=str, required=True, choices=['AUCell', 'UCell'],
                    help='Scoring method to use (AUCell or UCell)')
parser.add_argument('--dataset', type=str, default='HER2', choices=['HER2', 'Skin'],
                    help='Dataset to use (HER2 or Skin)')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-5,
                    help='Learning rate')
parser.add_argument('--n_layers', type=int, default=8,
                    help='Number of transformer layers')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of data loading workers')
parser.add_argument('--device', type=str, default='auto',
                    help='Device to use (auto/cpu/cuda)')


def train_pathway(fold, pathway, score_method, dataset_type='HER2', 
                 epochs=100, learning_rate=1e-5, n_layers=8,
                 batch_size=1, num_workers=4, device='auto'):
    """
          Train HisToGene for single pathway prediction
    """
    
    print(f"Training HisToGene for pathway prediction:")
    print(f"Pathway: {pathway}")
    print(f"Score method: {score_method}")
    print(f"Dataset: {dataset_type}")
    print(f"Fold: {fold}")
    print(f"Output dimension: 1 (single pathway)")
    
    # Create pathway dataset

    if dataset_type == 'HER2':
        dataset = ViT_HER2ST_Pathway(
            train=True, 
            fold=fold,
            pathway=pathway,
            score_method=score_method
        )
    elif dataset_type == 'Skin':
        dataset = SKIN_Pathway(
            train=True,
            fold=fold,
            pathway=pathway,
            score_method=score_method
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=True
    )
    
    # Create model with single pathway output

    tagname = f"-htg_{pathway}_{score_method}_cv"
    
    # Model with single output dimension for pathway prediction
    model = HisToGene(
        n_genes=1,  # Changed from multiple genes to single pathway
        learning_rate=learning_rate,
        n_layers=n_layers
    )

    # Setup logger

    log_name = f"pathway_log_{tagname}_{fold}"
    mylogger = CSVLogger(save_dir="./logs/", name=log_name)
    
    # Setup trainer with device detection

    if device == 'auto':
        if torch.cuda.is_available():
            device_config = {"accelerator": "gpu", "devices": 1}
            print(f"Using GPU for training")
        else:
            device_config = {"accelerator": "cpu"}
            print("Using CPU for training")
    elif device == 'cuda':
        device_config = {"accelerator": "gpu", "devices": 1}
    else:
        device_config = {"accelerator": "cpu"}

    trainer = pl.Trainer(
        **device_config,
        max_epochs=epochs, 
        logger=mylogger,
        enable_progress_bar=True,
        log_every_n_steps=10
    )

    # Train model

    print("Starting pathway prediction training...")
    trainer.fit(model, train_loader)

    # Save model

    model_save_dir = f"./model/pathway/{score_method}/{pathway}/"
    os.makedirs(model_save_dir, exist_ok=True)
    model_filename = f"last_train_{tagname}_{fold}.ckpt"
    model_path = model_save_dir + model_filename
    
    trainer.save_checkpoint(model_path)
    print(f"Model saved to: {model_path}")

    print("Pathway training completed!")
    return model_path


def tutorial_example():
    """
    Tutorial example similar to the original HisToGene tutorial
    """
    print("=" * 60)
    print("HisToGene Pathway Prediction Tutorial")
    print("=" * 60)
    
    # Parameters
    fold = 5
    pathway = 'KEGG_P53_SIGNALING_PATHWAY'
    score_method = 'AUCell'
    dataset_type = 'HER2'
    
    print("\n1. HisToGene Pathway Training")
    print("-" * 30)
    
    # Create dataset
    dataset = ViT_HER2ST_Pathway(
        train=True, 
        fold=fold,
        pathway=pathway,
        score_method=score_method
    )
    train_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
    
    # Create model for single pathway prediction
    model = HisToGene(n_layers=8, n_genes=1, learning_rate=1e-5)  # n_genes=1 for pathway
    
    # Setup trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=100
    )
    
    # Train
    trainer.fit(model, train_loader)
    
    # Save model
    tag = f'-htg_{pathway}_{score_method}_cv'
    model_path = f"model/last_train_{tag}_{fold}.ckpt"
    trainer.save_checkpoint(model_path)
    print(f"Model saved to: {model_path}")
    
    print("\n2. HisToGene Pathway Prediction")
    print("-" * 30)
    
    # Load trained model
    model = HisToGene.load_from_checkpoint(
        model_path,
        n_layers=8, 
        n_genes=1,  # Single pathway
        learning_rate=1e-5
    )
    
    # Create test dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = ViT_HER2ST_Pathway(
        train=False,
        sr=False,
        fold=fold,
        pathway=pathway,
        score_method=score_method
    )
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4)
    
    # Make predictions
    adata_pred, adata_truth = model_predict_pathway(
        model, test_loader, attention=False, device=device
    )
    
    # Set pathway name
    adata_pred.var_names = [pathway]
    adata_truth.var_names = [pathway]
    
    # Process results
    adata_pred = comp_tsne_km(adata_pred, 4)
    
    print(f"Pathway prediction completed!")
    print(f"Predicted data shape: {adata_pred.shape}")
    print(f"Ground truth shape: {adata_truth.shape}")
    
    # Calculate correlation
    from scipy.stats import pearsonr
    r, p = pearsonr(adata_pred.X[:, 0], adata_truth.X[:, 0])
    print(f"Pathway correlation: {r:.4f} (p={p:.4e})")
    
    return adata_pred, adata_truth


def main():
    """
    Main training function
    """
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"Starting HisToGene pathway training with parameters:")
    print(f"  Fold: {args.fold}")
    print(f"  Pathway: {args.pathway}")
    print(f"  Score Method: {args.score}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Device: {args.device}")

    # Train the model
    model_path = train_pathway(
        fold=args.fold,
        pathway=args.pathway,
        score_method=args.score,
        dataset_type=args.dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device
    )

    print(f"Training completed! Model saved to: {model_path}")


if __name__ == '__main__':
    # Check if running with arguments (command line) or as tutorial
    import sys
    
    if len(sys.argv) == 1:
        # No arguments provided, run tutorial
        print("No arguments provided. Running tutorial example...")
        tutorial_example()
    else:
        # Arguments provided, run main training
        main()
