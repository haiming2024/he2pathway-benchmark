# Based on THItoGene https://github.com/yrjia1015/THItoGene/tree/main

# coding:utf-8 
import random
import os
import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from dataset_pathway import ViT_HER2ST_Pathway
from utils import *
from vis_model import THItoGene

# Command line argument parser
parser = argparse.ArgumentParser(description='Single pathway cross-validation training for THItoGene')
parser.add_argument('--fold', type=int, required=True, 
                    help='Fold number for cross-validation (which image to use as test set)')
parser.add_argument('--pathway', type=str, required=True,
                    choices=['GO_0034340', 'GO_0060337', 'GO_0071357',
                             'KEGG_P53_SIGNALING_PATHWAY', 'KEGG_BREAST_CANCER', 'KEGG_APOPTOSIS',
                             'BREAST_CANCER_LUMINAL_A', 'HER2_AMPLICON_SIGNATURE', 'HALLMARK_MYC_TARGETS_V1'],
                    help='Pathway to train on')
parser.add_argument('--score', type=str, required=True, choices=['AUCell', 'UCell'],
                    help='Scoring method to use (AUCell or UCell)')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of training epochs')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device to use for training')

args = parser.parse_args()


def train(test_sample_ID, epochs, modelsave_address, pathway, score_method):
    """Simplified: single pathway training"""

    print(f"Training single pathway: {pathway} using {score_method}")
    print("Output dimension: 1 (single pathway)")

    # Create dataset
    dataset = ViT_HER2ST_Pathway(
        train=True, 
        fold=test_sample_ID,
        pathway=pathway,
        score_method=score_method
    )
    train_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

    # Model tag
    tagname = f"-htg_{pathway}_{score_method}_cv"

    # Initialize model (output dimension = 1 for single pathway)
    model = THItoGene(n_genes=1, learning_rate=1e-5, route_dim=64, caps=20, heads=[16, 8], n_layers=4)

    # Logger
    log_name = f"log_{tagname}_{test_sample_ID}"
    mylogger = CSVLogger(save_dir=modelsave_address + "/../logs/", name=log_name)

    # Trainer
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")
        trainer = pl.Trainer(
            accelerator="gpu", 
            devices=1,
            max_epochs=epochs, 
            logger=mylogger,
            enable_progress_bar=True,
            log_every_n_steps=10
        )
    else:
        print("No GPU available, using CPU")
        trainer = pl.Trainer(
            accelerator="cpu", 
            max_epochs=epochs, 
            logger=mylogger,
            enable_progress_bar=True,
            log_every_n_steps=10
        )

    # Training
    print("Starting training...")
    trainer.fit(model, train_loader)

    # Save model
    model_save_dir = f"{modelsave_address}/{score_method}/{pathway}/"
    os.makedirs(model_save_dir, exist_ok=True)
    model_filename = f"fold_{test_sample_ID:02d}_{pathway}_{score_method}.ckpt"
    model_path = model_save_dir + model_filename
    
    trainer.save_checkpoint(model_path)
    print(f"Model saved to: {model_path}")
    print("Training completed.")


if __name__ == '__main__':
    # Set random seeds
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    fold = args.fold
    pathway = args.pathway
    score_method = args.score
    epochs = args.epochs
    device = args.device
    
    print("Starting single pathway training:")
    print(f"  Fold: {fold}")
    print(f"  Pathway: {pathway}")
    print(f"  Score Method: {score_method}")
    print(f"  Epochs: {epochs}")
    print(f"  Device: {device}")

    train(fold, epochs, "model_pathway_THItoGene", pathway, score_method)

    print("Training job finished.")
