# Based on THItoGene https://github.com/yrjia1015/THItoGene/tree/main

# coding:utf-8 
import random
import os
import argparse

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dataset_pathway import ViT_HER2ST_Pathway
from predict_pathway import model_predict
from utils import *
from vis_model import THItoGene

# Command line argument parser
parser = argparse.ArgumentParser(description='Single pathway testing for THItoGene')
parser.add_argument('--fold', type=int, required=True, 
                    help='Fold number for testing (which image to test)')
parser.add_argument('--pathway', type=str, required=True,
                    choices=['GO_0034340', 'GO_0060337', 'GO_0071357',
                             'KEGG_P53_SIGNALING_PATHWAY', 'KEGG_BREAST_CANCER', 'KEGG_APOPTOSIS',
                             'BREAST_CANCER_LUMINAL_A', 'HER2_AMPLICON_SIGNATURE', 'HALLMARK_MYC_TARGETS_V1'],
                    help='Pathway to test')
parser.add_argument('--score', type=str, required=True, choices=['AUCell', 'UCell'],
                    help='Scoring method used for training (AUCell or UCell)')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device to use for testing')

args = parser.parse_args()


def test(test_sample_ID, model_address, pathway, score_method):
    """Simplified: single pathway testing only"""
    
    print(f"Testing single pathway: {pathway} using {score_method}")
    
    # Build model file path
    model_path = f"{model_address}/{score_method}/{pathway}/fold_{test_sample_ID:02d}_{pathway}_{score_method}.ckpt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please make sure you have trained the model first.")
        return None
    
    print(f"Loading model from: {model_path}")
    
    # Load pretrained model
    model = THItoGene.load_from_checkpoint(
        model_path,
        n_genes=1,  # Single pathway
        learning_rate=1e-5, route_dim=64, caps=20, heads=[16, 8],
        n_layers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create test dataset
    dataset = ViT_HER2ST_Pathway(
        train=False, 
        sr=False, 
        fold=test_sample_ID,
        pathway=pathway,
        score_method=score_method
    )
    test_loader = DataLoader(dataset, batch_size=1, num_workers=0)

    print("Running inference...")
    adata_pred, adata_truth = model_predict(model, test_loader, attention=False, device=device)

    adata_pred.var_names = [pathway]

    return adata_pred, adata_truth, dataset.names[0]


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
    device = args.device
    
    print(f"Starting single pathway testing:")
    print(f"  Fold: {fold}")
    print(f"  Pathway: {pathway}")
    print(f"  Score Method: {score_method}")
    print(f"  Device: {device}")

    print("Starting pathway prediction testing...")
    try:
        result = test(fold, "model_pathway_THItoGene", pathway, score_method)
        
        if result is not None:
            pred, gt, test_sample_name = result
            
            # Compute correlation
            R, p_val = get_R(pred, gt)
            test_pcc = np.nanmean(R)
            
            print("-" * 60)
            print(f"PATHWAY TESTING RESULTS")
            print(f"Pathway: {pathway}")
            print(f"Score Method: {score_method}")
            print(f"Test Sample: {test_sample_name}")
            print(f"Fold: {fold}")
            print("-" * 60)
            
            print(f"Prediction shape: {pred.shape}")
            print(f"Ground truth shape: {gt.shape}")
            
            print(f"{pathway:30} | PCC: {test_pcc:6.4f} | p-value: {p_val[0]:.2e} | n_spots: {len(pred)}")
            print("-" * 60)
            
            # Data quality check
            pred_array = pred.X if hasattr(pred, 'X') else np.array(pred)
            gt_array = gt.X if hasattr(gt, 'X') else np.array(gt)
            
            if hasattr(pred_array, 'toarray'):
                pred_array = pred_array.toarray()
            if hasattr(gt_array, 'toarray'):
                gt_array = gt_array.toarray()
            
            print(f"DATA QUALITY CHECK:")
            print(f"  Prediction range: [{pred_array.min():.3f}, {pred_array.max():.3f}]")
            print(f"  Ground truth range: [{gt_array.min():.3f}, {gt_array.max():.3f}]")
            print(f"  Negative values (pred): {(pred_array < 0).sum()}/{pred_array.size} ({(pred_array < 0).mean()*100:.1f}%)")
            print(f"  Negative values (gt): {(gt_array < 0).sum()}/{gt_array.size} ({(gt_array < 0).mean()*100:.1f}%)")
            
            # Save results
            results_dir = f"./results_THItoGene/{score_method}/{pathway}/"
            os.makedirs(results_dir, exist_ok=True)
            
            result_file = f"{results_dir}fold_{fold:02d}_{pathway}_{score_method}_results.txt"
            with open(result_file, 'w') as f:
                f.write(f"Pathway: {pathway}\n")
                f.write(f"Score Method: {score_method}\n")
                f.write(f"Fold: {fold}\n")
                f.write(f"Test Sample: {test_sample_name}\n")
                f.write(f"PCC: {test_pcc:.6f}\n")
                f.write(f"P-value: {p_val[0]:.2e}\n")
                f.write(f"N_spots: {len(pred)}\n")
                
            print(f"\nResults saved to: {result_file}")
            print("Testing completed successfully.")
            print("Usage examples:")
            print(f"  Test another fold: python test_1_0817.py --fold {fold+1} --pathway {pathway} --score {score_method}")
            print(f"  Test different pathway: python test_1_0817.py --fold {fold} --pathway GO_0060337 --score {score_method}")
            print(f"  Test different score: python test_1_0817.py --fold {fold} --pathway {pathway} --score {'UCell' if score_method=='AUCell' else 'AUCell'}")
        
    except Exception as e:
        print(f"Error in testing: {e}")
        import traceback
        traceback.print_exc()

    print("Testing job finished.")
