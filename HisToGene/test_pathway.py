# Based on HisToGene https://github.com/maxpmx/HisToGene/tree/main

# coding:utf-8 
"""
HisToGene Pathway Prediction Testing Script
Test script for pathway prediction models trained with the modified HisToGene pipeline.
Similar to the original gene prediction testing but adapted for single pathway outputs.
"""

import random
import os
import argparse
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dataset_pathway import ViT_HER2ST_Pathway, SKIN_Pathway
from predict_pathway import model_predict_pathway, get_pathway_correlation
from utils import *
from vis_model import HisToGene

# NEW: Command line argument support for pathway testing

parser = argparse.ArgumentParser(description='HisToGene pathway prediction testing')
parser.add_argument('--fold', type=int, required=True, 
                    help='Fold number for testing (which sample to test)')
parser.add_argument('--pathway', type=str, required=True,
                    choices=['GO_0034340', 'GO_0060337', 'GO_0071357',
                             'KEGG_P53_SIGNALING_PATHWAY', 'KEGG_BREAST_CANCER', 'KEGG_APOPTOSIS',
                             'BREAST_CANCER_LUMINAL_A', 'HER2_AMPLICON_SIGNATURE', 'HALLMARK_MYC_TARGETS_V1'],
                    help='Pathway to test')
parser.add_argument('--score', type=str, required=True, choices=['AUCell', 'UCell'],
                    help='Scoring method used for training (AUCell or UCell)')
parser.add_argument('--dataset', type=str, default='HER2', choices=['HER2', 'Skin'],
                    help='Dataset to test on (HER2 or Skin)')
parser.add_argument('--device', type=str, default='auto',
                    help='Device to use for testing (auto/cpu/cuda)')
parser.add_argument('--sr', action='store_true',
                    help='Use super-resolution prediction')
parser.add_argument('--save_results', action='store_true', default=True,
                    help='Save prediction results to file')

args = parser.parse_args()


def test_pathway(test_sample_ID, model_address, pathway, score_method, 
                dataset_type='HER2', device='auto', sr=False, save_results=True):
    """
          Test single pathway prediction model
    
    Args:
        test_sample_ID: Fold number for testing
        model_address: Base directory where models are saved
        pathway: Pathway name to test
        score_method: Scoring method (AUCell or UCell)
        dataset_type: Dataset type (HER2 or Skin)
        device: Device to use for testing
        sr: Whether to use super-resolution
        save_results: Whether to save results to files
    """
    
    print(f"Testing HisToGene pathway prediction:")
    print(f"Pathway: {pathway}")
    print(f"Score Method: {score_method}")
    print(f"Dataset: {dataset_type}")
    print(f"Fold: {test_sample_ID}")
    print(f"Super-resolution: {sr}")
    
    # Build model file path

    tag = f"-htg_{pathway}_{score_method}_cv"
    
    # Try different possible model path structures
    possible_paths = [
        f"{model_address}/pathway/{score_method}/{pathway}/last_train_{tag}_{test_sample_ID}.ckpt",
        f"{model_address}/{score_method}/{pathway}/fold_{test_sample_ID:02d}_{pathway}_{score_method}.ckpt",
        f"{model_address}/last_train_{tag}_{test_sample_ID}.ckpt",
        f"model/last_train_{tag}_{test_sample_ID}.ckpt"
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print(f"Model file not found. Tried paths:")
        for path in possible_paths:
            print(f"{path}")
        print("Please make sure you have trained the model first.")
        return None
    
    print(f"Loading model from: {model_path}")
    
    # Load pre-trained model (key change: n_genes=1 for pathway)

    try:
        model = HisToGene.load_from_checkpoint(
            model_path,
            n_genes=1,  # Single pathway output
            learning_rate=1e-5,
            n_layers=8  # Default value, adjust if needed
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying with different parameters...")
        try:
            model = HisToGene.load_from_checkpoint(model_path)
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            return None

    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # Create test dataset

    try:
        if dataset_type == 'HER2':
            dataset = ViT_HER2ST_Pathway(
                train=False, 
                sr=sr, 
                fold=test_sample_ID,
                pathway=pathway,
                score_method=score_method
            )
        elif dataset_type == 'Skin':
            dataset = SKIN_Pathway(
                train=False,
                sr=sr,
                fold=test_sample_ID,
                pathway=pathway,
                score_method=score_method
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
            
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return None

    test_loader = DataLoader(dataset, batch_size=1, num_workers=0)
    test_sample_name = dataset.names[0] if hasattr(dataset, 'names') else f"fold_{test_sample_ID}"

    print(f"Test dataset: {len(dataset)} samples")
    print("Running pathway prediction...")
    
    # Run prediction

    try:
        if sr:
            # Super-resolution prediction (if implemented)
            from predict_pathway import sr_predict_pathway
            adata_pred = sr_predict_pathway(model, test_loader, device=device)
            adata_truth = None  # SR doesn't have ground truth
        else:
            # Normal prediction
            adata_pred, adata_truth = model_predict_pathway(
                model, test_loader, attention=False, device=device
            )
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Set pathway name
    adata_pred.var_names = [pathway]
    if adata_truth is not None:
        adata_truth.var_names = [pathway]

    return adata_pred, adata_truth, test_sample_name


def analyze_results(pred, gt, pathway, score_method, fold, test_sample_name):
    """
    Analyze and display prediction results
    """
    print("="*60)
    print(f"PATHWAY PREDICTION RESULTS")
    print(f"Pathway: {pathway}")
    print(f"Score Method: {score_method}")
    print(f"Test Sample: {test_sample_name}")
    print(f"Fold: {fold}")
    print("="*60)
    
    print(f"Prediction shape: {pred.shape}")
    if gt is not None:
        print(f"Ground truth shape: {gt.shape}")
        
        # Calculate correlation
        try:
            R, p_val = get_pathway_correlation(pred, gt)
            test_pcc = R[0]
            
            print(f"{pathway:30} | PCC: {test_pcc:6.4f} | p-value: {p_val[0]:.2e} | n_spots: {len(pred)}")
        except Exception as e:
            print(f"Error calculating correlation: {e}")
            test_pcc, p_val = np.nan, [np.nan]
    else:
        print("No ground truth available (super-resolution mode)")
        test_pcc, p_val = np.nan, [np.nan]
    
    print("-"*60)
    
    # Data quality check
    pred_array = pred.X if hasattr(pred, 'X') else np.array(pred)
    
    if hasattr(pred_array, 'toarray'):
        pred_array = pred_array.toarray()
    
    print(f"DATA QUALITY CHECK:")
    print(f"  Prediction range: [{pred_array.min():.3f}, {pred_array.max():.3f}]")
    print(f"  Prediction mean: {pred_array.mean():.3f}")
    print(f"  Prediction std: {pred_array.std():.3f}")
    print(f"  Negative values (pred): {(pred_array < 0).sum()}/{pred_array.size} ({(pred_array < 0).mean()*100:.1f}%)")
    
    if gt is not None:
        gt_array = gt.X if hasattr(gt, 'X') else np.array(gt)
        if hasattr(gt_array, 'toarray'):
            gt_array = gt_array.toarray()
            
        print(f"  Ground truth range: [{gt_array.min():.3f}, {gt_array.max():.3f}]")
        print(f"  Ground truth mean: {gt_array.mean():.3f}")
        print(f"  Ground truth std: {gt_array.std():.3f}")
        print(f"  Negative values (gt): {(gt_array < 0).sum()}/{gt_array.size} ({(gt_array < 0).mean()*100:.1f}%)")
    
    return test_pcc, p_val


def save_test_results(pred, gt, pathway, score_method, fold, test_sample_name, 
                     test_pcc, p_val, dataset_type='HER2'):
    """
    Save test results to files
    """
    # Create results directory
    results_dir = f"./results_HisToGene_pathway/{dataset_type}/{score_method}/{pathway}/"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save summary results
    result_file = f"{results_dir}fold_{fold:02d}_{pathway}_{score_method}_results.txt"
    with open(result_file, 'w') as f:
        f.write(f"HisToGene Pathway Prediction Results\n")
        f.write(f"===================================\n")
        f.write(f"Pathway: {pathway}\n")
        f.write(f"Score Method: {score_method}\n")
        f.write(f"Dataset: {dataset_type}\n")
        f.write(f"Fold: {fold}\n")
        f.write(f"Test Sample: {test_sample_name}\n")
        f.write(f"PCC: {test_pcc:.6f}\n")
        f.write(f"P-value: {p_val[0]:.2e}\n")
        f.write(f"N_spots: {len(pred)}\n")
        
        # Data statistics
        pred_array = pred.X if hasattr(pred, 'X') else np.array(pred)
        if hasattr(pred_array, 'toarray'):
            pred_array = pred_array.toarray()
            
        f.write(f"\nPrediction Statistics:\n")
        f.write(f"  Range: [{pred_array.min():.3f}, {pred_array.max():.3f}]\n")
        f.write(f"  Mean: {pred_array.mean():.3f}\n")
        f.write(f"  Std: {pred_array.std():.3f}\n")
        
        if gt is not None:
            gt_array = gt.X if hasattr(gt, 'X') else np.array(gt)
            if hasattr(gt_array, 'toarray'):
                gt_array = gt_array.toarray()
                
            f.write(f"\nGround Truth Statistics:\n")
            f.write(f"  Range: [{gt_array.min():.3f}, {gt_array.max():.3f}]\n")
            f.write(f"  Mean: {gt_array.mean():.3f}\n")
            f.write(f"  Std: {gt_array.std():.3f}\n")
    
    print(f"\n Summary results saved to: {result_file}")
    
    # Save AnnData objects
    pred_file = f"{results_dir}fold_{fold:02d}_{pathway}_{score_method}_pred.h5ad"
    pred.write(pred_file)
    print(f"Predictions saved to: {pred_file}")
    
    if gt is not None:
        gt_file = f"{results_dir}fold_{fold:02d}_{pathway}_{score_method}_truth.h5ad"
        gt.write(gt_file)
        print(f"Ground truth saved to: {gt_file}")


def main():
    """
    Main testing function
    """
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Get parameters from command line
    fold = args.fold
    pathway = args.pathway
    score_method = args.score
    dataset_type = args.dataset
    device = args.device
    sr = args.sr
    save_results = args.save_results
    
    print(f"Starting HisToGene pathway testing:")
    print(f"  Fold: {fold}")
    print(f"  Pathway: {pathway}")
    print(f"  Score Method: {score_method}")
    print(f"  Dataset: {dataset_type}")
    print(f"  Device: {device}")
    print(f"  Super-resolution: {sr}")

    # Run pathway testing

    print("\nStarting pathway prediction testing...")
    try:
        result = test_pathway(
            fold, 
            "model", 
            pathway, 
            score_method,
            dataset_type=dataset_type,
            device=device,
            sr=sr,
            save_results=save_results
        )
        
        if result is not None:
            pred, gt, test_sample_name = result
            
            # Analyze results
            test_pcc, p_val = analyze_results(
                pred, gt, pathway, score_method, fold, test_sample_name
            )
            
            # Save results if requested
            if save_results:
                save_test_results(
                    pred, gt, pathway, score_method, fold, test_sample_name,
                    test_pcc, p_val, dataset_type
                )
            
            print("="*60)
            print(f"Testing completed successfully!")
            print(f"USAGE EXAMPLES:")
            print(f"  Test another fold: python test_pathway.py --fold {fold+1} --pathway {pathway} --score {score_method}")
            print(f"  Test different pathway: python test_pathway.py --fold {fold} --pathway GO_0060337 --score {score_method}")
            print(f"  Test different score: python test_pathway.py --fold {fold} --pathway {pathway} --score {'UCell' if score_method=='AUCell' else 'AUCell'}")
            print(f"  Test on skin dataset: python test_pathway.py --fold {fold} --pathway {pathway} --score {score_method} --dataset Skin")
            print(f"  Super-resolution test: python test_pathway.py --fold {fold} --pathway {pathway} --score {score_method} --sr")
        
    except Exception as e:
        print(f"Error in testing: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting job finished.")


def run_batch_testing():
    """
    Run batch testing for multiple pathways/folds
    """
    pathways = ['KEGG_P53_SIGNALING_PATHWAY', 'KEGG_BREAST_CANCER', 'KEGG_APOPTOSIS']
    score_methods = ['AUCell', 'UCell']
    folds = [0, 1, 2]
    
    results = []
    
    for pathway in pathways:
        for score_method in score_methods:
            for fold in folds:
                print(f"\n{'='*60}")
                print(f"Testing: {pathway} | {score_method} | Fold {fold}")
                print(f"{'='*60}")
                
                try:
                    result = test_pathway(
                        fold, "model", pathway, score_method,
                        dataset_type='HER2', device='auto'
                    )
                    
                    if result is not None:
                        pred, gt, test_sample_name = result
                        test_pcc, p_val = analyze_results(
                            pred, gt, pathway, score_method, fold, test_sample_name
                        )
                        
                        results.append({
                            'pathway': pathway,
                            'score_method': score_method,
                            'fold': fold,
                            'pcc': test_pcc,
                            'p_value': p_val[0] if p_val is not None else np.nan,
                            'n_spots': len(pred)
                        })
                        
                except Exception as e:
                    print(f"Error testing {pathway} | {score_method} | Fold {fold}: {e}")
                    
    # Summary of batch results
    print(f"\n{'='*80}")
    print("BATCH TESTING SUMMARY")
    print(f"{'='*80}")
    
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        
        # Group by pathway and score method
        summary = df.groupby(['pathway', 'score_method']).agg({
            'pcc': ['mean', 'std', 'count'],
            'p_value': 'mean',
            'n_spots': 'mean'
        }).round(4)
        
        print(summary)
        
        # Save batch results
        df.to_csv('batch_testing_results.csv', index=False)
        print(f"\n Batch results saved to: batch_testing_results.csv")
    else:
        print("No successful tests completed.")


if __name__ == '__main__':
    import sys
    
    # Check if running in batch mode
    if '--batch' in sys.argv:
        run_batch_testing()
    else:
        main()
