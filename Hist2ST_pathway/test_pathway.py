# Based on Hist2ST: https://github.com/biomed-AI/Hist2ST
import torch
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from predict_pathway import *
from HIST2ST_pathway import *
from dataset_pathway import ViT_HER2ST
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
import argparse
import os
import random

parser = argparse.ArgumentParser(description='Single pathway testing for Hist2ST')
parser.add_argument('--fold', type=int, required=True, help='Fold number for testing (which image to test)')
parser.add_argument('--pathway', type=str, required=True,
                    choices=['GO_0034340', 'GO_0060337', 'GO_0071357',
                             'KEGG_P53_SIGNALING_PATHWAY', 'KEGG_BREAST_CANCER', 'KEGG_APOPTOSIS',
                             'BREAST_CANCER_LUMINAL_A', 'HER2_AMPLICON_SIGNATURE', 'HALLMARK_MYC_TARGETS_V1'])
parser.add_argument('--score', type=str, required=True, choices=['AUCell', 'UCell'],
                    help='Scoring method (AUCell or UCell)')
parser.add_argument('--data', type=str, default='her2st', choices=['her2st', 'cscc'],
                    help='Dataset to use')
parser.add_argument('--device', type=str, default='cuda', help='Device to use for testing')
args = parser.parse_args()

# Dataset fold names
name=[*[f'A{i}' for i in range(1,7)],*[f'B{i}' for i in range(1,7)],
      *[f'C{i}' for i in range(1,7)],*[f'D{i}' for i in range(1,7)],
      *[f'E{i}' for i in range(1,4)],*[f'F{i}' for i in range(1,4)],
      *[f'G{i}' for i in range(1,4)],*[f'H{i}' for i in range(1,4)]]

device = args.device
fold = args.fold
pathway = args.pathway
score_method = args.score
data = args.data

# Random seed
random.seed(12000)
np.random.seed(12000)
torch.manual_seed(12000)
torch.cuda.manual_seed_all(12000)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

prune = 'Grid' if data=='her2st' else 'NA'

print("Starting single pathway testing...")
print(f"  Fold: {fold}")
print(f"  Pathway: {pathway}")
print(f"  Score Method: {score_method}")
print(f"  Test image: {name[fold] if fold < len(name) else 'Unknown'}")
print(f"  Device: {device}")

# Load test data
testset = pk_load(fold, 'test', dataset=data, flatten=False, adj=True, ori=True, prune=prune,
                  pathway=pathway, score_method=score_method)
test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)

# Load model
model_path = f'./model/{score_method}/{pathway}/fold_{fold:02d}_{pathway}_{score_method}.ckpt'
if not os.path.exists(model_path):
    print(f"Error: Model not found: {model_path}")
    exit(1)

model = Hist2ST(
    depth1=2, depth2=8, depth3=4, n_pathways=1, 
    kernel_size=5, patch_size=7,
    heads=16, channel=32, dropout=0.2,
    zinb=0.25, nb=False,
    bake=5, lamb=0.5
)
model.load_state_dict(torch.load(model_path))
print("Model loaded.")

# Inference
print("Running inference...")
pred, gt = test(model, test_loader, device)

# Convert to numpy
pred_array = pred.X.toarray() if hasattr(pred, 'X') and hasattr(pred.X, 'toarray') else np.array(pred)
gt_array = gt.X.toarray() if hasattr(gt, 'X') and hasattr(gt.X, 'toarray') else np.array(gt)

# Compute correlation
if np.var(pred_array) > 1e-10 and np.var(gt_array) > 1e-10:
    correlation, p_value = pearsonr(pred_array.flatten(), gt_array.flatten())
    print(f"Results:")
    print(f"  Pathway: {pathway}")
    print(f"  Score Method: {score_method}")
    print(f"  Test Sample: {testset.names[0]}")
    print(f"  Fold: {fold}")
    print(f"  PCC: {correlation:.4f}")
    print(f"  p-value: {p_value:.2e}")
    print(f"  Spots: {len(pred_array)}")
else:
    print("No variation in data. Skipping correlation.")

# Save results
results_dir = f"./results/{score_method}/{pathway}/"
os.makedirs(results_dir, exist_ok=True)
result_file = f"{results_dir}fold_{fold:02d}_{pathway}_{score_method}_results.txt"
with open(result_file, 'w') as f:
    f.write(f"Pathway: {pathway}\n")
    f.write(f"Score Method: {score_method}\n")
    f.write(f"Fold: {fold}\n")
    f.write(f"Test Sample: {testset.names[0]}\n")
    if np.var(pred_array) > 1e-10 and np.var(gt_array) > 1e-10:
        f.write(f"PCC: {correlation:.6f}\n")
        f.write(f"P-value: {p_value:.2e}\n")
    f.write(f"N_spots: {len(pred_array)}\n")

print(f"Results saved to: {result_file}")
