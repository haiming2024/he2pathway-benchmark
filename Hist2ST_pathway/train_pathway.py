# Based on Hist2ST: https://github.com/biomed-AI/Hist2ST
import torch
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as tf
from tqdm import tqdm
from predict_pathway import *
from HIST2ST_pathway import *
from dataset_pathway import ViT_HER2ST
from scipy.stats import pearsonr,spearmanr
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from copy import deepcopy as dcp
from collections import defaultdict as dfd
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import argparse
import os

parser = argparse.ArgumentParser(description='Single pathway cross-validation training for Hist2ST')
parser.add_argument('--fold', type=int, required=True, 
                    help='Fold number for cross-validation (which image to use as test set)')
parser.add_argument('--pathway', type=str, required=True,
                    choices=['GO_0034340', 'GO_0060337', 'GO_0071357',
                             'KEGG_P53_SIGNALING_PATHWAY', 'KEGG_BREAST_CANCER', 'KEGG_APOPTOSIS',
                             'BREAST_CANCER_LUMINAL_A', 'HER2_AMPLICON_SIGNATURE', 'HALLMARK_MYC_TARGETS_V1'],
                    help='Pathway to train on')
parser.add_argument('--score', type=str, required=True, choices=['AUCell', 'UCell'],
                    help='Scoring method to use (AUCell or UCell)')
parser.add_argument('--data', type=str, default='her2st', choices=['her2st', 'cscc'],
                    help='Dataset to use')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of training epochs')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device to use for training')

args = parser.parse_args()

random.seed(12000)
np.random.seed(12000)
torch.manual_seed(12000)
torch.cuda.manual_seed(12000)
torch.cuda.manual_seed_all(12000)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

name=[*[f'A{i}' for i in range(1,7)],*[f'B{i}' for i in range(1,7)],
      *[f'C{i}' for i in range(1,7)],*[f'D{i}' for i in range(1,7)],
      *[f'E{i}' for i in range(1,4)],*[f'F{i}' for i in range(1,4)],*[f'G{i}' for i in range(1,4)]]
patients = ['P2', 'P5', 'P9', 'P10']
reps = ['rep1', 'rep2', 'rep3']
skinname = []
for i in patients:
    for j in reps:
        skinname.append(i+'_ST_'+j)

tag='5-7-2-8-4-16-32'
k,p,d1,d2,d3,h,c=map(lambda x:int(x),tag.split('-'))
dropout=0.2

fold = args.fold
pathway = args.pathway  
score_method = args.score  
data = args.data
device = args.device
prune = 'Grid' if data=='her2st' else 'NA'
genes = 171 if data=='cscc' else 785

print(f"Starting single pathway cross-validation training:")
print(f"  Fold: {fold}")
print(f"  Pathway: {pathway}")  
print(f"  Score Method: {score_method}")  
print(f"  Dataset: {data}")
print(f"  Test image: {name[fold] if fold < len(name) else 'Unknown'}")
print(f"  Device: {device}")
print(f"  Epochs: {args.epochs}")

print("Loading training data...")
trainset = pk_load(fold, 'train', dataset=data, flatten=False, adj=True, ori=True, prune=prune,
                   pathway=pathway, score_method=score_method)  
train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)

print(f"Training set size: {len(trainset)}")

model = Hist2ST(
    depth1=d1, depth2=d2, depth3=d3, n_pathways=1,  
    kernel_size=k, patch_size=p,
    heads=h, channel=c, dropout=dropout,
    zinb=0.25, nb=False,
    bake=5, lamb=0.5, 
)

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

logger = None
trainer = pl.Trainer(
    devices=1, accelerator='gpu',
    max_epochs=args.epochs,
    logger=logger,
)

print("Starting training...")
trainer.fit(model, train_loader)

model_dir = f"./model/{score_method}/{pathway}/"  
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = f"{model_dir}fold_{fold:02d}_{pathway}_{score_method}.ckpt"  
torch.save(model.state_dict(), model_path)

print(f"Training completed!")
print(f"Model saved to: {model_path}")
print(f"Test image for this fold: {name[fold] if fold < len(name) else 'Unknown'}")

del model
del trainer
del trainset
del train_loader
torch.cuda.empty_cache()

print("Memory cleared. Training job finished.")
