# Based on THItoGene https://github.com/yrjia1015/THItoGene/tree/main

import warnings

import torch
from tqdm import tqdm

from utils import *

warnings.filterwarnings('ignore')

MODEL_PATH = ''


def model_predict(model, test_loader, adata=None, attention=True, device=torch.device('cpu')):
    """
    Fixed version: Correct implementation based on reference code
    Key improvements:
    1. Use squeeze(0) instead of squeeze()
    2. Correctly handle batch dimension
    3. Preserve 2D data structure
    """
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_gt = []
    all_ct = []

    with torch.no_grad():
        for patch, position, exp, center, adj in tqdm(test_loader):
            patch, position, adj = patch.to(device), position.to(device), adj.to(device)
            pred = model(patch, position, adj)

            # Remove only batch dimension, keep feature dimension
            preds_batch = pred.squeeze(0).cpu().numpy()    # (n_spots, n_pathways) 
            gt_batch = exp.squeeze(0).cpu().numpy()        # (n_spots, n_pathways)
            ct_batch = center.squeeze(0).cpu().numpy()     # (n_spots, 2)
            
            all_preds.append(preds_batch)
            all_gt.append(gt_batch)
            all_ct.append(ct_batch)

    if len(all_preds) > 1:
        preds_final = np.concatenate(all_preds, axis=0)
        gt_final = np.concatenate(all_gt, axis=0)
        ct_final = np.concatenate(all_ct, axis=0)
    else:
        preds_final = all_preds[0]
        gt_final = all_gt[0]
        ct_final = all_ct[0]

    print(f"Final data shapes:")
    print(f"  Predictions: {preds_final.shape}")
    print(f"  Ground truth: {gt_final.shape}")
    print(f"  Coordinates: {ct_final.shape}")

    adata = ann.AnnData(preds_final)
    adata.obsm['spatial'] = ct_final

    adata_gt = ann.AnnData(gt_final)
    adata_gt.obsm['spatial'] = ct_final

    return adata, adata_gt


def model_predict_alternative(model, test_loader, adata=None, attention=True, device=torch.device('cpu')):
    """
    Alternative version: closer to reference code
    Suitable for single-sample case
    """
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        # Assume only one batch (reference code assumption)
        for patch, position, exp, center, adj in tqdm(test_loader):
            patch, position, adj = patch.to(device), position.to(device), adj.to(device)
            pred = model(patch, position, adj)
            
            preds = pred.squeeze(0).cpu().numpy()    # (n_spots, 1)
            gt = exp.squeeze(0).cpu().numpy()        # (n_spots, 1) 
            ct = center.squeeze(0).cpu().numpy()     # (n_spots, 2)
            break

    print(f"Alternative method shapes:")
    print(f"  Predictions: {preds.shape}")
    print(f"  Ground truth: {gt.shape}")
    print(f"  Coordinates: {ct.shape}")

    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct

    adata_gt = ann.AnnData(gt)
    adata_gt.obsm['spatial'] = ct

    return adata, adata_gt


def sr_predict(model, test_loader, device=torch.device('cpu')):
    """
    Super-resolution prediction function
    """
    model.eval()
    model = model.to(device)
    preds = None
    with torch.no_grad():
        for patch, position, center in tqdm(test_loader):
            patch, position = patch.to(device), position.to(device)
            pred = model(patch, position)

            if preds is None:
                preds = pred.squeeze(0) if pred.shape[0] == 1 else pred
                ct = center.squeeze(0) if center.shape[0] == 1 else center
            else:
                pred_to_cat = pred.squeeze(0) if pred.shape[0] == 1 else pred
                center_to_cat = center.squeeze(0) if center.shape[0] == 1 else center
                preds = torch.cat((preds, pred_to_cat), dim=0)
                ct = torch.cat((ct, center_to_cat), dim=0)
                
    preds = preds.cpu().numpy()
    ct = ct.cpu().numpy()
    
    # Ensure predictions are 2D
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
        
    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct

    return adata


def get_R(data1, data2, dim=1, func=None):
    """
    Calculate Pearson correlation between two matrices
    Adapted from reference code
    """
    from scipy.stats import pearsonr
    if func is None:
        func = pearsonr
        
    adata1 = data1.X
    adata2 = data2.X
    r1, p1 = [], []
    
    for g in range(data1.shape[dim]):
        if dim == 1:
            r, pv = func(adata1[:, g], adata2[:, g])
        elif dim == 0:
            r, pv = func(adata1[g, :], adata2[g, :])
        r1.append(r)
        p1.append(pv)
        
    r1 = np.array(r1)
    p1 = np.array(p1)
    return r1, p1
