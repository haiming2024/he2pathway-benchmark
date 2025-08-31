# Based on HisToGene https://github.com/maxpmx/HisToGene/tree/main
import torch
from torch.utils.data import DataLoader
from utils import *
from vis_model import HisToGene
import warnings
from dataset_pathway import ViT_HER2ST_Pathway, SKIN_Pathway
from tqdm import tqdm
import numpy as np
warnings.filterwarnings('ignore')

MODEL_PATH = ''

def model_predict_pathway(model, test_loader, adata=None, attention=True, device=torch.device('cpu')):
    """
           Modified for pathway prediction - single pathway output
    """
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_gt = []
    all_ct = []

    with torch.no_grad():
        for batch_data in tqdm(test_loader):
            if len(batch_data) == 4:  # training format
                patch, position, exp, center = batch_data
            else:  # test format  
                patch, position, exp = batch_data
                center = None

            patch, position = patch.to(device), position.to(device)
            pred = model(patch, position)

            # Key fix: pathway prediction outputs single value per spot
            if pred.dim() > 2:
                pred = pred.squeeze(0)  # Remove batch dimension
            if exp.dim() > 2:
                exp = exp.squeeze(0)
                
            preds_batch = pred.cpu().numpy()
            gt_batch = exp.cpu().numpy()
            
            if center is not None:
                if center.dim() > 2:
                    center = center.squeeze(0)
                ct_batch = center.cpu().numpy()
            else:
                # For training data without center coordinates
                ct_batch = np.zeros((preds_batch.shape[0], 2))
            
            all_preds.append(preds_batch)
            all_gt.append(gt_batch)
            all_ct.append(ct_batch)

    # Concatenate all batches
    if len(all_preds) > 1:
        preds_final = np.concatenate(all_preds, axis=0)
        gt_final = np.concatenate(all_gt, axis=0)
        ct_final = np.concatenate(all_ct, axis=0)
    else:
        preds_final = all_preds[0]
        gt_final = all_gt[0]
        ct_final = all_ct[0]

    # Ensure 2D shape for single pathway
    if preds_final.ndim == 1:
        preds_final = preds_final.reshape(-1, 1)
    if gt_final.ndim == 1:
        gt_final = gt_final.reshape(-1, 1)

    print(f"Final pathway prediction shapes:")
    print(f"Predictions: {preds_final.shape}")
    print(f"Ground truth: {gt_final.shape}")
    print(f"Coordinates: {ct_final.shape}")

    # Create AnnData objects
    adata_pred = ann.AnnData(preds_final)
    adata_pred.obsm['spatial'] = ct_final

    adata_gt = ann.AnnData(gt_final)
    adata_gt.obsm['spatial'] = ct_final

    return adata_pred, adata_gt


def sr_predict_pathway(model, test_loader, attention=True, device=torch.device('cpu')):
    """
          Super-resolution prediction for pathway scores
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
    
    # Ensure 2D shape for single pathway
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
        
    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct

    return adata


def get_pathway_correlation(data1, data2, func=None):
    """
    Calculate correlation for pathway predictions
    """
    from scipy.stats import pearsonr, spearmanr
    if func is None:
        func = pearsonr
        
    adata1 = data1.X
    adata2 = data2.X
    
    # For single pathway, calculate correlation across all spots
    if adata1.shape[1] == 1:
        r, pv = func(adata1[:, 0], adata2[:, 0])
        return np.array([r]), np.array([pv])
    else:
        # For multiple pathways
        r_list, p_list = [], []
        for i in range(adata1.shape[1]):
            r, pv = func(adata1[:, i], adata2[:, i])
            r_list.append(r)
            p_list.append(pv)
        return np.array(r_list), np.array(p_list)


def main_pathway():
    """
    Main function for pathway prediction pipeline
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Example parameters - modify as needed
    pathway = 'KEGG_P53_SIGNALING_PATHWAY'
    score_method = 'AUCell'
    dataset_type = 'HER2'  # or 'Skin'
    
    for fold in range(12):  # Cross-validation
        tag = f'-htg_{pathway}_{score_method}_cv'
        
        print(f'Loading model for fold {fold}...')
        model_path = f"model/last_train_{tag}_{fold}.ckpt"
        
        # Load model with single pathway output (n_genes=1)
        model = HisToGene.load_from_checkpoint(
            model_path,
            n_layers=8, 
            n_genes=1,  # Changed from 785 to 1 for single pathway
            learning_rate=1e-5
        )
        model = model.to(device)
        
        print('Loading data...')
        if dataset_type == 'HER2':
            dataset = ViT_HER2ST_Pathway(
                train=False,
                sr=False,
                fold=fold,
                pathway=pathway,
                score_method=score_method
            )
        else:  # Skin
            dataset = SKIN_Pathway(
                train=False,
                sr=False,
                fold=fold,
                pathway=pathway,
                score_method=score_method
            )

        test_loader = DataLoader(dataset, batch_size=1, num_workers=4)
        print('Making pathway prediction...')

        adata_pred, adata_gt = model_predict_pathway(
            model, test_loader, attention=False, device=device
        )

        # Set pathway name
        adata_pred.var_names = [pathway]
        adata_gt.var_names = [pathway]
        
        # Calculate correlation
        r, p = get_pathway_correlation(adata_pred, adata_gt)
        print(f'Pathway {pathway} correlation: {r[0]:.4f} (p={p[0]:.4e})')

        print('Saving files...')
        adata_pred = comp_tsne_km(adata_pred, 4)
        
        # Save results
        output_path = f'processed/pathway_pred_{dataset_type}_{pathway}_{score_method}_{fold}.h5ad'
        adata_pred.write(output_path)
        print(f'Saved to: {output_path}')


if __name__ == '__main__':
    main_pathway()
