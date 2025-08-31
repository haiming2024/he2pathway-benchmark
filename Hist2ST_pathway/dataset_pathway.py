# Based on Hist2ST: https://github.com/biomed-AI/Hist2ST
import os
import glob
import torch
import torchvision
import numpy as np
import scanpy as sc
import pandas as pd 
import scprep as scp
import anndata as ad
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import ImageFile, Image
from utils import read_tiff, get_data
from graph_construction import calcADJ
from collections import defaultdict as dfd
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class ViT_HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self, train=True, fold=0, r=4, flatten=True, ori=False, adj=False, prune='Grid', neighs=4,
                 pathway=None, score_method=None):  
        super(ViT_HER2ST, self).__init__()
        
        if pathway is None or score_method is None:
            raise ValueError("Both 'pathway' and 'score_method' must be specified")
        
        self.pathway = pathway  
        self.score_method = score_method  
        
        self.cnt_dir = f'../data_pathway/{score_method}/{pathway}'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.pos_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.r = 224//r

        self.pathway_list = ['score']  

        names = ['A1','A2','A3','A4','A5','A6','B1','B2','B3','B4','B5','B6','C1','C2','C3','C4','C5','C6','D1','D2','D3','D4','D5','D6','E1','E2','E3','F1','F2','F3','G1','G2','G3','H1','H2','H3']
        names.sort()
        print(f"Found samples: {names}")
        print(f"Training pathway: {pathway} using {score_method} scores")  

        self.train = train
        self.ori = ori
        self.adj = adj
        samples = names[0:36]

        te_names = [samples[fold]]
        print(te_names)
        tr_names = list(set(samples)-set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}
        self.label={i:None for i in self.names}
        self.lbl2id={
            'invasive cancer':0, 'breast glands':1, 'immune infiltrate':2, 
            'cancer in situ':3, 'connective tissue':4, 'adipose tissue':5, 'undetermined':-1
        }
        if not train and self.names[0] in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
            self.lbl_dict={i:self.get_lbl(i) for i in self.names}
            idx=self.meta_dict[self.names[0]].index
            lbl=self.lbl_dict[self.names[0]]
            lbl=lbl.loc[idx,:]['label'].values
            self.label[self.names[0]]=lbl
        elif train:
            for i in self.names:
                idx=self.meta_dict[i].index
                if i in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
                    lbl=self.get_lbl(i)
                    lbl=lbl.loc[idx,:]['label'].values
                    lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                    self.label[i]=lbl
                else:
                    self.label[i]=torch.full((len(idx),),-1)
        
        self.pathway_set = self.pathway_list
        print(f'Loading {pathway} pathway enrichment data using {score_method}...')  
        
        print('Loading raw pathway scores (no normalization)...')
        self.exp_dict = {}
        
        for sample_name in self.names:
            df = self.get_cnt(sample_name)
            original_data = df.values  
            
            self.exp_dict[sample_name] = original_data
            
            print(f"{sample_name} {pathway} data:")
            print(f"Shape: {original_data.shape}")
            print(f"Range: [{original_data.min():.3f}, {original_data.max():.3f}]")
            print(f"Mean: {original_data.mean():.3f}, Std: {original_data.std():.3f}")
            
            negative_count = (original_data < 0).sum()
            if negative_count > 0:
                print(f"Warning: {negative_count} negative values found!")
            
            if original_data.max() > 1.5:
                print(f"Note: Max value > 1.5, may not be standard AUCell/UCell scores")
        
        print(f"{pathway} pathway data loading completed using {score_method}!") 
        
        self.center_dict = {
            i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) 
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i:calcADJ(m,neighs,pruneTag=prune)
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.flatten=flatten
        
    def __getitem__(self, index):
        ID = self.id2name[index]
        im = self.img_dict[ID]
        im = im.permute(1,0,2)
        
        exps = self.exp_dict[ID]  
        
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        label = self.label[ID]
        exps = torch.Tensor(exps)
        
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)
            self.patch_dict[ID]=patches
        
        data=[patches, positions, exps]
        if self.adj:
            data.append(adj)
        data.append(torch.Tensor(centers))
        return data
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        
        filename = f'{name}.csv'
        
        path = os.path.join(self.cnt_dir, filename)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Score file not found: {path}")
        
        df = pd.read_csv(path, index_col=0)
        
        print(f"Loading {self.pathway} {self.score_method} scores for {name}")
        print(f"File: {filename}")
        print(f"Columns: {list(df.columns)}")
        
        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        return meta

    def get_lbl(self,name):
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id',inplace=True)
        return df
