# Based on THItoGene https://github.com/yrjia1015/THItoGene/tree/main

import glob
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scprep as scp
import torch
import torchvision.transforms as transforms
from PIL import ImageFile, Image

from graph_construction import calcADJ

Image.MAX_IMAGE_PIXELS = 2300000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ViT_HER2ST_Pathway(torch.utils.data.Dataset):

    def __init__(self, train=True, ds=None, sr=False, fold=0,
                 pathway=None, score_method=None):  
        super(ViT_HER2ST_Pathway, self).__init__()
        
        # Single-pathway mode: must specify both pathway and score_method
        if pathway is None or score_method is None:
            raise ValueError("Both 'pathway' and 'score_method' must be specified")
        
        self.cnt_dir = f'../data_pathway/{score_method}/{pathway}'
        self.pathway = pathway
        self.score_method = score_method
        
        self.img_dir = r'./data/her2st/data/ST-imgs'
        self.pos_dir = r'./data/her2st/data/ST-spotfiles'
        self.lbl_dir = r'./data/her2st/data/ST-pat/lbl'
        self.r = 224 // 4

        print(f"Training pathway: {pathway} using {score_method}")

        # Only one pathway is used
        self.pathway_list = [pathway]

        # Load sample names from CSV files (e.g., A1.csv)
        names = os.listdir(self.cnt_dir)
        names = [i.replace('.csv', '') for i in names if i.endswith('.csv')]
        # Exclude specific samples
        excluded_samples = ['A1', 'H1', 'H2', 'H3']
        names = [name for name in names if name not in excluded_samples]
        names.sort()
        print(f"Excluded samples: {excluded_samples}")        
        print(f"Found samples: {names}")

        self.train = train
        self.sr = sr

        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples) - set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print('Loading images...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in self.names}

        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in self.names}

        self.label = {i: None for i in self.names}

        self.lbl2id = {
            'invasive cancer': 0,
            'breast glands': 1,
            'immune infiltrate': 2,
            'cancer in situ': 3,
            'connective tissue': 4,
            'adipose tissue': 5,
            'undetermined': -1
        }

        # Process label data if available
        if not train and self.names[0] in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1', 'J1']:
            self.lbl_dict = {i: self.get_lbl(i) for i in self.names}
            idx = self.meta_dict[self.names[0]].index
            lbl = self.lbl_dict[self.names[0]]
            lbl = lbl.loc[idx, :]['label'].values
            self.label[self.names[0]] = lbl
        elif train:
            for i in self.names:
                idx = self.meta_dict[i].index
                if i in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1', 'J1']:
                    lbl = self.get_lbl(i)
                    lbl = lbl.loc[idx, :]['label'].values
                    lbl = torch.Tensor(list(map(lambda i: self.lbl2id[i], lbl)))
                    self.label[i] = lbl
                else:
                    self.label[i] = torch.full((len(idx),), -1)

        # Single-pathway data processing
        self.pathway_set = [pathway]
        print(f'Loading pathway data: {pathway} using {score_method}...')
        
        # Pathway data uses AUCell/UCell scores directly (no log transform)
        self.exp_dict = {i: m[self.pathway_set].values for i, m in self.meta_dict.items()}
        
        # Show statistics for the first two samples
        for sample_name in list(self.exp_dict.keys())[:2]:
            data = self.exp_dict[sample_name]
            print(f"{sample_name} pathway data:")
            print(f"    Shape: {data.shape}")
            print(f"    Range: [{data.min():.3f}, {data.max():.3f}]")
            print(f"    Pathway: {pathway}, Score method: {score_method}")
        
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in self.meta_dict.items()}
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}
        self.patch_dict = {}
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])
        self.adj_dict = {i: calcADJ(coord=m, k=4, pruneTag='NA') for i, m in self.loc_dict.items()}

    def filter_helper(self):
        # Only one pathway
        a = np.zeros(1)
        n = 0
        for i, exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp > 0] = 1
            a[0] += np.sum(exp[:, 0])

    def __getitem__(self, index):
        i = index
        name = self.id2name[i]
        im = self.img_dict[self.id2name[i]]
        im = im.permute(1, 0, 2)
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        label = self.label[self.id2name[i]]
        if self.id2name[i] in self.patch_dict:
            patches = self.patch_dict[self.id2name[i]]
        else:
            patches = None

        adj = self.adj_dict[name]
        patch_dim = 3 * self.r * self.r * 4

        if self.sr:
            centers = torch.LongTensor(centers)
            max_x = centers[:, 0].max().item()
            max_y = centers[:, 1].max().item()
            min_x = centers[:, 0].min().item()
            min_y = centers[:, 1].min().item()
            r_x = (max_x - min_x) // 30
            r_y = (max_y - min_y) // 30
            centers = torch.LongTensor([min_x, min_y]).view(1, -1)
            positions = torch.LongTensor([0, 0]).view(1, -1)
            x = min_x
            y = min_y
            while y < max_y:
                x = min_x
                while x < max_x:
                    centers = torch.cat((centers, torch.LongTensor([x, y]).view(1, -1)), dim=0)
                    positions = torch.cat((positions, torch.LongTensor([x // r_x, y // r_y]).view(1, -1)), dim=0)
                    x += 56
                y += 56
            centers = centers[1:, :]
            positions = positions[1:, :]
            n_patches = len(centers)
            patches = torch.zeros((n_patches, patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                patches[i] = patch.flatten()
            return patches, positions, centers
        else:
            n_patches = len(centers)
            exps = torch.Tensor(exps).float()
            if patches is None:
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r)).float()
                for i in range(n_patches):
                    center = centers[i]
                    x, y = center
                    patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                    patches[i] = patch.permute(2, 0, 1)
                self.patch_dict[name] = patches
            adj = torch.Tensor(adj).float()
            if self.train:
                return patches.float(), positions, exps.float(), adj.float()
            else:
                return patches.float(), positions, exps.float(), torch.Tensor(centers).float(), adj.float()

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name
        fig_name = os.listdir(pre)[0]
        path = pre + '/' + fig_name
        print(path)
        im = Image.open(path)
        return im

    # Only support single-pathway file format (A1.csv)
    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '.csv'
        if not os.path.exists(path):
            raise FileNotFoundError(f"Score file not found: {path}")
        df = pd.read_csv(path, index_col=0)
        # Ensure column name is set to pathway
        if df.shape[1] == 1:
            df.columns = [self.pathway]
        print(f"Loading {self.pathway} {self.score_method} scores for {name}")
        return df

    def get_pos(self, name):
        path = self.pos_dir + '/' + name + '_selection.tsv'
        df = pd.read_csv(path, sep='\t')
        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        return df

    def get_lbl(self, name):
        path = self.lbl_dir + '/' + name + '_labeled_coordinates.tsv'
        df = pd.read_csv(path, sep='\t')
        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i]) + 'x' + str(y[i]))
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id', inplace=True)
        return df

    def get_meta(self, name, pathway_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        return meta
