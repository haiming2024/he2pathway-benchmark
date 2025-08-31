# Based on HisToGene https://github.com/maxpmx/HisToGene/tree/main
import glob
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scprep as scp
import torch
import torchvision.transforms as transforms
from PIL import ImageFile, Image
from utils import read_tiff

Image.MAX_IMAGE_PIXELS = 2300000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ViT_HER2ST_Pathway(torch.utils.data.Dataset):
    """HisToGene pathway prediction dataset - modified from gene prediction"""
    
    def __init__(self, train=True, ds=None, sr=False, fold=0,
                 pathway=None, score_method=None):
        super(ViT_HER2ST_Pathway, self).__init__()
        
        # MODIFIED: Single pathway mode, must specify pathway and scoring method

        if pathway is None or score_method is None:
            raise ValueError("Both 'pathway' and 'score_method' must be specified")
        
        self.cnt_dir = f'../data_pathway/{score_method}/{pathway}'
        self.pathway = pathway
        self.score_method = score_method
        
        self.img_dir = './data/her2st/data/ST-imgs'
        self.pos_dir = './data/her2st/data/ST-spotfiles'
        self.lbl_dir = './data/her2st/data/ST-pat/lbl'
        self.r = 224 // 4

        print(f"Training pathway: {pathway} using {score_method}")

        # SIMPLIFIED: Single pathway mode, pathway list has only one element
        self.pathway_list = [pathway]  # Only one pathway

        # SIMPLIFIED: Get sample names from CSV files (A1.csv format)

        names = os.listdir(self.cnt_dir)
        names = [i.replace('.csv', '') for i in names if i.endswith('.csv')]
        # NEW: Exclude specified samples
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

        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in self.names}

        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in self.names}

        # SIMPLIFIED: Single pathway data processing
        self.pathway_set = [pathway]  # Only one pathway
        print(f'Loading pathway data: {pathway} using {score_method}...')
        
        # Pathway data directly uses AUCell/UCell scores, no log transformation needed
        self.exp_dict = {i: m[self.pathway_set].values for i, m in self.meta_dict.items()}
        
        # Display data statistics
        for sample_name in list(self.exp_dict.keys())[:2]:  # Only show first 2 samples
            data = self.exp_dict[sample_name]
            print(f"{sample_name} pathway data:")
            print(f"Shape: {data.shape}")
            print(f"Range: [{data.min():.3f}, {data.max():.3f}]")
            print(f"Pathway: {pathway}, Score method: {score_method}")
        
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in
                            self.meta_dict.items()}
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        im = im.permute(1, 0, 2)
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
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
            patches = torch.zeros((n_patches, patch_dim))
            exps = torch.Tensor(exps)

            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
                patches[i] = patch.flatten()

            if self.train:
                return patches, positions, exps
            else:
                return patches, positions, exps, torch.Tensor(centers)

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        pre = self.img_dir + '/' + name[0] + '/' + name
        fig_name = os.listdir(pre)[0]
        path = pre + '/' + fig_name
        im = Image.open(path)
        return im

    # SIMPLIFIED: Only support single pathway file format (A1.csv)

    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '.csv'
        
        # Check if file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Score file not found: {path}")
        
        df = pd.read_csv(path, index_col=0)
        
        # For consistency, set column name to pathway name
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

    def get_meta(self, name, pathway_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        return meta


class SKIN_Pathway(torch.utils.data.Dataset):
    """SKIN pathway prediction dataset - modified from gene prediction"""
    
    def __init__(self, train=True, ds=None, sr=False, fold=0,
                 pathway=None, score_method=None):
        super(SKIN_Pathway, self).__init__()

        if pathway is None or score_method is None:
            raise ValueError("Both 'pathway' and 'score_method' must be specified")

        self.dir = '/ibex/scratch/pangm0a/spatial/data/GSE144240_RAW/'
        self.cnt_dir = f'./data/skin/data/{score_method}/{pathway}'
        self.pathway = pathway
        self.score_method = score_method
        self.r = 224 // 2

        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i+'_ST_'+j)

        self.pathway_list = [pathway]
        self.train = train
        self.sr = sr

        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples) - set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print('Loading imgs...')
        self.img_dict = {i: self.get_img(i) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in self.names}

        self.pathway_set = [pathway]
        self.exp_dict = {i: m[self.pathway_set].values for i, m in self.meta_dict.items()}
        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in self.meta_dict.items()}
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}

        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        i = 0
        while index >= self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i-1]
        
        exp = self.exp_dict[self.id2name[i]][idx]
        center = self.center_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]

        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)

        x, y = center
        patch = self.img_dict[self.id2name[i]].crop((x-self.r, y-self.r, x+self.r, y+self.r))
        if self.train:
            patch = self.transforms(patch)
        else:
            patch = transforms.ToTensor()(patch)

        if self.train:
            return patch, loc, exp
        else: 
            return patch, loc, exp, torch.Tensor(center)

    def __len__(self):
        return self.cumlen[-1]

    def get_img(self, name):
        path = glob.glob(self.dir+'*'+name+'.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self, name):
        path = self.cnt_dir + '/' + name + '.csv'
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Score file not found: {path}")
        
        df = pd.read_csv(path, index_col=0)
        
        if df.shape[1] == 1:
            df.columns = [self.pathway]
            
        return df

    def get_pos(self, name):
        path = glob.glob(self.dir+'*spot*'+name+'.tsv')[0]
        df = pd.read_csv(path, sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self, name, pathway_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'), how='inner')
        return meta


# For compatibility with original STDataset
class STDataset_Pathway(torch.utils.data.Dataset):
    """ST pathway prediction dataset - modified from gene prediction"""
    
    def __init__(self, adata, img_path, diameter=177.5, train=True, 
                 pathway=None, score_method=None):
        super(STDataset_Pathway, self).__init__()

        if pathway is None or score_method is None:
            raise ValueError("Both 'pathway' and 'score_method' must be specified")

        # Instead of gene expression, use pathway scores
        if pathway in adata.var_names:
            self.exp = adata[:, pathway].X.toarray()
        else:
            raise ValueError(f"Pathway {pathway} not found in adata.var_names")
            
        self.im = read_tiff(img_path)
        self.r = np.ceil(diameter/2).astype(int)
        self.train = train
        
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])
        
        self.centers = adata.obsm['spatial']
        self.pos = adata.obsm['position_norm']
        
    def __getitem__(self, index):
        exp = self.exp[index]
        center = self.centers[index]
        x, y = center
        patch = self.im.crop((x-self.r, y-self.r, x+self.r, y+self.r))
        exp = torch.Tensor(exp)
        mask = exp != 0
        mask = mask.float()
        if self.train:
            patch = self.transforms(patch)
        pos = torch.Tensor(self.pos[index])
        return patch, pos, exp, mask

    def __len__(self):
        return len(self.centers)


if __name__ == '__main__':
    # Test the pathway dataset
    dataset = ViT_HER2ST_Pathway(
        train=True, 
        fold=0,
        pathway='KEGG_P53_SIGNALING_PATHWAY',
        score_method='AUCell'
    )
    
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)
