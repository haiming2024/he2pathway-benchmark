# he2pathway-benchmark
Benchmarking models (Hist2ST, HisToGene, THItoGene) for pathway expression prediction from H&amp;E images

# he2pathway-benchmark

Benchmarking deep learning models (**Hist2ST, HisToGene, THItoGene**) for **gene and pathway expression prediction from H&E-stained images**.

This repository contains **modified implementations** of three state-of-the-art methods.  
The modifications add **pathway-level prediction** capability (via AUCell/UCell scores) to the original repositories, evaluated on the **HER2ST breast cancer dataset**.

---

## Repository Structure
```
he2pathway-benchmark/
│── Hist2ST/
│ ├── data/ # Raw HER2ST data (download via download.sh)
│ │ └── download.sh
│ ├── dataset_pathway.py # Modified dataset pipeline for pathway scoring
│ ├── HIST2ST_pathway.py # Modified main model script
│ ├── train_pathway.py # Training script (pathway-level)
│ ├── test_pathway.py # Testing script (pathway-level)
│ ├── predict_pathway.py # Prediction script (pathway-level)
│ ├── selective_submit.sh # Training automation
│ ├── selective_test.sh # Testing automation
│ ├── result_summary.sh # Summarize results to CSV
│ └── (original code from Hist2ST
)
│
│── HisToGene/
│ ├── data/
│ │ └── download.sh
│ ├── dataset_pathway.py
│ ├── train_pathway.py
│ ├── test_pathway.py
│ ├── predict_pathway.py
│ ├── selective_submit.sh
│ ├── selective_test.sh
│ ├── result_summary.sh
│ └── (original code from HisToGene
)
│
│── THItoGene/
│ ├── data/
│ │ └── download.sh
│ ├── dataset_pathway.py
│ ├── train_pathway.py
│ ├── test_pathway.py
│ ├── predict_pathway.py
│ ├── selective_submit.sh
│ ├── selective_test.sh
│ ├── result_summary.sh
│ └── (original code from THItoGene
)
│
│── data_pathway/ # Precomputed pathway activity scores (AUCell / UCell, GO / KEGG / MSigDB)
│── README.md
```

---

## Modifications
Compared with the original repositories, the following files were **added or modified** to support **pathway-level prediction**:

- `dataset_pathway.py` → loads AUCell/UCell pathway scores from `data_pathway/`  
- `train_pathway.py` → training script for pathway targets  
- `test_pathway.py` → evaluation with PCC metric  
- `predict_pathway.py` → prediction of pathway activity maps  
- `HIST2ST_pathway.py` (only in Hist2ST) → modified main entry for pathway training  

Additionally, each model folder contains:  
- `selective_submit.sh` → run training across folds and scoring methods  
- `selective_test.sh` → run evaluation  
- `result_summary.sh` → aggregate results into a summary CSV (saved inside each `results/` folder)  
- `results/` → stores experiment outputs for each method separately  

---

## Data Preparation

### 1. Raw HER2ST data
Each model has its own `data/` folder.  
Run the following inside each model’s data folder to download/prepare raw data:

```bash
cd Hist2ST/data && bash download.sh
Run gunzip *.gz in the dir Hist2ST/data/her2st/data/ST-cnts/ to unzip the gz files
cd ../../HisToGene/data && bash download.sh
Run gunzip *.gz in the dir HisToGene/data/her2st/data/ST-cnts/ to unzip the gz files
cd ../../THItoGene/data && bash download.sh
Run gunzip *.gz in the dir THItoGene/data/her2st/data/ST-cnts/ to unzip the gz files

Usage

Run all scripts inside each model’s folder.

bash selective_submit.sh          # all folds, all scoring
bash selective_submit.sh 0 5      # folds 0–5
bash selective_submit.sh 0 5 AUCell
bash selective_submit.sh 0 5 AUCell GO

bash selective_test.sh
bash selective_test.sh 0 5
bash selective_test.sh 0 5 AUCell
bash selective_test.sh 0 5 AUCell GO

# Default usage (writes to ./results/pathway_performance_summary.csv)
bash result_summary.sh

