# DynamicGBert

Implementation of **ETH-GBERT**: dynamic feature fusion for blockchain fraud detection on Ethereum transaction graphs.

> Based on the paper: **Dynamic Feature Fusion: Combining Global Graph Structures and Local Semantics for Blockchain Fraud Detection**  
> Zhang Sheng, Liangliang Song, Yanbin Wang

---

## Overview

DynamicGBert combines:

- **Global structural features** from a graph-based representation of Ethereum accounts and transactions
- **Local semantic features** from BERT applied to transaction text sequences

The model fuses both streams using a dynamic multimodal fusion mechanism to improve fraud detection performance.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/anhnd210020/dynamicgbert.git
cd dynamicgbert
````

### 2. Create environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Prepare data

Put the raw graph file here:

```text
Dataset/MulDiGraph.pkl
```

Then run preprocessing:

```bash
cd Dataset
python prepare_data.py
cd ..
```

### 4. Train

```bash
python trainModel.py --ds Dataset
```

---

## Project Structure

```text
dynamicgbert/
├── Dataset/
│   ├── prepare_data.py
│   ├── dataset1.py ~ dataset11.py
│   ├── adjust_matrix.py
│   ├── BERT_text_data.py
│   └── MulDiGraph.pkl               # not tracked in Git
├── data/
│   └── preprocessed/
│       └── Dataset/                 # generated during preprocessing
├── ETH_GBert.py
├── trainModel.py
├── utils.py
├── env_config.py
├── requirements.txt
└── README.md
```

---

## Repository Policy

This GitHub repository contains **source code only**.

The following assets are **not tracked in Git**:

* raw datasets
* preprocessed artifacts
* graph/cache files (`.pkl`, `.npz`)
* model checkpoints (`.pt`)
* zip archives and generated outputs

These files will be hosted separately on **Hugging Face Hub**.

---

## Dataset

This project uses an Ethereum transaction graph dataset in `NetworkX MultiDiGraph` format.

Expected raw file:

```text
Dataset/MulDiGraph.pkl
```

Generated preprocessing outputs are stored in:

```text
data/preprocessed/Dataset/
```

---

## Data Preprocessing

Run the full preprocessing pipeline:

```bash
cd Dataset
python prepare_data.py
```

Pipeline summary:

1. Extract transactions from graph
2. Add incoming transaction records
3. Compute n-gram time-difference features
4. Attach features to transactions
5. Filter and clean accounts
6. Shuffle transaction order
7. Keep tag only on the first transaction
8. Balance class ratio
9. Normalize tags
10. Convert transactions to text
11. Split into train/valid/test TSV

Additional generated graph/text artifacts are also created for training.

---

## Training

Train from the repository root:

```bash
python trainModel.py --ds Dataset
```

Resume from checkpoint:

```bash
python trainModel.py --ds Dataset --load 1
```

Smoke test:

```bash
python trainModel.py --ds Dataset --validate_program
```

Example with explicit arguments:

```bash
python trainModel.py --ds Dataset --dim 16 --lr 8e-6 --l2 0.001 --sw 0
```

---

## Requirements

Main dependencies:

```text
torch
numpy
scipy
scikit-learn
pandas
networkx
tqdm
nltk
pytorch-pretrained-bert==0.6.2
python-dotenv
```

> Note: this project uses `pytorch-pretrained-bert==0.6.2`, not the modern `transformers` package.

---

## Configuration

You can override runtime settings using environment variables or a `.env` file:

```env
GLOBAL_SEED=44
TRANSFORMERS_OFFLINE=0
HUGGING_LOCAL_MODEL_FILES_PATH=
```

---

## Hugging Face Hub

External assets will be published separately:

* Dataset / preprocessing artifacts: **TBD**
* Model checkpoints: **TBD**

After uploading, add the links here.

---

## Results

Reported paper results:

| Dataset             | F1 Score | Recall | Precision |
| ------------------- | -------: | -----: | --------: |
| Multigraph          |   94.71% | 94.71% |    94.71% |
| Transaction Network |   86.16% | 87.82% |    84.56% |
| B4E                 |   89.79% | 89.57% |    90.84% |

---

## Citation

```bibtex
@article{zhang2025ethgbert,
  title   = {Dynamic Feature Fusion: Combining Global Graph Structures and Local Semantics for Blockchain Fraud Detection},
  author  = {Zhang, Sheng and Song, Liangliang and Wang, Yanbin},
  journal = {arXiv preprint arXiv:2501.02032},
  year    = {2025}
}
```

---