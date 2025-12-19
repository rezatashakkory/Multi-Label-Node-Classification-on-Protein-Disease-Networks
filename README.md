# Disease-Gene Association Prediction 🧬

**Final Project - Analysis of Complex Networks**  
**Achievement: 2nd Place (AP: 0.057353)**

## Overview

This project tackles multi-label disease-gene association prediction using the BioGRID protein-protein interaction network. Through systematic experimentation, we discovered that **low homophily (0.0252)** in biological networks makes traditional GNNs ineffective, leading us to develop an ensemble approach combining SIGN, XGBoost, and Label Propagation.

### Key Achievements
- 🥈 **2nd Place** in class competition with 0.057353 Average Precision
- 🔍 **Critical Discovery**: Network homophily of 0.0252 explains why GNNs underperform
- 🎯 **Winning Strategy**: 60% Label Propagation + 20% XGBoost + 20% SIGN ensemble
- 📊 **Feature Engineering**: Enhanced 37 biological features to 103 features using Node2Vec and network centrality

## Repository Structure

```
Group_HAR_COWINE/
├── Codes/
│   ├── Final_Pipeline (Optional).ipynb    # Main reproducible pipeline (merged from drafts)
│   ├── Exploration_Data_Analysis.ipynb    # Network property analysis & visualization
│   └── Drafts/                            # Development history
│       ├── Draft4_GNN+C&S.ipynb           # GNN experiments with Correct & Smooth
│       ├── Draft8_Node2Vec_Baseline.ipynb # Node2Vec embedding training
│       ├── Draft9_Advanced_Ensemble.ipynb # Ensemble experimentation
│       ├── Draft11_MultiScale.ipynb       # Multi-scale Label Propagation
│       └── Draft13_Ensemble.ipynb         # Final winning ensemble
│       └── Pure_LP.py                     # Final Label Propagation script
├── data/
│   ├── edge_index.pt                      # BioGRID PPI network edges
│   ├── node_features.pt                   # 37-dimensional biological features
│   ├── y.pt                               # 305 disease labels
│   ├── train_idx.pt / test_idx.pt        # Train/test split indices
│   ├── node2vec_*.pt                      # Pre-computed Node2Vec embeddings
│   └── sample_submission.csv              # Submission format template
├── Submissions/                           # Output folder for predictions
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 12.9+ (for GPU acceleration)
- 16GB+ RAM recommended

### Setup Environment

```bash
# Clone or download this repository
cd Group_HAR_COWINE

# Install dependencies
pip install -r requirements.txt
```

**Note**: The `requirements.txt` includes:
- PyTorch 2.8.0 with CUDA 12.9
- PyTorch Geometric 2.7.0
- XGBoost 3.1.2
- NetworkX 3.6.1
- scikit-learn, pandas, numpy, scipy, matplotlib, seaborn

## Quick Start

### Running the Complete Pipeline

1. **Open the main notebook**:
   ```bash
   jupyter notebook "Codes/Final_Pipeline (Optional).ipynb"
   ```

2. **Execute cells sequentially**:
   - Cell 1-3: Environment setup and data loading
   - Cell 4-8: Node2Vec embedding training (or loads pre-computed)
   - Cell 9-12: Feature engineering (37 → 103 features)
   - Cell 13-20: GNN baseline models (GraphSAGE, GCN, GAT)
   - Cell 21-28: SIGN model training with Focal Loss
   - Cell 29-35: XGBoost 305 binary classifiers
   - Cell 36-40: Label Propagation (α=0.85, 50 iterations)
   - Cell 41-43: Probability calibration (temperature scaling)
   - Cell 44-45: Ensemble & submission generation

3. **Output**: Final predictions saved to `Submissions/submission_final.csv`

### Exploring the Data

Open `Codes/Exploration_Data_Analysis.ipynb` to see:
- Network statistics (19,765 nodes, 777,395 edges)
- Homophily analysis (critical 0.0252 discovery)
- Degree distributions and centrality measures
- Label sparsity analysis (96.96% sparsity)
- Visualization of network structure

## Data Description

- **Nodes**: 19,765 proteins in BioGRID PPI network
- **Edges**: 777,395 undirected protein-protein interactions
- **Features**: 37 biological features per node (pre-computed embeddings)
- **Labels**: 305 diseases (multi-label, extremely sparse - 3.04% density)
- **Split**: 15,812 training nodes, 3,953 test nodes
- **Network Type**: Heterophilic (homophily = 0.0252)

## Methodology

### Model Pipeline

Our final pipeline merges components from multiple experimental drafts (see `Codes/Drafts/`):

1. **Node2Vec Embeddings** (Draft 8)
   - 64-dimensional embeddings with p=1, q=1
   - Trained on full BioGRID network
   - Pre-computed for efficiency

2. **Feature Engineering**
   - Original: 37 biological features
   - Enhanced: 103 features (Node2Vec 64d + log-degree + PageRank)

3. **SIGN Model** (Draft 4, Draft 13)
   - K=3 hop propagation
   - Label reuse: 305 → 32 label features
   - Focal Loss (α=0.25, γ=2.0) for imbalance
   - Temperature scaling calibration (T=2.0)

4. **XGBoost Ensemble**
   - 305 independent binary classifiers
   - `scale_pos_weight` for class imbalance
   - Probability rescaling calibration

5. **Label Propagation** (Draft 11, Draft 13)
   - α=0.85 (85% neighbor influence)
   - 50 iterations
   - Symmetric normalization
   - **Contributes 95.8% of final performance**

6. **Calibrated Ensemble** (Draft 13)
   - 20% SIGN + 20% XGBoost + 60% Label Propagation
   - Weighted average of calibrated probabilities
   - Final AP: **0.057353** (2nd place)

### Why This Works

**Key Insight**: Traditional GNNs fail because biological networks are **heterophilic** (homophily = 0.0252). Proteins connected to each other often have *different* functions, violating the homophily assumption that underlies message-passing neural networks.

**Solution**: Label Propagation doesn't assume homophily and performs simple probability smoothing, making it ideal for heterophilic networks. The ensemble adds complementary signal from SIGN (which incorporates label information) and XGBoost (which captures feature patterns).

## Results

| Model                    | Average Precision |
|--------------------------|-------------------|
| GraphSAGE Baseline       | 0.026             |
| GCN + C&S                | 0.036             |
| GAT + C&S                | 0.052             |
| Pure Label Propagation   | 0.057156          |
| **Final Ensemble**       | **0.057353** ✅   |

**Analysis**: Label Propagation alone achieves 99.66% of our final performance, validating that simpler methods matched to data properties outperform complex models with mismatched assumptions.

## Drafts Folder

The `Codes/Drafts/` folder contains our development history:
- **Draft 4**: GNN architectures with Correct & Smooth post-processing
- **Draft 8**: Node2Vec baseline and embedding training
- **Draft 9**: Advanced ensemble experiments
- **Draft 11**: Multi-scale Label Propagation exploration
- **Draft 13**: Final winning ensemble (merged into main pipeline)

The final code in `Final_Pipeline (Optional).ipynb` represents a carefully curated merge of the most effective components from these experimental notebooks.

## Citation

If you use this code or methodology, please reference:
```
Disease-Gene Association Prediction via Heterophilic Graph Learning
Analysis of Complex Networks - Final Project
December 2025
```

## Acknowledgments

- BioGRID database for protein-protein interaction data
- PyTorch Geometric team for graph learning framework
- Course instructors for guidance and evaluation

---

**Authors**: Group_HAR_COWINE  
**Course**: Analysis of Complex Networks, Semester 3  
**Date**: December 2025
