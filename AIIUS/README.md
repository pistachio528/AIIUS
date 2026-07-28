# Decision-Boundary Aware Undersampling via Adversarial Influence for Imbalanced Software Defect Prediction

### Overview

**AIIUS (Adversarial Influence-Based Intelligent Undersampling)** is an intelligent undersampling method for handling imbalanced datasets in software defect prediction. 
### Repository Structure

```
AIIUS/
├── AIIUS.py                    # Main implementation
├── AIIUS-multiModel.py         # Multi-architecture surrogate model support
├── AIIUS-noCluster.py          # Version without clustering
├── baseline/                   # Baseline methods for comparison
│   ├── borderline-smote.py     # Borderline-SMOTE
│   ├── UFIDSF.py               # UFIDSF method
│   ├── smote lr.py             # SMOTE
│   ├── rus knn.py              # Random Undersampling
│   └── ...
├── result/                     # Experimental results
│   ├── *_mean_results.csv      # Average results for each classifier
│   ├── boxplot_*.png           # Performance comparison charts
│   └── ...
├── boxplot.py                  # Boxplot visualization script
├── Statistical analysis-autom.py # Statistical analysis
├── vision.py                   # Result visualization
└── zhexian.py                  # Line chart generation
```

### Installation

#### Requirements

- Python 3.7+
- PyTorch 1.8+
- scikit-learn 0.24+
- pandas, numpy, scipy

#### Install Dependencies

```bash
pip install torch torchvision
pip install scikit-learn pandas numpy scipy matplotlib
pip install imbalanced-learn  # For baseline methods
```

### Quick Start

```bash
python AIIUS.py
```

### Evaluation Metrics

- **AUC**: Area Under ROC Curve
- **MCC**: Matthews Correlation Coefficient
- **Balance**: `1 - sqrt((PF² + (1-PD)²)/2)`
- **PD**: Probability of Detection (Recall)
- **PF**: Probability of False Alarm

  
### Datasets

Our experiments uses the following publicly available software defect prediction datasets:

#### 1. **AEEEM Dataset**
- **Projects**: JDT, LC, ML, PDE
- **Repository**: [GitHub - bharlow058/AEEEM-and-other-SDP-datasets](https://github.com/bharlow058/AEEEM-and-other-SDP-datasets)


#### 2. **NASA MDP (Metrics Data Program) Dataset**
- **Projects**: CM1, MW1, PC1, PC3, PC4
- **Repository**:  [GitHub - klainfo/NASADefectDataset](https://github.com/klainfo/NASADefectDataset) (Cleaned NASA datasets)


#### 3. **PROMISE Repository**
- **Projects**: ant-1.3, camel-1.0, synapse-1.0, xalan-2.4
- **Repository**: [http://openscience.us/repo/defect/](http://openscience.us/repo/defect/)


#### 4. **SOFTLAB Dataset**
- **Projects**: AR1, AR3, AR4, AR5, AR6

- **Repository**:  [GitHub - klainfo/DefectData](https://github.com/klainfo/DefectData) (R package containing SOFTLAB datasets)

#### 5. **ReLink Dataset**
- **Repository**: [http://www.cse.ust.hk/~scc/ReLink.htm](http://www.cse.ust.hk/~scc/ReLink.htm)

#### Data Preprocessing

All datasets undergo the following preprocessing steps:
1. Label encoding (0: non-defective, 1: defective/buggy)
2. Log transformation: `log(x + 1)` for all numeric features
3. Stratified train-test split to maintain class distribution
