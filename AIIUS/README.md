# AIIUS: Adversarial-Inspired Intelligent Undersampling with Surrogate Models

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## English

### Overview

**AIIUS (Adversarial-Inspired Intelligent Undersampling)** is an intelligent undersampling method for handling imbalanced datasets in software defect prediction. The method combines clustering analysis, adversarial attack techniques, and global surrogate models to select the most informative majority class samples, thereby achieving more effective class balance.

### Repository Structure

```
AIIUS/
├── AIIUS.py                    # Main implementation: clustering + adversarial selection
├── AIIUS-multiModel.py         # Multi-architecture surrogate model support
├── AIIUS-noCluster.py          # Version without clustering (direct selection on all samples)
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

#### 1. Run Main Algorithm

```bash
python AIIUS.py
```

#### 2. Run Multi-Architecture Experiments

```bash
python AIIUS-multiModel.py
```
Supports four surrogate model architectures:
- `mlp`: Basic Multi-Layer Perceptron
- `deep_mlp`: Deep MLP (4 hidden layers)
- `cnn`: Convolutional Neural Network
- `resnet`: Residual Network

#### 3. Run Baseline Methods

```bash
# Run Borderline-SMOTE
python baseline/borderline-smote.py

# Run UFIDSF
python baseline/UFIDSF.py
```

### Evaluation Metrics

- **AUC**: Area Under ROC Curve
- **MCC**: Matthews Correlation Coefficient
- **Balance**: `1 - sqrt((PF² + (1-PD)²)/2)`
- **PD**: Probability of Detection (Recall)
- **PF**: Probability of False Alarm
