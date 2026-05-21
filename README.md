# 🚀 Boom Challenge AI Solution

Physics-Informed Machine Learning solution for asteroid impact ejecta prediction and inverse impact design.

---

# 📌 Overview

This repository contains a complete AI pipeline developed for the Boom Challenge.

The project combines:

- Physics-informed feature engineering
- Gradient boosting models
- CatBoost optimization
- Automated inverse design
- Bayesian optimization using Optuna
- SHAP explainability
- Cross-validation analysis

The objective is to predict fragmentation and ejecta behavior after asteroid impacts.

---

# 🎯 Challenge Tasks

## Forward Prediction

Predict:

- P80
- fines_frac
- oversize_frac
- R95
- R50_fines
- R50_oversize

from asteroid impact parameters.

---

## Inverse Design

Search for impact scenarios satisfying:

```text
96 <= P80 <= 101
R95 <= 175
```

while minimizing:

- impact energy
- ejecta distance

---

# 🧠 Machine Learning Models

The following models were tested:

| Model | MAE |
|---|---|
| LightGBM | 22.74 |
| XGBoost | 21.08 |
| CatBoost | 19.75 |
| Ensemble | 21.02 |

---

# 🏆 Best Model

```text
CatBoost
```

Final validation MAE:

```text
19.75
```

---

# ✅ Cross Validation Results

5-Fold Cross Validation Results:

| Fold | MAE |
|---|---|
| 1 | 20.34 |
| 2 | 21.76 |
| 3 | 20.31 |
| 4 | 20.19 |
| 5 | 20.12 |

Average MAE:

```text
20.55
```

Standard deviation:

```text
0.61
```

---

# ⚙️ Physics-Informed Features

Custom engineered physical features:

- effective_energy
- fragmentation_index
- energy_log
- angle_sin
- angle_cos

These features improve generalization on out-of-distribution scenarios.

---

# 📊 Feature Importance

Top variables discovered by CatBoost:

- energy
- effective_energy
- strength
- fragmentation_index
- gravity
- angle_rad

Generated automatically using feature importance analysis.

---

# 🔍 SHAP Explainability

SHAP analysis was used to interpret model behavior and identify the most influential variables.

Generated output:

```text
outputs/shap_summary.png
```

This analysis improves:

- interpretability
- model transparency
- scientific credibility

---

# ⚡ Hyperparameter Optimization

CatBoost hyperparameters were automatically optimized using Optuna.

Optimized parameters:

```text
iterations = 693
learning_rate = 0.01766
depth = 5
l2_leaf_reg = 8.00
random_strength = 2.77
```

This optimization improved the MAE from:

```text
20.09 → 19.75
```

---

# 📑 Technical Report

A complete technical report is automatically generated after training and optimization.

Generated file:

```text
outputs/technical_report.md
```

The report includes:

- validation metrics
- cross-validation results
- optimized inverse design scenarios
- feature engineering summary
- generated outputs

---

# 📂 Project Structure

```text
boom-challenge/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── outputs/
│   ├── submission.csv
│   ├── inverse_design.csv
│   ├── validation_results.csv
│   ├── cross_validation_results.csv
│   ├── technical_report.md
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   └── shap_summary.png
│
├── notebooks/
│   └── boom_challenge_analysis.ipynb
│
├── src/
│   ├── config.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   ├── inverse_design.py
│   ├── analyze_model.py
│   ├── shap_analysis.py
│   ├── cross_validate.py
│   ├── tune_catboost.py
│   └── generate_report.py
│
├── README.md
└── requirements.txt
```

---

# 🛠️ Installation

Clone repository:

```bash
git clone https://github.com/Xbolinha-10/boom-challenge-ai.git
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

```bash
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🏋️ Training

Train the model:

```bash
python src/train.py
```

---

# 🔮 Forward Prediction

Generate challenge predictions:

```bash
python src/predict.py
```

Output:

```text
outputs/submission.csv
```

---

# 🎯 Inverse Design

Generate optimized impact scenarios:

```bash
python src/inverse_design.py
```

Output:

```text
outputs/inverse_design.csv
```

---

# 📈 Model Analysis

Generate feature importance analysis:

```bash
python src/analyze_model.py
```

Outputs:

```text
outputs/feature_importance.csv
outputs/feature_importance.png
```

---

# 🧪 SHAP Explainability

Generate SHAP explainability plots:

```bash
python src/shap_analysis.py
```

Outputs:

```text
outputs/shap_summary.png
```

---

# 🔬 Optimization Strategy

Inverse design uses:

- Optuna
- Bayesian Optimization
- Constraint optimization
- Physics-informed penalties

Optimization goals:

- minimize impact energy
- minimize ejecta distance
- satisfy challenge constraints

---

# 🧪 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- CatBoost
- XGBoost
- LightGBM
- Optuna
- SHAP
- Matplotlib
- Jupyter

---

# 📦 Final Deliverables

This repository generates all challenge deliverables automatically.

## Forward Prediction

```text
outputs/submission.csv
```

## Inverse Design

```text
outputs/inverse_design.csv
```

## Technical Analysis

```text
outputs/technical_report.md
outputs/feature_importance.csv
outputs/feature_importance.png
outputs/shap_summary.png
```

---

# 🚧 Future Improvements

Possible future upgrades:

- Advanced stacking ensembles
- Physics-Informed Neural Networks (PINNs)
- Multi-objective optimization
- GPU acceleration
- Transformer-based surrogate models

---

# 👨‍💻 Author

## Breno Henrique Shimada

GitHub:

https://github.com/Xbolinha-10

---

# ⭐ Final Notes

This project was developed as a complete end-to-end machine learning pipeline for impact ejecta prediction and optimization.

The solution emphasizes:

- generalization
- physics-aware learning
- reproducibility
- optimization
- engineering quality
- explainability
- automated inverse design
```