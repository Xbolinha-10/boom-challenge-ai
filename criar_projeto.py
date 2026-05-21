from pathlib import Path

ROOT = Path.cwd()

folders = [
    "data/raw",
    "data/processed",
    "models",
    "outputs",
    "notebooks",
    "src",
]

for folder in folders:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)

files = {}

files["requirements.txt"] = """numpy
pandas
scikit-learn
xgboost
lightgbm
catboost
optuna
shap
scikit-optimize
matplotlib
joblib
"""

files[".gitignore"] = """venv/
__pycache__/
*.pyc
models/
outputs/
data/raw/
data/processed/
.ipynb_checkpoints/
"""

files["README.md"] = """# Boom Challenge - Physics-Informed ML Solution

This repository contains a machine learning solution for the Boom: Trajectory Unknown Challenge.

## Approach

The solution combines:

- Physics-informed feature engineering
- LightGBM
- XGBoost
- CatBoost
- Ensemble prediction
- Inverse design optimization

## Official Dataset Structure

Place the official dataset inside:

```text
data/raw/
```
"""