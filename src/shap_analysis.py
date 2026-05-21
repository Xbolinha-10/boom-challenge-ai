import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

from config import *
from features import add_features


print("Lendo dados...")

train = pd.read_csv(TRAIN_PATH)

train = add_features(train)

if "scenario_id" in train.columns:
    train = train.drop(columns=["scenario_id"])

print("Carregando modelo CatBoost...")

model = joblib.load(MODELS_DIR / "catboost.pkl")

# Pegamos apenas o primeiro target
estimator = model.estimators_[0]

print("Calculando SHAP values...")

sample = train.sample(
    n=min(500, len(train)),
    random_state=42
)

explainer = shap.TreeExplainer(estimator)

shap_values = explainer.shap_values(sample)

OUTPUTS_DIR.mkdir(exist_ok=True)

print("Gerando gráfico SHAP summary...")

plt.figure()

shap.summary_plot(
    shap_values,
    sample,
    show=False
)

plt.tight_layout()

plt.savefig(
    OUTPUTS_DIR / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

print("Arquivo salvo em:")
print(OUTPUTS_DIR / "shap_summary.png")