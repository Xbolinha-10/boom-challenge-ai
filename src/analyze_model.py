import joblib
import pandas as pd
import matplotlib.pyplot as plt

from config import *
from features import add_features


print("Lendo dados...")

train = pd.read_csv(TRAIN_PATH)
labels = pd.read_csv(TRAIN_LABELS_PATH)

train = add_features(train)

if "scenario_id" in train.columns:
    train = train.drop(columns=["scenario_id"])

print("Carregando modelo CatBoost...")

model = joblib.load(MODELS_DIR / "catboost.pkl")

feature_names = train.columns.tolist()

importances = []

for i, estimator in enumerate(model.estimators_):
    target = TARGET_COLUMNS[i]

    values = estimator.get_feature_importance()

    for feature, importance in zip(feature_names, values):
        importances.append({
            "target": target,
            "feature": feature,
            "importance": importance
        })

importance_df = pd.DataFrame(importances)

summary = (
    importance_df
    .groupby("feature")["importance"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

OUTPUTS_DIR.mkdir(exist_ok=True)

summary.to_csv(
    OUTPUTS_DIR / "feature_importance.csv",
    index=False
)

plt.figure(figsize=(10, 6))
plt.barh(
    summary["feature"].head(15)[::-1],
    summary["importance"].head(15)[::-1]
)
plt.xlabel("Importância média")
plt.ylabel("Variável")
plt.title("Importância das variáveis - CatBoost")
plt.tight_layout()

plt.savefig(
    OUTPUTS_DIR / "feature_importance.png",
    dpi=300
)

print("Análise concluída.")
print("Arquivos criados:")
print(OUTPUTS_DIR / "feature_importance.csv")
print(OUTPUTS_DIR / "feature_importance.png")