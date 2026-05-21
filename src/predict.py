import joblib
import pandas as pd

from config import *
from features import add_features


print("Lendo test.csv...")

test = pd.read_csv(TEST_PATH)

# Se existir scenario_id, usamos ele.
# Se não existir, criamos IDs automáticos.
if "scenario_id" in test.columns:
    ids = test["scenario_id"]
else:
    ids = range(len(test))

test = add_features(test)

if "scenario_id" in test.columns:
    test = test.drop(columns=["scenario_id"])

print("Carregando modelo...")

model = joblib.load(MODELS_DIR / "model.pkl")

print("Gerando previsões...")

pred = model.predict(test)

submission = pd.DataFrame()

submission["scenario_id"] = ids

for i, col in enumerate(TARGET_COLUMNS):
    submission[col] = pred[:, i]

OUTPUTS_DIR.mkdir(exist_ok=True)

output_path = OUTPUTS_DIR / "submission.csv"

submission.to_csv(output_path, index=False)

print("Arquivo salvo em:")
print(output_path)