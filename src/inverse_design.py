import json
import joblib
import pandas as pd
import optuna

from config import *
from features import add_features


INVERSE_DIR = DATA_RAW / "inverse_design"
CONSTRAINTS_PATH = INVERSE_DIR / "constraints.json"
OUTPUT_PATH = OUTPUTS_DIR / "inverse_design.csv"

print("Carregando modelo CatBoost...")
model = joblib.load(MODELS_DIR / "catboost.pkl")

print("Lendo constraints.json...")
with open(CONSTRAINTS_PATH, "r", encoding="utf-8") as f:
    config_json = json.load(f)

constraints = config_json["constraints"]
bounds = config_json["input_bounds"]

input_columns = [
    "energy",
    "angle_rad",
    "coupling",
    "strength",
    "porosity",
    "gravity",
    "atmosphere",
    "shape_factor",
]

accepted = []


def predict(params):
    df = pd.DataFrame([params])

    # Garante ordem correta das colunas
    df = df[input_columns]

    df = add_features(df)

    pred = model.predict(df)[0]

    return {
        "P80": pred[0],
        "fines_frac": pred[1],
        "oversize_frac": pred[2],
        "R95": pred[3],
        "R50_fines": pred[4],
        "R50_oversize": pred[5],
    }


def objective(trial):
    params = {}

    for col in input_columns:
        low = bounds[col]["min"]
        high = bounds[col]["max"]

        params[col] = trial.suggest_float(col, low, high)

    pred = predict(params)

    p80 = pred["P80"]
    r95 = pred["R95"]

    p80_min = constraints["p80_min"]
    p80_max = constraints["p80_max"]
    r95_max = constraints["r95_max"]

    penalty = 0

    if p80 < p80_min:
        penalty += (p80_min - p80) ** 2

    if p80 > p80_max:
        penalty += (p80 - p80_max) ** 2

    if r95 > r95_max:
        penalty += (r95 - r95_max) ** 2

    # Critério adicional: preferir impactos menores
    penalty += params["energy"] * 0.001

    # Critério adicional: preferir menor alcance
    penalty += r95 * 0.01

    if p80_min <= p80 <= p80_max and r95 <= r95_max:
        row = params.copy()
        row["predicted_P80"] = p80
        row["predicted_R95"] = r95
        row["score"] = penalty
        accepted.append(row)

    return penalty


print("Rodando otimização inversa...")

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5000)

if len(accepted) > 0:
    print(f"Soluções válidas encontradas: {len(accepted)}")
    result = pd.DataFrame(accepted)
    result = result.sort_values("score").head(20)
else:
    print("Nenhuma solução perfeita encontrada. Usando os 20 melhores trials.")
    best_trials = sorted(study.trials, key=lambda t: t.value)[:20]

    rows = []

    for trial in best_trials:
        params = trial.params
        pred = predict(params)

        row = params.copy()
        row["predicted_P80"] = pred["P80"]
        row["predicted_R95"] = pred["R95"]
        row["score"] = trial.value

        rows.append(row)

    result = pd.DataFrame(rows)

OUTPUTS_DIR.mkdir(exist_ok=True)

result.to_csv(OUTPUT_PATH, index=False)

print("Arquivo salvo em:")
print(OUTPUT_PATH)

print(result)