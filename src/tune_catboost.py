import json
import optuna
import pandas as pd

from catboost import CatBoostRegressor

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from config import *
from features import add_features


print("Lendo dados...")

train = pd.read_csv(TRAIN_PATH)
labels = pd.read_csv(TRAIN_LABELS_PATH)

train = add_features(train)

if "scenario_id" in train.columns:
    train = train.drop(columns=["scenario_id"])

X = train
y = labels[TARGET_COLUMNS]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 600, 1800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "random_strength": trial.suggest_float("random_strength", 0.1, 3.0),
        "loss_function": "RMSE",
        "verbose": False,
        "random_seed": 42,
    }

    model = MultiOutputRegressor(
        CatBoostRegressor(**params)
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, pred)

    return mae


print("Iniciando busca Optuna...")

study = optuna.create_study(direction="minimize")

study.optimize(objective, n_trials=30)

print("Melhor MAE:")
print(study.best_value)

print("Melhores parâmetros:")
print(study.best_params)

OUTPUTS_DIR.mkdir(exist_ok=True)

with open(OUTPUTS_DIR / "best_catboost_params.json", "w") as f:
    json.dump(
        {
            "best_mae": study.best_value,
            "best_params": study.best_params,
        },
        f,
        indent=4
    )

print("Arquivo salvo em:")
print(OUTPUTS_DIR / "best_catboost_params.json")