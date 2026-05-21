import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from config import *
from features import add_features


print("Lendo datasets...")

train = pd.read_csv(TRAIN_PATH)
labels = pd.read_csv(TRAIN_LABELS_PATH)

print("Train:", train.shape)
print("Labels:", labels.shape)

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

MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

models = {
    "lightgbm": MultiOutputRegressor(
        LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.025,
            num_leaves=64,
            random_state=42,
            verbose=-1
        )
    ),

    "xgboost": MultiOutputRegressor(
        XGBRegressor(
            n_estimators=900,
            learning_rate=0.025,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42
        )
    ),

        "catboost": MultiOutputRegressor(
        CatBoostRegressor(
            iterations=693,
            learning_rate=0.017661730825381874,
            depth=5,
            l2_leaf_reg=8.00070976705026,
            random_strength=2.778623087933151,
            loss_function="RMSE",
            verbose=False,
            random_seed=42
        )
    )

}

predictions = {}
results = []

for name, model in models.items():
    print(f"\nTreinando {name}...")

    model.fit(X_train, y_train)

    pred = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, pred)

    print(f"MAE {name}: {mae}")

    predictions[name] = pred

    joblib.dump(model, MODELS_DIR / f"{name}.pkl")

    results.append({
        "modelo": name,
        "mae": mae
    })


print("\nCriando Ensemble...")

ensemble_pred = (
    0.40 * predictions["lightgbm"]
    + 0.35 * predictions["xgboost"]
    + 0.25 * predictions["catboost"]
)

ensemble_mae = mean_absolute_error(y_valid, ensemble_pred)

print("MAE Ensemble:", ensemble_mae)

results.append({
    "modelo": "ensemble",
    "mae": ensemble_mae
})

pd.DataFrame(results).to_csv(
    OUTPUTS_DIR / "validation_results.csv",
    index=False
)

metadata = {
    "target_columns": TARGET_COLUMNS,
    "weights": {
        "lightgbm": 0.40,
        "xgboost": 0.35,
        "catboost": 0.25
    }
}

joblib.dump(metadata, MODELS_DIR / "metadata.pkl")

print("\nTreinamento finalizado.")
print("Resultados salvos em outputs/validation_results.csv")