import pandas as pd
import numpy as np

from catboost import CatBoostRegressor

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
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

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X), start=1):
    print(f"\nTreinando fold {fold}...")

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    model = MultiOutputRegressor(
        CatBoostRegressor(
            iterations=1200,
            learning_rate=0.025,
            depth=6,
            loss_function="RMSE",
            verbose=False,
            random_seed=42
        )
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, pred)

    print(f"MAE fold {fold}: {mae}")

    scores.append(mae)

print("\nResultado final da validação cruzada:")
print("MAE médio:", np.mean(scores))
print("Desvio padrão:", np.std(scores))

OUTPUTS_DIR.mkdir(exist_ok=True)

pd.DataFrame({
    "fold": list(range(1, 6)),
    "mae": scores
}).to_csv(
    OUTPUTS_DIR / "cross_validation_results.csv",
    index=False
)

print("\nArquivo salvo em:")
print(OUTPUTS_DIR / "cross_validation_results.csv")