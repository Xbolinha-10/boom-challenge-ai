import joblib
import pandas as pd

from lightgbm import LGBMRegressor

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from config import *
from features import add_features


print("Lendo datasets...")

train = pd.read_csv(TRAIN_PATH)

labels = pd.read_csv(TRAIN_LABELS_PATH)

print(train.shape)

print(labels.shape)

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

print("Treinando modelo...")

model = MultiOutputRegressor(
    LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=64
    )
)

model.fit(X_train, y_train)

pred = model.predict(X_valid)

mae = mean_absolute_error(y_valid, pred)

print("MAE:", mae)

MODELS_DIR.mkdir(exist_ok=True)

joblib.dump(model, MODELS_DIR / "model.pkl")

print("Modelo salvo.")