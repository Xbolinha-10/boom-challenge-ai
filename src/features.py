import numpy as np


def add_features(df):

    df = df.copy()

    df["energy_log"] = np.log1p(df["energy"])

    df["angle_sin"] = np.sin(df["angle_rad"])

    df["angle_cos"] = np.cos(df["angle_rad"])

    df["effective_energy"] = (
        df["energy"] * df["coupling"]
    )

    df["fragmentation_index"] = (
        df["effective_energy"]
        / (df["strength"] + 1e-9)
    )

    return df