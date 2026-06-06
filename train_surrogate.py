"""
Surrogate Model Training
QM9 -> Morgan Fingerprint -> RandomForestRegressor -> dipole moment prediction
"""

import os
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def smiles_to_fp(smiles: str, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    return gen.GetFingerprintAsNumPy(mol).astype(np.float32)


def load_qm9() -> pd.DataFrame:
    print("[INFO] Downloading QM9 dataset...")
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
    df = pd.read_csv(url)
    df = df[["smiles", "mu"]].dropna().reset_index(drop=True)
    print(f"[INFO] Loaded {len(df)} molecules.")
    return df


def build_features(df: pd.DataFrame, n_bits: int = 2048):
    fps, targets, valid_smiles = [], [], []
    print("[INFO] Converting SMILES to Morgan Fingerprints...")
    for idx, row in df.iterrows():
        fp = smiles_to_fp(row["smiles"], n_bits=n_bits)
        if fp is not None:
            fps.append(fp)
            targets.append(row["mu"])
            valid_smiles.append(row["smiles"])
        if (idx + 1) % 40000 == 0:
            print(f" -> Processed {idx + 1} molecules...")
    print(f"[INFO] Valid molecules: {len(fps)} / {len(df)}")
    return np.array(fps), np.array(targets), valid_smiles


def train_surrogate(save_path: str = "models/surrogate_rf.pkl",
                    n_estimators: int = 100, test_size: float = 0.2):
    df = load_qm9()
    X, y, smiles_list = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    print(f"[INFO] Training RandomForest (n_estimators={n_estimators})...")
    model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\n===== Surrogate Model Performance =====")
    print(f"  R2   : {r2_score(y_test, y_pred):.4f}")
    print(f"  RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f} Debye")
    print(f"  MAE  : {mean_absolute_error(y_test, y_pred):.4f} Debye")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"[INFO] Model saved: {save_path}")

    os.makedirs("data", exist_ok=True)
    np.save("data/fps.npy", X)
    np.save("data/targets.npy", y)
    pd.Series(smiles_list).to_csv("data/smiles_list.csv", index=False, header=["smiles"])
    print("[INFO] Data saved: data/")


if __name__ == "__main__":
    train_surrogate()
