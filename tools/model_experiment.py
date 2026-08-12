import sys
import os

# Allow imports from the MatterGen project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.config import ML_CONFIG


DATASET_PATH = "data/materials_dataset.csv"

TARGETS = [
    "band_gap_ev",
    "formation_energy",
    "stability_score",
    "melting_point_k"
]

# ---------------------------------------------------------
# 21-feature representation
# ---------------------------------------------------------

FEATURES = [
    "MolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "TPSA",
    "NumRotatableBonds",
    "NumAromaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
    "NumHeteroatoms",
    "RingCount",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "HeavyAtomMolWt",
    "ExactMolWt",
    "MolMR",
    "LabuteASA",
    "BalabanJ",
    "BertzCT"
]


def calculate_features(smiles):
    """Calculate the 21 RDKit descriptors for one molecule."""

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    values = []

    for feature_name in FEATURES:
        feature_function = getattr(Descriptors, feature_name)
        value = feature_function(mol)

        if not np.isfinite(value):
            value = 0.0

        values.append(float(value))

    return np.array(values)


def build_feature_matrix(df):

    features = []
    valid_indices = []

    for index, smiles in enumerate(df["smiles"]):

        feature_vector = calculate_features(smiles)

        if feature_vector is not None:
            features.append(feature_vector)
            valid_indices.append(index)

    return np.vstack(features), np.array(valid_indices)


def evaluate_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return mae, rmse, r2


def main():

    print("\n========== MATTERGEN MODEL EXPERIMENT ==========\n")

    df = pd.read_csv(DATASET_PATH)

    X, valid_indices = build_feature_matrix(df)

    print(f"Dataset size: {len(df)}")
    print(f"Feature count: {X.shape[1]}")

    # ---------------------------------------------------------
    # One fixed split for every model and every property
    # ---------------------------------------------------------

    indices = np.arange(len(valid_indices))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=ML_CONFIG["test_size"],
        random_state=ML_CONFIG["random_state"]
    )

    X_train = X[train_indices]
    X_test = X[test_indices]

    print(f"Training samples: {len(train_indices)}")
    print(f"Testing samples:  {len(test_indices)}")

    # ---------------------------------------------------------
    # Models
    # ---------------------------------------------------------

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=ML_CONFIG["random_state"],
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=ML_CONFIG["random_state"]
        ),

        "Ridge": Ridge(
            alpha=1.0
        )
    }

    # Store results for final comparison
    all_results = {}

    # ---------------------------------------------------------
    # Evaluate every model on every property
    # ---------------------------------------------------------

    for model_name, model in models.items():

        print(f"\n========== {model_name.upper()} ==========\n")

        all_results[model_name] = {}

        for target in TARGETS:

            y = df[target].iloc[valid_indices].values

            y_train = y[train_indices]
            y_test = y[test_indices]

            mae, rmse, r2 = evaluate_model(
                model,
                X_train,
                X_test,
                y_train,
                y_test
            )

            all_results[model_name][target] = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2
            }

            print(target)
            print("-" * len(target))
            print(f"MAE:   {mae:.4f}")
            print(f"RMSE:  {rmse:.4f}")
            print(f"R²:    {r2:.4f}")
            print()

    # ---------------------------------------------------------
    # R² comparison
    # ---------------------------------------------------------

    print("\n========== R² COMPARISON ==========\n")

    for target in TARGETS:

        print(target)

        for model_name in models:

            r2 = all_results[model_name][target]["r2"]

            print(f"  {model_name:<20} {r2:.4f}")

        print()

    print("============================================\n")


if __name__ == "__main__":
    main()