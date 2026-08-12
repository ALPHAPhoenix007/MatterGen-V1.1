import sys
import os

# Allow imports from the MatterGen project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
# Current MatterGen feature set
# ---------------------------------------------------------

BASE_FEATURES = [
    "MolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "TPSA",
    "NumRotatableBonds",
    "NumAromaticRings",
    "FractionCSP3"
]

# ---------------------------------------------------------
# Additional descriptors for the V1.1 experiment
# ---------------------------------------------------------

EXPANDED_FEATURES = BASE_FEATURES + [
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


def calculate_features(smiles, descriptor_names):
    """Calculate selected RDKit descriptors for one molecule."""

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    values = []

    for descriptor_name in descriptor_names:
        descriptor_function = getattr(Descriptors, descriptor_name)
        value = descriptor_function(mol)

        if not np.isfinite(value):
            value = 0.0

        values.append(float(value))

    return np.array(values)


def evaluate_feature_set(df, feature_names, train_indices, test_indices):
    """Evaluate one feature set using the same train/test split."""

    features = []
    valid_indices = []

    for index, smiles in enumerate(df["smiles"]):

        feature_vector = calculate_features(
            smiles,
            feature_names
        )

        if feature_vector is not None:
            features.append(feature_vector)
            valid_indices.append(index)

    X = np.vstack(features)

    # The dataset is already validated, so these should match.
    valid_indices = np.array(valid_indices)

    X_train = X[train_indices]
    X_test = X[test_indices]

    results = {}

    for target in TARGETS:

        y = df[target].iloc[valid_indices].values

        y_train = y[train_indices]
        y_test = y[test_indices]

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=ML_CONFIG["random_state"],
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results[target] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }

    return results


def print_results(title, results):
    print(f"\n========== {title} ==========\n")

    for target, metrics in results.items():

        print(target)
        print("-" * len(target))
        print(f"MAE:   {metrics['mae']:.4f}")
        print(f"RMSE:  {metrics['rmse']:.4f}")
        print(f"R²:    {metrics['r2']:.4f}")
        print()


def main():

    print("\n========== MATTERGEN FEATURE EXPERIMENT ==========\n")

    df = pd.read_csv(DATASET_PATH)

    print(f"Dataset size: {len(df)}")
    print(f"Base feature count: {len(BASE_FEATURES)}")
    print(f"Expanded feature count: {len(EXPANDED_FEATURES)}")

    # ---------------------------------------------------------
    # Create ONE fixed train/test split.
    #
    # Both feature sets use exactly the same molecules
    # in training and testing.
    # ---------------------------------------------------------

    indices = np.arange(len(df))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=ML_CONFIG["test_size"],
        random_state=ML_CONFIG["random_state"]
    )

    print(f"Training samples: {len(train_indices)}")
    print(f"Testing samples:  {len(test_indices)}")

    # ---------------------------------------------------------
    # BASELINE
    # ---------------------------------------------------------

    base_results = evaluate_feature_set(
        df,
        BASE_FEATURES,
        train_indices,
        test_indices
    )

    print_results(
        "BASELINE — 8 FEATURES",
        base_results
    )

    # ---------------------------------------------------------
    # EXPANDED FEATURES
    # ---------------------------------------------------------

    expanded_results = evaluate_feature_set(
        df,
        EXPANDED_FEATURES,
        train_indices,
        test_indices
    )

    print_results(
        "EXPERIMENT — EXPANDED FEATURES",
        expanded_results
    )

    # ---------------------------------------------------------
    # COMPARISON
    # ---------------------------------------------------------

    print("\n========== R² COMPARISON ==========\n")

    for target in TARGETS:

        base_r2 = base_results[target]["r2"]
        expanded_r2 = expanded_results[target]["r2"]

        improvement = expanded_r2 - base_r2

        print(
            f"{target:<22} "
            f"Base: {base_r2:.4f}   "
            f"Expanded: {expanded_r2:.4f}   "
            f"Change: {improvement:+.4f}"
        )

    print("\n============================================\n")


if __name__ == "__main__":
    main()