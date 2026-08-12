import sys
import os

# Allow imports from the MatterGen project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

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
    """Build the feature matrix from the dataset."""

    features = []
    valid_indices = []

    for index, smiles in enumerate(df["smiles"]):

        feature_vector = calculate_features(smiles)

        if feature_vector is not None:
            features.append(feature_vector)
            valid_indices.append(index)

    return np.vstack(features), np.array(valid_indices)


def main():

    print("\n========== MATTERGEN 5-FOLD CROSS-VALIDATION ==========\n")

    df = pd.read_csv(DATASET_PATH)

    X, valid_indices = build_feature_matrix(df)

    print(f"Dataset size: {len(df)}")
    print(f"Feature count: {X.shape[1]}")

    # ---------------------------------------------------------
    # 5-fold cross-validation
    # ---------------------------------------------------------

    cv = KFold(
        n_splits=ML_CONFIG["cv_folds"],
        shuffle=True,
        random_state=ML_CONFIG["random_state"]
    )

    # ---------------------------------------------------------
    # Models
    #
    # Ridge gets StandardScaler because it is sensitive
    # to feature scale.
    #
    # Tree models do not require scaling.
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

        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ])
    }

    all_results = {}

    # ---------------------------------------------------------
    # Evaluate every model on every property
    # ---------------------------------------------------------

    for model_name, model in models.items():

        print(f"\n========== {model_name.upper()} ==========\n")

        all_results[model_name] = {}

        for target in TARGETS:

            y = df[target].iloc[valid_indices].values

            scores = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring={
                    "r2": "r2",
                    "mae": "neg_mean_absolute_error",
                    "rmse": "neg_root_mean_squared_error"
                },
                n_jobs=-1
            )

            r2_scores = scores["test_r2"]

            mae_scores = -scores["test_mae"]

            rmse_scores = -scores["test_rmse"]

            all_results[model_name][target] = {
                "r2_mean": np.mean(r2_scores),
                "r2_std": np.std(r2_scores),
                "mae_mean": np.mean(mae_scores),
                "mae_std": np.std(mae_scores),
                "rmse_mean": np.mean(rmse_scores),
                "rmse_std": np.std(rmse_scores)
            }

            print(target)
            print("-" * len(target))
            print(
                f"R²:   {np.mean(r2_scores):.4f} "
                f"+/- {np.std(r2_scores):.4f}"
            )
            print(
                f"MAE:  {np.mean(mae_scores):.4f} "
                f"+/- {np.std(mae_scores):.4f}"
            )
            print(
                f"RMSE: {np.mean(rmse_scores):.4f} "
                f"+/- {np.std(rmse_scores):.4f}"
            )
            print()

    # ---------------------------------------------------------
    # Final R² comparison
    # ---------------------------------------------------------

    print("\n========== CROSS-VALIDATION R² COMPARISON ==========\n")

    for target in TARGETS:

        print(target)

        for model_name in models:

            result = all_results[model_name][target]

            print(
                f"  {model_name:<20} "
                f"{result['r2_mean']:.4f} "
                f"+/- {result['r2_std']:.4f}"
            )

        print()

    print("=====================================================\n")


if __name__ == "__main__":
    main()