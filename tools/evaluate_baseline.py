import sys
import os

# Allow imports from the MatterGen project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend.features import extract_features_from_smiles
from utils.config import ML_CONFIG, PROPERTIES


DATASET_PATH = "data/materials_dataset.csv"


def main():
    print("\n========== MATTERGEN V1 BASELINE ==========\n")

    df = pd.read_csv(DATASET_PATH)

    # ---------------------------------------------------------
    # Extract molecular features
    # ---------------------------------------------------------

    features = []
    valid_indices = []

    for index, smiles in enumerate(df["smiles"]):
        feature_vector = extract_features_from_smiles(smiles)

        if feature_vector is not None:
            features.append(feature_vector)
            valid_indices.append(index)

    X = np.vstack(features)

    print(f"Dataset size: {len(df)}")
    print(f"Feature count: {X.shape[1]}")

    # ---------------------------------------------------------
    # Train / Test split
    # ---------------------------------------------------------

    train_indices, test_indices = train_test_split(
        np.arange(len(valid_indices)),
        test_size=ML_CONFIG["test_size"],
        random_state=ML_CONFIG["random_state"]
    )

    X_train = X[train_indices]
    X_test = X[test_indices]

    # ---------------------------------------------------------
    # Feature names
    # ---------------------------------------------------------

    feature_names = [
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
    # Train and evaluate each property
    # ---------------------------------------------------------

    for property_name in PROPERTIES:

        y = df[property_name].iloc[valid_indices].values

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

        # Metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        print(f"\n{property_name}")
        print("-" * len(property_name))
        print(f"MAE:   {mae:.4f}")
        print(f"RMSE:  {rmse:.4f}")
        print(f"R²:    {r2:.4f}")

        # -----------------------------------------------------
        # Melting point analysis
        # -----------------------------------------------------

        if property_name == "melting_point_k":

            # Feature importance
            importances = model.feature_importances_

            print("\n--- Melting Point Feature Importance ---")

            for name, importance in sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"{name:<22} {importance:.4f}")

            # Individual prediction errors
            results = pd.DataFrame({
                "name": df["name"].iloc[valid_indices].iloc[test_indices].values,
                "actual": y_test,
                "predicted": predictions
            })

            results["error"] = results["predicted"] - results["actual"]
            results["absolute_error"] = abs(results["error"])

            results = results.sort_values(
                "absolute_error",
                ascending=False
            )

            print("\n--- Largest Melting Point Errors ---")

            print(
                results.head(10).to_string(
                    index=False,
                    formatters={
                        "actual": "{:.2f}".format,
                        "predicted": "{:.2f}".format,
                        "error": "{:.2f}".format,
                        "absolute_error": "{:.2f}".format
                    }
                )
            )

    print("\n============================================\n")


if __name__ == "__main__":
    main()