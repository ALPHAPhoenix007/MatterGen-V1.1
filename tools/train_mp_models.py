import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ============================================================
# MATTERGEN V1.2 — MATERIALS PROJECT ML BENCHMARK
# ============================================================

INPUT_PATH = Path(
    "data/materials_project_ml_reduced.csv"
)

MODEL_DIR = Path(
    "models/mp_v1_2"
)


TARGETS = [
    "band_gap_ev",
    "formation_energy_per_atom",
    "energy_above_hull",
]

CLASSIFICATION_TARGET = "is_stable"


IDENTIFIER_COLUMNS = [
    "material_id",
    "formula",
]


RANDOM_STATE = 42


# ============================================================
# MODEL DEFINITIONS
# ============================================================

REGRESSORS = {

    "random_forest": RandomForestRegressor(
        n_estimators=150,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=RANDOM_STATE,
    ),

    "extra_trees": ExtraTreesRegressor(
        n_estimators=150,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
    ),
}


CLASSIFIERS = {

    "random_forest": RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        max_features="sqrt",
    ),

    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=RANDOM_STATE,
    ),

    "extra_trees": ExtraTreesClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        max_features="sqrt",
    ),
}


# ============================================================
# REGRESSION TRAINING
# ============================================================

def train_regression(
    df,
    target,
):

    print("\n" + "=" * 60)
    print(
        f"REGRESSION TARGET: {target}"
    )
    print("=" * 60)

    feature_columns = [
        column
        for column in df.columns
        if column not in (
            IDENTIFIER_COLUMNS
            + TARGETS
            + [CLASSIFICATION_TARGET]
        )
    ]

    X = df[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    y = pd.to_numeric(
        df[target],
        errors="coerce"
    )

    valid = y.notna()

    X = X.loc[valid]
    y = y.loc[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    results = []

    for model_name, model in REGRESSORS.items():

        print(
            f"\nTraining {model_name}..."
        )

        pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                model,
            ),
        ])

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        results.append({
            "target": target,
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        print(
            f"  MAE : {mae:.6f}"
        )

        print(
            f"  RMSE: {rmse:.6f}"
        )

        print(
            f"  R²  : {r2:.6f}"
        )

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = MODEL_DIR / (
            f"{target}_{model_name}.joblib"
        )

        joblib.dump(
            pipeline,
            model_path
        )

    return results


# ============================================================
# CLASSIFICATION TRAINING
# ============================================================

def train_classification(
    df,
):

    target = CLASSIFICATION_TARGET

    print("\n" + "=" * 60)
    print(
        f"CLASSIFICATION TARGET: {target}"
    )
    print("=" * 60)

    feature_columns = [
        column
        for column in df.columns
        if column not in (
            IDENTIFIER_COLUMNS
            + TARGETS
            + [CLASSIFICATION_TARGET]
        )
    ]

    X = df[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results = []

    for model_name, model in CLASSIFIERS.items():

        print(
            f"\nTraining {model_name}..."
        )

        pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                model,
            ),
        ])

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        probabilities = pipeline.predict_proba(
            X_test
        )[:, 1]

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions
        )

        auc = roc_auc_score(
            y_test,
            probabilities
        )

        results.append({
            "target": target,
            "model": model_name,
            "accuracy": accuracy,
            "F1": f1,
            "ROC_AUC": auc,
        })

        print(
            f"  Accuracy: {accuracy:.6f}"
        )

        print(
            f"  F1      : {f1:.6f}"
        )

        print(
            f"  ROC-AUC : {auc:.6f}"
        )

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = MODEL_DIR / (
            f"{target}_{model_name}.joblib"
        )

        joblib.dump(
            pipeline,
            model_path
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "MATTERGEN V1.2 — MATERIALS PROJECT ML BENCHMARK"
    )
    print("=" * 60)

    print(
        "\nLoading ML dataset..."
    )

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    print(
        f"Materials: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Regression
    # --------------------------------------------------------

    regression_results = []

    for target in TARGETS:

        results = train_regression(
            df,
            target
        )

        regression_results.extend(
            results
        )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    classification_results = (
        train_classification(
            df
        )
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    regression_df = pd.DataFrame(
        regression_results
    )

    classification_df = pd.DataFrame(
        classification_results
    )

    print("\n" + "=" * 60)
    print("REGRESSION RESULTS")
    print("=" * 60)

    print(
        regression_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)

    print(
        classification_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save benchmark results
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    regression_df.to_csv(
        MODEL_DIR / "regression_results.csv",
        index=False
    )

    classification_df.to_csv(
        MODEL_DIR / "classification_results.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("V1.2 ML BENCHMARK COMPLETE")
    print("=" * 60)

    print(
        f"\nModels saved to: {MODEL_DIR}"
    )


if __name__ == "__main__":
    main()