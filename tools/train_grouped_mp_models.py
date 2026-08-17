import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
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

from sklearn.pipeline import Pipeline


# ============================================================
# MATTERGEN V1.2 — FORMULA-GROUPED ML BENCHMARK
# ============================================================

INPUT_PATH = Path(
    "data/materials_project_ml_reduced.csv"
)

MODEL_DIR = Path(
    "models/mp_v1_2/grouped"
)

RANDOM_STATE = 42

TRAIN_FRACTION = 0.80


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

TARGET_COLUMNS = [
    "band_gap_ev",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
]


# ============================================================
# MODELS
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
# FORMULA-GROUPED SPLIT
# ============================================================

def create_grouped_split(df):

    print("\n" + "=" * 60)
    print("CREATING FORMULA-GROUPED SPLIT")
    print("=" * 60)

    # Get unique formulas
    formulas = (
        df["formula"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    rng.shuffle(formulas)

    split_index = int(
        len(formulas) * TRAIN_FRACTION
    )

    train_formulas = set(
        formulas[:split_index]
    )

    test_formulas = set(
        formulas[split_index:]
    )

    train_mask = (
        df["formula"]
        .astype(str)
        .isin(train_formulas)
    )

    test_mask = (
        df["formula"]
        .astype(str)
        .isin(test_formulas)
    )

    train_df = df.loc[
        train_mask
    ].copy()

    test_df = df.loc[
        test_mask
    ].copy()

    print(
        f"Unique formulas: {len(formulas)}"
    )

    print(
        f"Training formulas: {len(train_formulas)}"
    )

    print(
        f"Testing formulas: {len(test_formulas)}"
    )

    print(
        f"\nTraining materials: {len(train_df)}"
    )

    print(
        f"Testing materials: {len(test_df)}"
    )

    overlap = (
        set(train_df["formula"].astype(str))
        &
        set(test_df["formula"].astype(str))
    )

    print(
        f"Formula overlap: {len(overlap)}"
    )

    if len(overlap) != 0:

        raise RuntimeError(
            "Formula leakage detected!"
        )

    print(
        "\nFormula grouping verified."
    )

    return train_df, test_df


# ============================================================
# REGRESSION
# ============================================================

def train_regression(
    train_df,
    test_df,
    target,
):

    print("\n" + "=" * 60)
    print(
        f"REGRESSION TARGET: {target}"
    )
    print("=" * 60)

    feature_columns = [
        column
        for column in train_df.columns
        if column not in (
            IDENTIFIER_COLUMNS
            + TARGET_COLUMNS
        )
    ]

    X_train = train_df[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_test = test_df[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    y_train = pd.to_numeric(
        train_df[target],
        errors="coerce"
    )

    y_test = pd.to_numeric(
        test_df[target],
        errors="coerce"
    )

    train_valid = y_train.notna()
    test_valid = y_test.notna()

    X_train = X_train.loc[
        train_valid
    ]

    y_train = y_train.loc[
        train_valid
    ]

    X_test = X_test.loc[
        test_valid
    ]

    y_test = y_test.loc[
        test_valid
    ]

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

        print(
            f"  MAE : {mae:.6f}"
        )

        print(
            f"  RMSE: {rmse:.6f}"
        )

        print(
            f"  R²  : {r2:.6f}"
        )

        results.append({
            "target": target,
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            pipeline,
            MODEL_DIR
            / f"{target}_{model_name}.joblib"
        )

    return results


# ============================================================
# CLASSIFICATION
# ============================================================

def train_classification(
    train_df,
    test_df,
):

    target = CLASSIFICATION_TARGET

    print("\n" + "=" * 60)
    print(
        f"CLASSIFICATION TARGET: {target}"
    )
    print("=" * 60)

    feature_columns = [
        column
        for column in train_df.columns
        if column not in (
            IDENTIFIER_COLUMNS
            + TARGET_COLUMNS
        )
    ]

    X_train = train_df[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_test = test_df[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    y_train = train_df[
        target
    ].astype(int)

    y_test = test_df[
        target
    ].astype(int)

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

        print(
            f"  Accuracy: {accuracy:.6f}"
        )

        print(
            f"  F1      : {f1:.6f}"
        )

        print(
            f"  ROC-AUC : {auc:.6f}"
        )

        results.append({
            "target": target,
            "model": model_name,
            "accuracy": accuracy,
            "F1": f1,
            "ROC_AUC": auc,
        })

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            pipeline,
            MODEL_DIR
            / f"{target}_{model_name}.joblib"
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "MATTERGEN V1.2 — FORMULA-GROUPED ML BENCHMARK"
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
    # Create formula-grouped split
    # --------------------------------------------------------

    train_df, test_df = create_grouped_split(
        df
    )

    # --------------------------------------------------------
    # Train regression models
    # --------------------------------------------------------

    regression_results = []

    for target in TARGETS:

        results = train_regression(
            train_df,
            test_df,
            target
        )

        regression_results.extend(
            results
        )

    # --------------------------------------------------------
    # Train classification models
    # --------------------------------------------------------

    classification_results = (
        train_classification(
            train_df,
            test_df
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
    print("GROUPED REGRESSION RESULTS")
    print("=" * 60)

    print(
        regression_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 60)
    print("GROUPED CLASSIFICATION RESULTS")
    print("=" * 60)

    print(
        classification_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    regression_df.to_csv(
        MODEL_DIR
        / "regression_results.csv",
        index=False
    )

    classification_df.to_csv(
        MODEL_DIR
        / "classification_results.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print(
        "FORMULA-GROUPED BENCHMARK COMPLETE"
    )
    print("=" * 60)

    print(
        f"\nModels saved to: {MODEL_DIR}"
    )


if __name__ == "__main__":
    main()