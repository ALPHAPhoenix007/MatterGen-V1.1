import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# MATTERGEN V1.2 — FEATURE IMPORTANCE & CORRELATION ANALYSIS
# ============================================================

INPUT_PATH = Path(
    "data/materials_project_ml.csv"
)

OUTPUT_DIR = Path(
    "models/mp_v1_2"
)


TARGETS = [
    "band_gap_ev",
    "formation_energy_per_atom",
    "energy_above_hull",
]

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


def main():

    print("=" * 60)
    print(
        "MATTERGEN V1.2 — FEATURE ANALYSIS"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading ML dataset...")

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in (
            IDENTIFIER_COLUMNS
            + TARGET_COLUMNS
        )
    ]

    X = df[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    print(
        f"Materials: {len(df)}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Correlation analysis
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TARGET CORRELATIONS")
    print("=" * 60)

    correlation_rows = []

    for target in TARGETS:

        target_values = pd.to_numeric(
            df[target],
            errors="coerce"
        )

        correlations = (
            X.corrwith(
                target_values
            )
            .abs()
            .sort_values(
                ascending=False
            )
        )

        print(
            f"\n{target}"
        )

        for feature, value in correlations.head(10).items():

            print(
                f"  {feature:<35} "
                f"{value:.4f}"
            )

            correlation_rows.append({
                "target": target,
                "feature": feature,
                "absolute_correlation": value,
            })

    correlation_df = pd.DataFrame(
        correlation_rows
    )

    # --------------------------------------------------------
    # Random Forest feature importance
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 60)

    importance_rows = []

    for target in TARGETS:

        print(
            f"\nAnalyzing {target}..."
        )

        y = pd.to_numeric(
            df[target],
            errors="coerce"
        )

        valid = y.notna()

        X_target = X.loc[
            valid
        ]

        y_target = y.loc[
            valid
        ]

        # Median imputation for importance analysis
        X_target = X_target.fillna(
            X_target.median()
        )

        model = RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt",
        )

        model.fit(
            X_target,
            y_target
        )

        importances = pd.Series(
            model.feature_importances_,
            index=feature_columns
        ).sort_values(
            ascending=False
        )

        print(
            f"\nTop features for {target}:"
        )

        for feature, importance in importances.head(15).items():

            print(
                f"  {feature:<35} "
                f"{importance:.6f}"
            )

            importance_rows.append({
                "target": target,
                "feature": feature,
                "importance": importance,
            })

    importance_df = pd.DataFrame(
        importance_rows
    )

    # --------------------------------------------------------
    # Feature-to-feature correlation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("HIGHLY CORRELATED FEATURES")
    print("=" * 60)

    feature_corr = X.corr().abs()

    pairs = []

    for i in range(
        len(feature_corr.columns)
    ):

        for j in range(i + 1, len(feature_corr.columns)):

            feature_a = feature_corr.columns[i]
            feature_b = feature_corr.columns[j]

            value = feature_corr.iloc[
                i,
                j
            ]

            if value >= 0.95:

                pairs.append({
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "absolute_correlation": value,
                })

    pairs_df = pd.DataFrame(
        pairs
    )

    if len(pairs_df) == 0:

        print(
            "\nNo feature pairs above 0.95 correlation."
        )

    else:

        pairs_df = pairs_df.sort_values(
            "absolute_correlation",
            ascending=False
        )

        print(
            f"\nFound {len(pairs_df)} highly correlated pairs."
        )

        print(
            pairs_df.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    correlation_df.to_csv(
        OUTPUT_DIR /
        "feature_target_correlations.csv",
        index=False
    )

    importance_df.to_csv(
        OUTPUT_DIR /
        "feature_importance.csv",
        index=False
    )

    pairs_df.to_csv(
        OUTPUT_DIR /
        "highly_correlated_features.csv",
        index=False
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS COMPLETE")
    print("=" * 60)

    print(
        "\nSaved:"
    )

    print(
        "  - feature_target_correlations.csv"
    )

    print(
        "  - feature_importance.csv"
    )

    print(
        "  - highly_correlated_features.csv"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()