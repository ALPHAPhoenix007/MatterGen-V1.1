import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# MATTERGEN V1.2 — ADVANCED FEATURE IMPORTANCE
# ============================================================

INPUT_PATH = Path(
    "data/materials_project_ml_advanced.csv"
)

OUTPUT_DIR = Path(
    "models/mp_v1_2/advanced_grouped"
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
    print("MATTERGEN V1.2 — ADVANCED FEATURE IMPORTANCE")
    print("=" * 60)

    print("\nLoading advanced ML dataset...")

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
    # Feature importance
    # --------------------------------------------------------

    all_results = []

    for target in TARGETS:

        print("\n" + "=" * 60)
        print(
            f"TARGET: {target}"
        )
        print("=" * 60)

        y = pd.to_numeric(
            df[target],
            errors="coerce"
        )

        valid = y.notna()

        X_target = X.loc[
            valid
        ].copy()

        y_target = y.loc[
            valid
        ]

        X_target = X_target.fillna(
            X_target.median()
        )

        model = RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt",
        )

        print(
            "\nTraining feature-importance model..."
        )

        model.fit(
            X_target,
            y_target
        )

        importance = pd.Series(
            model.feature_importances_,
            index=feature_columns
        ).sort_values(
            ascending=False
        )

        print(
            "\nTop 20 features:"
        )

        for rank, (
            feature,
            value
        ) in enumerate(
            importance.head(20).items(),
            start=1
        ):

            print(
                f"{rank:2d}. "
                f"{feature:<38} "
                f"{value:.6f}"
            )

            all_results.append({
                "target": target,
                "feature": feature,
                "importance": value,
            })

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / "advanced_feature_importance.csv"
    )

    pd.DataFrame(
        all_results
    ).to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("ADVANCED FEATURE ANALYSIS COMPLETE")
    print("=" * 60)

    print(
        f"\nOutput: {output_path}"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()