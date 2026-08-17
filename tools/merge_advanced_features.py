import pandas as pd
from pathlib import Path


# ============================================================
# MATTERGEN V1.2 — ADVANCED ML DATASET BUILDER
# ============================================================

BASE_PATH = Path(
    "data/materials_project_ml_reduced.csv"
)

ADVANCED_PATH = Path(
    "data/materials_project_advanced_features.csv"
)

OUTPUT_PATH = Path(
    "data/materials_project_ml_advanced.csv"
)


ADVANCED_FEATURES = [
    "stoich_max_fraction",
    "stoich_min_fraction",
    "stoich_fraction_std",
    "stoich_fraction_range",
    "stoich_entropy",
    "stoich_normalized_entropy",
    "dominant_element_fraction",
    "effective_element_count",
    "composition_element_count",
]


def main():

    print("=" * 60)
    print(
        "MATTERGEN V1.2 — ADVANCED ML DATASET BUILDER"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading reduced ML dataset...")

    base_df = pd.read_csv(
        BASE_PATH,
        low_memory=False
    )

    print(
        f"Base rows: {len(base_df)}"
    )

    print(
        f"Base columns: {len(base_df.columns)}"
    )

    print("\nLoading advanced descriptors...")

    advanced_df = pd.read_csv(
        ADVANCED_PATH,
        keep_default_na=False
    )

    print(
        f"Advanced rows: {len(advanced_df)}"
    )

    # --------------------------------------------------------
    # Verify IDs
    # --------------------------------------------------------

    print("\nChecking material IDs...")

    base_duplicates = (
        base_df["material_id"]
        .duplicated()
        .sum()
    )

    advanced_duplicates = (
        advanced_df["material_id"]
        .duplicated()
        .sum()
    )

    print(
        f"Base duplicate IDs: {base_duplicates}"
    )

    print(
        f"Advanced duplicate IDs: "
        f"{advanced_duplicates}"
    )

    # --------------------------------------------------------
    # Keep only required advanced columns
    # --------------------------------------------------------

    advanced_df = advanced_df[
        [
            "material_id",
            "formula",
        ]
        + ADVANCED_FEATURES
    ]

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    print("\nMerging datasets...")

    df = base_df.merge(
        advanced_df,
        on="material_id",
        how="inner",
        suffixes=(
            "",
            "_advanced"
        )
    )

    print(
        f"Merged rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Verify formula consistency
    # --------------------------------------------------------

    formula_mismatch = (
        df["formula"]
        != df["formula_advanced"]
    ).sum()

    print(
        f"Formula mismatches: "
        f"{formula_mismatch}"
    )

    if "formula_advanced" in df.columns:

        df = df.drop(
            columns=[
                "formula_advanced"
            ]
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing advanced features:")

    print(
        df[
            ADVANCED_FEATURES
        ].isna().sum()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    identifier_columns = [
        "material_id",
        "formula",
    ]

    target_columns = [
        "band_gap_ev",
        "formation_energy_per_atom",
        "energy_above_hull",
        "is_stable",
    ]

    feature_count = (
        len(df.columns)
        - len(identifier_columns)
        - len(target_columns)
    )

    print("\n" + "=" * 60)
    print(
        "ADVANCED ML DATASET CREATED"
    )
    print("=" * 60)

    print(
        f"\nRows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"Features: {feature_count}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print("\nAdvanced features added:")

    for feature in ADVANCED_FEATURES:

        print(
            f"  - {feature}"
        )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()