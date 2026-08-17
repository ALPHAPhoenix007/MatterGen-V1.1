import pandas as pd
from pathlib import Path


# ============================================================
# MATTERGEN V1.2 — FEATURE MERGER
# ============================================================

COMPOSITION_PATH = Path(
    "data/materials_project_features.csv"
)

STRUCTURE_PATH = Path(
    "data/materials_project_structure_features.csv"
)

OUTPUT_PATH = Path(
    "data/materials_project_ml.csv"
)


def main():

    print("=" * 60)
    print("MATTERGEN V1.2 — ML FEATURE DATASET BUILDER")
    print("=" * 60)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading composition features...")

    composition_df = pd.read_csv(
        COMPOSITION_PATH,
        keep_default_na=False
    )

    print(
        f"Composition rows: {len(composition_df)}"
    )

    print("\nLoading structure features...")

    structure_df = pd.read_csv(
        STRUCTURE_PATH
    )

    print(
        f"Structure rows: {len(structure_df)}"
    )

    # --------------------------------------------------------
    # Check IDs
    # --------------------------------------------------------

    print("\nChecking material IDs...")

    print(
        "Composition duplicate IDs:",
        composition_df["material_id"].duplicated().sum()
    )

    print(
        "Structure duplicate IDs:",
        structure_df["material_id"].duplicated().sum()
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    print("\nMerging datasets...")

    df = composition_df.merge(
        structure_df,
        on="material_id",
        how="inner",
        suffixes=(
            "",
            "_structure"
        )
    )

    print(
        f"Merged rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Check matching
    # --------------------------------------------------------

    expected = len(
        composition_df
    )

    if len(df) != expected:

        print(
            "\nWARNING:"
        )

        print(
            f"Expected {expected} rows "
            f"but got {len(df)}."
        )

    # --------------------------------------------------------
    # Remove duplicate columns
    # --------------------------------------------------------

    df = df.loc[
        :,
        ~df.columns.duplicated()
    ]

    # --------------------------------------------------------
    # Check missing values
    # --------------------------------------------------------

    print("\nMissing values:")

    missing = df.isna().sum()

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    if len(missing) == 0:

        print(
            "No missing values."
        )

    else:

        print(
            missing
        )

    # --------------------------------------------------------
    # Feature list
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

    feature_columns = [
        column
        for column in df.columns
        if column not in (
            identifier_columns
            + target_columns
        )
    ]

    # --------------------------------------------------------
    # Check duplicate feature names
    # --------------------------------------------------------

    print(
        f"\nTotal columns: {len(df.columns)}"
    )

    print(
        f"Feature columns: {len(feature_columns)}"
    )

    print("\nFeature columns:")

    for column in feature_columns:

        print(
            f"  - {column}"
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

    print("\n" + "=" * 60)
    print("ML DATASET CREATED")
    print("=" * 60)

    print(
        f"\nRows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print("\nTargets:")

    for target in target_columns:

        print(
            f"  - {target}"
        )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()