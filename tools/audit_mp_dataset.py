import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# MATTERGEN V1.2 — MATERIALS PROJECT DATASET AUDIT
# ============================================================

DATASET_PATH = Path("data/materials_project_raw.csv")


def print_separator():
    print("=" * 60)


def audit_dataset():

    print_separator()
    print("MATTERGEN V1.2 — MATERIALS PROJECT DATASET AUDIT")
    print_separator()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not DATASET_PATH.exists():
        print(f"\nERROR: Dataset not found:")
        print(DATASET_PATH)
        return

    df = pd.read_csv(DATASET_PATH)

    print(f"\nDataset: {DATASET_PATH}")
    print(f"Total rows:    {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nColumns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\n--- Missing Values ---")

    missing = df.isnull().sum()

    print(missing)

    # --------------------------------------------------------
    # Duplicate checks
    # --------------------------------------------------------

    print("\n--- Duplicates ---")

    duplicate_rows = df.duplicated().sum()

    print(f"Duplicate rows:         {duplicate_rows}")

    if "material_id" in df.columns:
        duplicate_ids = df["material_id"].duplicated().sum()
        print(f"Duplicate material IDs: {duplicate_ids}")

    if "formula" in df.columns:
        duplicate_formulas = df["formula"].duplicated().sum()
        unique_formulas = df["formula"].nunique()

        print(f"Duplicate formulas:     {duplicate_formulas}")
        print(f"Unique formulas:        {unique_formulas}")

    # --------------------------------------------------------
    # Data types
    # --------------------------------------------------------

    print("\n--- Data Types ---")

    print(df.dtypes)

    # --------------------------------------------------------
    # Target statistics
    # --------------------------------------------------------

    target_columns = [
        "band_gap_ev",
        "formation_energy_per_atom",
        "energy_above_hull",
        "density",
        "volume",
        "nsites",
        "nelements",
    ]

    print("\n--- Target / Feature Statistics ---")

    for column in target_columns:

        if column not in df.columns:
            continue

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        print(f"\n{column}")

        print(f"  Min:     {series.min():.6f}")
        print(f"  Max:     {series.max():.6f}")
        print(f"  Mean:    {series.mean():.6f}")
        print(f"  Median:  {series.median():.6f}")
        print(f"  Std:     {series.std():.6f}")
        print(f"  Unique:  {series.nunique()}")

    # --------------------------------------------------------
    # Stability distribution
    # --------------------------------------------------------

    print("\n--- Stability Distribution ---")

    if "is_stable" in df.columns:

        print(df["is_stable"].value_counts(dropna=False))

        stable_count = (
            df["is_stable"]
            .fillna(False)
            .astype(bool)
            .sum()
        )

        print(f"\nStable materials: {stable_count}")
        print(
            f"Stable percentage: "
            f"{stable_count / len(df) * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Physical sanity checks
    # --------------------------------------------------------

    print("\n--- Physical Sanity Checks ---")

    if "band_gap_ev" in df.columns:

        invalid_band_gap = (
            df["band_gap_ev"] < 0
        ).sum()

        print(
            f"Negative band gaps: "
            f"{invalid_band_gap}"
        )

    if "density" in df.columns:

        invalid_density = (
            df["density"] <= 0
        ).sum()

        print(
            f"Non-positive densities: "
            f"{invalid_density}"
        )

    if "volume" in df.columns:

        invalid_volume = (
            df["volume"] <= 0
        ).sum()

        print(
            f"Non-positive volumes: "
            f"{invalid_volume}"
        )

    if "nsites" in df.columns:

        invalid_nsites = (
            df["nsites"] <= 0
        ).sum()

        print(
            f"Non-positive site counts: "
            f"{invalid_nsites}"
        )

    if "nelements" in df.columns:

        invalid_nelements = (
            df["nelements"] <= 0
        ).sum()

        print(
            f"Non-positive element counts: "
            f"{invalid_nelements}"
        )

    # --------------------------------------------------------
    # Formation energy checks
    # --------------------------------------------------------

    print("\n--- Formation Energy Check ---")

    if "formation_energy_per_atom" in df.columns:

        positive_formation = (
            df["formation_energy_per_atom"] > 0
        ).sum()

        zero_formation = (
            df["formation_energy_per_atom"] == 0
        ).sum()

        negative_formation = (
            df["formation_energy_per_atom"] < 0
        ).sum()

        print(
            f"Positive: {positive_formation}"
        )

        print(
            f"Zero:     {zero_formation}"
        )

        print(
            f"Negative: {negative_formation}"
        )

    # --------------------------------------------------------
    # Energy above hull distribution
    # --------------------------------------------------------

    print("\n--- Energy Above Hull Check ---")

    if "energy_above_hull" in df.columns:

        negative_hull = (
            df["energy_above_hull"] < 0
        ).sum()

        zero_hull = (
            np.isclose(
                df["energy_above_hull"],
                0
            )
        ).sum()

        low_hull = (
            df["energy_above_hull"] <= 0.05
        ).sum()

        print(
            f"Negative values:        {negative_hull}"
        )

        print(
            f"Approximately zero:     {zero_hull}"
        )

        print(
            f"<= 0.05 eV/atom:        {low_hull}"
        )

    # --------------------------------------------------------
    # Extreme values
    # --------------------------------------------------------

    print("\n--- Extreme Values ---")

    for column in [
        "band_gap_ev",
        "formation_energy_per_atom",
        "energy_above_hull",
        "density",
        "volume",
    ]:

        if column not in df.columns:
            continue

        print(f"\n{column}")

        print(
            "  Lowest 5:"
        )

        print(
            df.nsmallest(
                5,
                column
            )[
                ["material_id", "formula", column]
            ].to_string(index=False)
        )

        print(
            "\n  Highest 5:"
        )

        print(
            df.nlargest(
                5,
                column
            )[
                ["material_id", "formula", column]
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # Material complexity
    # --------------------------------------------------------

    print("\n--- Material Complexity ---")

    if "nelements" in df.columns:

        print(
            "\nNumber of elements distribution:"
        )

        print(
            df["nelements"]
            .value_counts()
            .sort_index()
            .head(20)
        )

    if "nsites" in df.columns:

        print(
            "\nCrystal sites statistics:"
        )

        print(
            df["nsites"].describe()
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print_separator()
    print("AUDIT COMPLETE")
    print_separator()

    print(
        f"\nFinal dataset size: "
        f"{len(df)} rows"
    )

    print(
        "\nNo data has been modified."
    )

    print(
        "This script only reads and analyzes the dataset."
    )

    print_separator()


if __name__ == "__main__":
    audit_dataset()