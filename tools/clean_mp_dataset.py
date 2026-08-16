from deltalake import DeltaTable
import pandas as pd
from pathlib import Path


# ============================================================
# MATTERGEN V1.2 — MATERIALS PROJECT DATASET CLEANER
# ============================================================

RAW_CSV = Path("data/materials_project_raw.csv")
OUTPUT_CSV = Path("data/materials_project_clean.csv")

MP_DATASET_PATH = Path(
    r"C:\Users\dangr\mp_datasets\build\collections\summary"
)


def main():

    print("=" * 60)
    print("MATTERGEN V1.2 — MATERIALS PROJECT DATASET CLEANER")
    print("=" * 60)

    # --------------------------------------------------------
    # Load raw dataset
    # --------------------------------------------------------

    print("\nLoading raw dataset...")

    df = pd.read_csv(RAW_CSV)

    print(f"Original rows: {len(df)}")

    # --------------------------------------------------------
    # Remove exact duplicate rows
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates()

    print(
        f"Exact duplicates removed: "
        f"{before - len(df)}"
    )

    # --------------------------------------------------------
    # Remove duplicate material IDs
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=["material_id"],
        keep="first"
    )

    print(
        f"Duplicate material IDs removed: "
        f"{before - len(df)}"
    )

    # --------------------------------------------------------
    # Recover missing formulas from Materials Project
    # --------------------------------------------------------

    missing_formula = df["formula"].isna()

    missing_count = missing_formula.sum()

    print(
        f"\nMissing formulas before recovery: "
        f"{missing_count}"
    )

    if missing_count > 0:

        print("Recovering formulas from local Materials Project data...")

        table = DeltaTable(
            str(MP_DATASET_PATH)
        )

        structure_data = table.to_pyarrow_table(
            columns=[
                "material_id",
                "formula_pretty",
            ]
        ).to_pandas()

        structure_data = structure_data.drop_duplicates(
            subset=["material_id"]
        )

        formula_map = dict(
            zip(
                structure_data["material_id"],
                structure_data["formula_pretty"]
            )
        )

        df.loc[
            missing_formula,
            "formula"
        ] = df.loc[
            missing_formula,
            "material_id"
        ].map(formula_map)

    # --------------------------------------------------------
    # Check formulas again
    # --------------------------------------------------------

    remaining_missing = df["formula"].isna().sum()

    print(
        f"Missing formulas after recovery: "
        f"{remaining_missing}"
    )

    # --------------------------------------------------------
    # Remove rows with missing critical targets
    # --------------------------------------------------------

    target_columns = [
        "band_gap_ev",
        "formation_energy_per_atom",
        "energy_above_hull",
    ]

    before = len(df)

    df = df.dropna(
        subset=target_columns
    )

    print(
        f"Rows removed due to missing targets: "
        f"{before - len(df)}"
    )

    # --------------------------------------------------------
    # Numerical sanity checks
    # --------------------------------------------------------

    print("\nChecking numerical validity...")

    invalid_band_gap = (
        df["band_gap_ev"] < 0
    )

    invalid_density = (
        df["density"] <= 0
    )

    invalid_volume = (
        df["volume"] <= 0
    )

    invalid_nsites = (
        df["nsites"] <= 0
    )

    invalid_nelements = (
        df["nelements"] <= 0
    )

    invalid_hull = (
        df["energy_above_hull"] < -1e-8
    )

    invalid_rows = (
        invalid_band_gap
        | invalid_density
        | invalid_volume
        | invalid_nsites
        | invalid_nelements
        | invalid_hull
    )

    print(
        f"Rows failing physical sanity checks: "
        f"{invalid_rows.sum()}"
    )

    if invalid_rows.sum() > 0:

        print(
            "Removing physically invalid rows..."
        )

        df = df.loc[
            ~invalid_rows
        ].copy()

    # --------------------------------------------------------
    # Keep scientifically valid extreme values
    # --------------------------------------------------------

    print(
        "\nNo arbitrary outlier removal performed."
    )

    print(
        "Scientifically unusual materials are retained."
    )

    # --------------------------------------------------------
    # Sort for reproducibility
    # --------------------------------------------------------

    df = df.sort_values(
        by="material_id"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save clean dataset
    # --------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)

    print(
        f"\nFinal rows: {len(df)}"
    )

    print(
        f"Final columns: {len(df.columns)}"
    )

    print(
        f"Output: {OUTPUT_CSV}"
    )

    print("\nColumns:")

    for column in df.columns:
        print(f"  - {column}")

    print("\nFinal missing values:")

    print(
        df.isnull().sum()
    )

    print("\n" + "=" * 60)
    print("V1.2 CLEAN DATASET READY")
    print("=" * 60)


if __name__ == "__main__":
    main()