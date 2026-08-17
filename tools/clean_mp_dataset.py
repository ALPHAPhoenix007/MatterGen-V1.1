from deltalake import DeltaTable
import pandas as pd
from pathlib import Path
from pymatgen.core import Composition


# ============================================================
# MATTERGEN V1.2 — MATERIALS PROJECT DATASET CLEANER
# ============================================================

RAW_CSV = Path("data/materials_project_raw.csv")
OUTPUT_CSV = Path("data/materials_project_clean.csv")

MP_DATASET_PATH = Path(
    r"C:\Users\dangr\mp_datasets\build\collections\summary"
)


def composition_to_formula(value):
    """
    Convert Materials Project composition data such as:

        [('Na', 1.0), ('N', 1.0)]

    into a reduced chemical formula such as:

        NaN
    """

    try:

        if value is None:
            return None

        if not isinstance(value, (list, tuple)):
            return None

        composition_dict = {}

        for item in value:

            if not isinstance(item, (list, tuple)):
                continue

            if len(item) < 2:
                continue

            element = str(item[0])
            amount = float(item[1])

            composition_dict[element] = amount

        if not composition_dict:
            return None

        composition = Composition(
            composition_dict
        )

        return composition.reduced_formula

    except Exception:
        return None


def main():

    print("=" * 60)
    print("MATTERGEN V1.2 — MATERIALS PROJECT DATASET CLEANER")
    print("=" * 60)

    # --------------------------------------------------------
    # Load raw dataset
    # --------------------------------------------------------

    print("\nLoading raw dataset...")

    df = pd.read_csv(
        RAW_CSV
    )

    print(
        f"Original rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Remove exact duplicates
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
    # Recover missing formulas
    # --------------------------------------------------------

    missing_formula = df["formula"].isna()

    missing_count = missing_formula.sum()

    print(
        f"\nMissing formulas before recovery: "
        f"{missing_count}"
    )

    if missing_count > 0:

        print(
            "Recovering formulas from local "
            "Materials Project data..."
        )

        table = DeltaTable(
            str(MP_DATASET_PATH)
        )

        mp_data = table.to_pyarrow_table(
            columns=[
                "material_id",
                "formula_pretty",
                "composition",
            ]
        ).to_pandas()

        mp_data = mp_data.drop_duplicates(
            subset=["material_id"]
        )

        mp_data = mp_data.set_index(
            "material_id"
        )

        recovered = 0

        for index in df.index[missing_formula]:

            material_id = str(
                df.loc[
                    index,
                    "material_id"
                ]
            )

            if material_id not in mp_data.index:
                continue

            row = mp_data.loc[
                material_id
            ]

            # ------------------------------------------------
            # First choice: formula_pretty
            # ------------------------------------------------

            formula = row[
                "formula_pretty"
            ]

            if pd.notna(formula):

                df.loc[
                    index,
                    "formula"
                ] = str(formula)

                recovered += 1

                continue

            # ------------------------------------------------
            # Second choice: composition
            # ------------------------------------------------

            formula = composition_to_formula(
                row["composition"]
            )

            if formula is not None:

                df.loc[
                    index,
                    "formula"
                ] = formula

                recovered += 1

        print(
            f"Formulas recovered: {recovered}"
        )

    # --------------------------------------------------------
    # Verify formula recovery
    # --------------------------------------------------------

    remaining_missing = (
        df["formula"].isna().sum()
    )

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

    print(
        "\nChecking numerical validity..."
    )

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

        df = df.loc[
            ~invalid_rows
        ].copy()

        print(
            "Physically invalid rows removed."
        )

    # --------------------------------------------------------
    # Final formula check
    # --------------------------------------------------------

    if df["formula"].isna().any():

        print(
            "\nWARNING: Some formulas could not be recovered."
        )

        print(
            df[
                df["formula"].isna()
            ][
                ["material_id"]
            ].to_string(
                index=False
            )
        )

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
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        by="material_id"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save
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

    print("\nFinal missing values:")

    print(
        df.isnull().sum()
    )

    print("\n" + "=" * 60)
    print("V1.2 CLEAN DATASET READY")
    print("=" * 60)


if __name__ == "__main__":
    main()