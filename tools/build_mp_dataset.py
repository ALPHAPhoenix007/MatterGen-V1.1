from deltalake import DeltaTable
import pandas as pd
from pathlib import Path


# ============================================================
# MATTERGEN V1.2 — MATERIALS PROJECT DATASET BUILDER
# ============================================================

MP_DATASET_PATH = Path(
    r"C:\Users\dangr\mp_datasets\build\collections\summary"
)

OUTPUT_PATH = Path("data/materials_project_raw.csv")


def main():

    print("=" * 60)
    print("MATTERGEN V1.2 — MATERIALS PROJECT DATASET BUILDER")
    print("=" * 60)

    print("\nLoading local Materials Project dataset...")

    table = DeltaTable(str(MP_DATASET_PATH))

    columns = [
        "material_id",
        "formula_pretty",
        "band_gap",
        "formation_energy_per_atom",
        "energy_above_hull",
        "is_stable",
        "density",
        "volume",
        "nsites",
        "nelements",
    ]

    print("Selecting required fields...")

    arrow_table = table.to_pyarrow_table(
        columns=columns
    )

    df = arrow_table.to_pandas()

    print(f"Records loaded: {len(df)}")

    # --------------------------------------------------------
    # Keep only records with all three primary targets
    # --------------------------------------------------------

    target_columns = [
        "band_gap",
        "formation_energy_per_atom",
        "energy_above_hull",
    ]

    before = len(df)

    df = df.dropna(
        subset=target_columns
    ).copy()

    print(
        f"Records after target filtering: "
        f"{len(df)}"
    )

    print(
        f"Records removed: "
        f"{before - len(df)}"
    )

    # --------------------------------------------------------
    # Rename columns to MatterGen naming convention
    # --------------------------------------------------------

    df = df.rename(
        columns={
            "formula_pretty": "formula",
            "band_gap": "band_gap_ev",
            "formation_energy_per_atom":
                "formation_energy_per_atom",
            "energy_above_hull":
                "energy_above_hull",
        }
    )

    # --------------------------------------------------------
    # Remove deprecated materials
    # --------------------------------------------------------

    # Note:
    # We intentionally do not filter on "deprecated" here
    # because that field was not loaded.
    #
    # We will handle this explicitly in the next dataset
    # cleaning stage.

    # --------------------------------------------------------
    # Remove duplicate material IDs
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["material_id"]
    )

    # --------------------------------------------------------
    # Sort for reproducibility
    # --------------------------------------------------------

    df = df.sort_values(
        by="material_id"
    ).reset_index(drop=True)

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

    print("\nDataset created successfully.")

    print(f"Output: {OUTPUT_PATH}")

    print(f"Final records: {len(df)}")

    print("\nColumns:")

    for column in df.columns:
        print(f"  - {column}")

    print("\n" + "=" * 60)
    print("V1.2 DATASET BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()