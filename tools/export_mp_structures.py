from deltalake import DeltaTable
import pandas as pd
import json
from pathlib import Path


# ============================================================
# MATTERGEN V1.2 — CRYSTAL STRUCTURE EXPORTER
# ============================================================

CLEAN_DATASET = Path(
    "data/materials_project_clean.csv"
)

MP_DATASET_PATH = Path(
    r"C:\Users\dangr\mp_datasets\build\collections\summary"
)

OUTPUT_PATH = Path(
    "data/materials_project_structures.jsonl"
)


def main():

    print("=" * 60)
    print("MATTERGEN V1.2 — CRYSTAL STRUCTURE EXPORTER")
    print("=" * 60)

    # --------------------------------------------------------
    # Load clean material IDs
    # --------------------------------------------------------

    print("\nLoading clean dataset...")

    df = pd.read_csv(
        CLEAN_DATASET,
        usecols=["material_id"]
    )

    material_ids = set(
        df["material_id"].astype(str)
    )

    print(
        f"Clean materials: {len(material_ids)}"
    )

    # --------------------------------------------------------
    # Load structures
    # --------------------------------------------------------

    print("\nLoading Materials Project structures...")

    table = DeltaTable(
        str(MP_DATASET_PATH)
    )

    arrow_table = table.to_pyarrow_table(
        columns=[
            "material_id",
            "structure",
        ]
    )

    structures = arrow_table.to_pandas()

    print(
        f"Structures available: {len(structures)}"
    )

    # --------------------------------------------------------
    # Filter to clean dataset
    # --------------------------------------------------------

    structures = structures[
        structures["material_id"]
        .astype(str)
        .isin(material_ids)
    ].copy()

    print(
        f"Structures matching clean dataset: "
        f"{len(structures)}"
    )

    # --------------------------------------------------------
    # Remove missing structures
    # --------------------------------------------------------

    missing_count = structures[
        "structure"
    ].isna().sum()

    print(
        f"Missing structures: {missing_count}"
    )

    structures = structures[
        structures["structure"].notna()
    ].copy()

    # --------------------------------------------------------
    # Write JSONL
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nWriting structure file...")

    written = 0

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        for _, row in structures.iterrows():

            structure = row["structure"]

            # Materials Project's local dataset already
            # provides the structure as a dictionary.
            if isinstance(structure, dict):
                structure_dict = structure

            else:
                # Safety fallback in case a pymatgen
                # Structure object is returned.
                structure_dict = structure.as_dict()

            record = {
                "material_id": str(
                    row["material_id"]
                ),
                "structure": structure_dict,
            }

            f.write(
                json.dumps(
                    record,
                    separators=(",", ":"),
                    default=str
                )
                + "\n"
            )

            written += 1

            if written % 10000 == 0:
                print(
                    f"  Written: {written}"
                )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STRUCTURE EXPORT COMPLETE")
    print("=" * 60)

    print(
        f"\nStructures written: {written}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print(
        f"Clean dataset materials: "
        f"{len(material_ids)}"
    )

    print(
        f"Structures missing: "
        f"{len(material_ids) - written}"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()