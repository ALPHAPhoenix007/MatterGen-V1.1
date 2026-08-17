from deltalake import DeltaTable
import pandas as pd
import json
from pathlib import Path

from pymatgen.core import Structure, Lattice


# ============================================================
# MATTERGEN V1.2 — CRYSTAL STRUCTURE EXPORTER
# ============================================================


def make_json_safe(value):
    """
    Recursively convert NumPy values into normal Python
    values that json.dumps() can serialize.
    """

    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            key: make_json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(val)
            for val in value
        ]

    return value


CLEAN_DATASET = Path(
    "data/materials_project_clean.csv"
)

MP_DATASET_PATH = Path(
    r"C:\Users\dangr\mp_datasets\build\collections\summary"
)

OUTPUT_PATH = Path(
    "data/materials_project_structures.jsonl"
)


def convert_structure(raw):

    """
    Convert the Arrow/NumPy representation from the local
    Materials Project dataset into a clean pymatgen Structure.
    """

    lattice_data = raw["lattice"]

    # --------------------------------------------------------
    # Lattice matrix
    # --------------------------------------------------------

    matrix = lattice_data["matrix"]

    matrix = [
        [float(x) for x in row]
        for row in matrix
    ]

    lattice = Lattice(matrix)

    # --------------------------------------------------------
    # Atomic sites
    # --------------------------------------------------------

    species = []
    frac_coords = []
    site_properties = {}

    for site in raw["sites"]:

        # ----------------------------------------------------
        # Species
        # ----------------------------------------------------

        site_species = site["species"]

        site_element = site_species[0]["element"]

        species.append(site_element)

        # ----------------------------------------------------
        # Fractional coordinates
        # ----------------------------------------------------

        coords = [
            float(x)
            for x in site["abc"]
        ]

        frac_coords.append(coords)

        # ----------------------------------------------------
        # Site properties
        # ----------------------------------------------------

        properties = site.get(
            "properties",
            {}
        )

        for key, value in properties.items():

            if value is None:
                continue

            if key not in site_properties:
                site_properties[key] = []

            site_properties[key].append(value)

    # --------------------------------------------------------
    # Keep only complete site-property arrays
    # --------------------------------------------------------

    valid_properties = {}

    number_of_sites = len(frac_coords)

    for key, values in site_properties.items():

        if len(values) == number_of_sites:

            valid_properties[key] = values

    # --------------------------------------------------------
    # Build pymatgen Structure
    # --------------------------------------------------------

    structure = Structure(
        lattice=lattice,
        species=species,
        coords=frac_coords,
        coords_are_cartesian=False,
        site_properties=valid_properties
    )

    return structure


def main():

    print("=" * 60)
    print(
        "MATTERGEN V1.2 — CRYSTAL STRUCTURE EXPORTER"
    )
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
    # Load Materials Project structures
    # --------------------------------------------------------

    print(
        "\nLoading Materials Project structures..."
    )

    table = DeltaTable(
        str(MP_DATASET_PATH)
    )

    arrow_table = table.to_pyarrow_table(
        columns=[
            "material_id",
            "structure"
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

    structures = structures[
        structures["structure"].notna()
    ].copy()

    print(
        f"Structures after removing missing: "
        f"{len(structures)}"
    )

    # --------------------------------------------------------
    # Write JSONL
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nWriting structure file...")

    written = 0
    failed = 0

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        for _, row in structures.iterrows():

            material_id = str(
                row["material_id"]
            )

            try:

                structure = convert_structure(
                    row["structure"]
                )

                structure_dict = make_json_safe(
                    structure.as_dict()
                )

                record = {
                    "material_id": material_id,
                    "structure": structure_dict
                }

                f.write(
                    json.dumps(
                        record,
                        separators=(",", ":")
                    )
                    + "\n"
                )

                written += 1

            except Exception as e:

                failed += 1

                print(
                    f"\nFailed: {material_id}"
                )

                print(
                    f"Reason: {e}"
                )

                continue

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
        f"Structures failed: {failed}"
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