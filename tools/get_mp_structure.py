import json
import sys
import re
from pathlib import Path

import numpy as np
from pymatgen.core import Structure


# ============================================================
# MATTERGEN V1.2 — LOCAL CRYSTAL STRUCTURE RETRIEVER
# ============================================================

STRUCTURE_PATH = Path(
    "data/materials_project_structures.jsonl"
)


def find_structure(material_id):

    with STRUCTURE_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            record = json.loads(line)

            if record.get("material_id") == material_id:
                return record

    return None


def parse_matrix(value):

    if isinstance(value, str):

        numbers = re.findall(
            r"[-+]?(?:\d+\.\d*|\.\d+|\d+)"
            r"(?:e[-+]?\d+)?",
            value,
            flags=re.IGNORECASE
        )

        if len(numbers) != 9:

            raise ValueError(
                f"Could not parse lattice matrix. "
                f"Found {len(numbers)} numbers."
            )

        return [
            [
                float(numbers[0]),
                float(numbers[1]),
                float(numbers[2])
            ],
            [
                float(numbers[3]),
                float(numbers[4]),
                float(numbers[5])
            ],
            [
                float(numbers[6]),
                float(numbers[7]),
                float(numbers[8])
            ]
        ]

    if isinstance(value, np.ndarray):

        return value.astype(float).tolist()

    return value


def parse_pbc(value):

    if isinstance(value, str):

        values = re.findall(
            r"True|False",
            value
        )

        if len(values) != 3:

            raise ValueError(
                f"Could not parse pbc: {value}"
            )

        return tuple(
            value == "True"
            for value in values
        )

    if isinstance(value, np.ndarray):

        return tuple(
            bool(x)
            for x in value
        )

    if isinstance(value, list):

        return tuple(
            bool(x)
            for x in value
        )

    if isinstance(value, tuple):

        return value

    return (
        True,
        True,
        True
    )


def main():

    print("=" * 60)
    print(
        "MATTERGEN V1.2 — CRYSTAL STRUCTURE RETRIEVER"
    )
    print("=" * 60)

    if len(sys.argv) < 2:

        print("\nUsage:")
        print(
            "python tools/get_mp_structure.py <material_id>"
        )

        return

    material_id = sys.argv[1]

    print(
        f"\nSearching for: {material_id}"
    )

    record = find_structure(
        material_id
    )

    if record is None:

        print(
            "\nStructure not found."
        )

        return

    print(
        "\nStructure found!"
    )

    structure_data = record["structure"]

    # --------------------------------------------------------
    # Normalize lattice
    # --------------------------------------------------------

    lattice = structure_data["lattice"]

    lattice["matrix"] = parse_matrix(
        lattice["matrix"]
    )

    lattice["pbc"] = parse_pbc(
        lattice.get("pbc")
    )

    # --------------------------------------------------------
    # Convert to pymatgen Structure
    # --------------------------------------------------------

    structure = Structure.from_dict(
        structure_data
    )

    # --------------------------------------------------------
    # Display information
    # --------------------------------------------------------

    print(
        f"Material ID: {material_id}"
    )

    print(
        f"Formula: {structure.formula}"
    )

    print(
        f"Reduced formula: "
        f"{structure.composition.reduced_formula}"
    )

    print(
        f"Number of sites: "
        f"{len(structure)}"
    )

    print(
        f"Lattice volume: "
        f"{structure.volume:.4f} Å³"
    )

    print("\nLattice parameters:")

    print(
        f"  a = {structure.lattice.a:.4f} Å"
    )

    print(
        f"  b = {structure.lattice.b:.4f} Å"
    )

    print(
        f"  c = {structure.lattice.c:.4f} Å"
    )

    print(
        f"  α = {structure.lattice.alpha:.4f}°"
    )

    print(
        f"  β = {structure.lattice.beta:.4f}°"
    )

    print(
        f"  γ = {structure.lattice.gamma:.4f}°"
    )

    # --------------------------------------------------------
    # Display atomic sites
    # --------------------------------------------------------

    print("\nFirst 5 atomic sites:")

    for i, site in enumerate(
        structure.sites[:5],
        start=1
    ):

        print(
            f"  {i}. "
            f"{site.species_string} "
            f"fractional={site.frac_coords}"
        )

    # --------------------------------------------------------
    # Save cleaned structure
    # --------------------------------------------------------

    output_dir = Path(
        "data/structures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / f"{material_id}.json"
    )

    cleaned_record = {
        "material_id": material_id,
        "structure": structure.as_dict()
    }

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            cleaned_record,
            file,
            indent=2
        )

    print(
        f"\nSaved cleaned structure:"
    )

    print(
        output_path
    )

    print("\n" + "=" * 60)

    print(
        "STRUCTURE RETRIEVAL COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()