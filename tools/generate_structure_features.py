import json
import pandas as pd
from pathlib import Path


# ============================================================
# MATTERGEN V1.2 — CRYSTAL STRUCTURE FEATURE ENGINEERING
# ============================================================

STRUCTURE_PATH = Path(
    "data/materials_project_structures.jsonl"
)

OUTPUT_PATH = Path(
    "data/materials_project_structure_features.csv"
)


def main():

    print("=" * 60)
    print(
        "MATTERGEN V1.2 — CRYSTAL STRUCTURE FEATURES"
    )
    print("=" * 60)

    print(
        "\nReading crystal structures line-by-line..."
    )

    rows = []
    failed = 0
    processed = 0

    with open(
        STRUCTURE_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            try:

                record = json.loads(line)

                material_id = record[
                    "material_id"
                ]

                structure = record[
                    "structure"
                ]

                lattice = structure[
                    "lattice"
                ]

                a = float(
                    lattice["a"]
                )

                b = float(
                    lattice["b"]
                )

                c = float(
                    lattice["c"]
                )

                alpha = float(
                    lattice["alpha"]
                )

                beta = float(
                    lattice["beta"]
                )

                gamma = float(
                    lattice["gamma"]
                )

                volume = float(
                    lattice["volume"]
                )

                sites = structure.get(
                    "sites",
                    []
                )

                nsites = len(sites)

                if nsites > 0:

                    volume_per_atom = (
                        volume / nsites
                    )

                else:

                    volume_per_atom = None

                rows.append({

                    "material_id":
                        material_id,

                    "lattice_a":
                        a,

                    "lattice_b":
                        b,

                    "lattice_c":
                        c,

                    "lattice_alpha":
                        alpha,

                    "lattice_beta":
                        beta,

                    "lattice_gamma":
                        gamma,

                    "structure_volume":
                        volume,

                    "structure_nsites":
                        nsites,

                    "structure_volume_per_atom":
                        volume_per_atom,

                })

                processed += 1

                if processed % 10000 == 0:

                    print(
                        f"  Processed: {processed}"
                    )

            except Exception:

                failed += 1

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Remove duplicate IDs
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["material_id"]
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

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(
        "STRUCTURE FEATURE ENGINEERING COMPLETE"
    )
    print("=" * 60)

    print(
        f"\nStructures processed: {processed}"
    )

    print(
        f"Failed structures: {failed}"
    )

    print(
        f"Final rows: {len(df)}"
    )

    print(
        f"Features: {len(df.columns)}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print("\nColumns:")

    for column in df.columns:

        print(
            f"  - {column}"
        )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()