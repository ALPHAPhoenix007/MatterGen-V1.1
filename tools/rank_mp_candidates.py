import pandas as pd
import numpy as np

from pathlib import Path


# ============================================================
# MATTERGEN V1.2 — MATERIALS PROJECT CANDIDATE RANKER
# ============================================================

INPUT_PATH = Path(
    "data/materials_project_ml_advanced.csv"
)


def rank_candidates(
    df,
    target_band_gap=None,
    max_hull=None,
    max_formation_energy=None,
    stable_only=False,
    top_n=10,
):

    candidates = df.copy()

    # --------------------------------------------------------
    # Apply constraints
    # --------------------------------------------------------

    if max_hull is not None:

        candidates = candidates[
            candidates["energy_above_hull"]
            <= max_hull
        ]

    if max_formation_energy is not None:

        candidates = candidates[
            candidates["formation_energy_per_atom"]
            <= max_formation_energy
        ]

    if stable_only:

        candidates = candidates[
            candidates["is_stable"] == True
        ]

    # --------------------------------------------------------
    # Band-gap ranking
    # --------------------------------------------------------

    if target_band_gap is not None:

        candidates = candidates.copy()

        candidates["band_gap_error"] = (
            candidates["band_gap_ev"]
            - target_band_gap
        ).abs()

        candidates = candidates.sort_values(
            [
                "band_gap_error",
                "energy_above_hull",
            ]
        )

    else:

        # If no band-gap target is supplied,
        # prioritize thermodynamic stability.

        candidates = candidates.sort_values(
            [
                "energy_above_hull",
                "formation_energy_per_atom",
            ]
        )

    return candidates.head(
        top_n
    )


def main():

    print("=" * 60)
    print(
        "MATTERGEN V1.2 — CANDIDATE RANKER"
    )
    print("=" * 60)

    print(
        "\nLoading Materials Project dataset..."
    )

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False
    )

    print(
        f"Materials available: {len(df)}"
    )

    # --------------------------------------------------------
    # Example search
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SEARCH TARGET")
    print("=" * 60)

    target_band_gap = 1.5
    max_hull = 0.05
    stable_only = True
    top_n = 10

    print(
        f"\nTarget band gap: {target_band_gap} eV"
    )

    print(
        f"Maximum energy above hull: "
        f"{max_hull} eV/atom"
    )

    print(
        f"Stable materials only: "
        f"{stable_only}"
    )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    results = rank_candidates(
        df=df,
        target_band_gap=target_band_gap,
        max_hull=max_hull,
        stable_only=stable_only,
        top_n=top_n,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TOP CANDIDATES")
    print("=" * 60)

    if len(results) == 0:

        print(
            "\nNo materials matched the requested constraints."
        )

        return

    display_columns = [
        "material_id",
        "formula",
        "band_gap_ev",
        "formation_energy_per_atom",
        "energy_above_hull",
        "is_stable",
        "density",
        "nsites",
        "nelements",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 60)

    print(
        f"Candidates returned: {len(results)}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()