import pandas as pd
import numpy as np

from pathlib import Path
from pymatgen.core import Composition, Element


# ============================================================
# MATTERGEN V1.2 — MATERIALS PROJECT FEATURE ENGINEERING
# ============================================================

INPUT_PATH = Path(
    "data/materials_project_clean.csv"
)

OUTPUT_PATH = Path(
    "data/materials_project_features.csv"
)


def safe_stats(values, prefix):

    values = np.asarray(
        values,
        dtype=float
    )

    return {
        f"{prefix}_mean": np.mean(values),
        f"{prefix}_min": np.min(values),
        f"{prefix}_max": np.max(values),
        f"{prefix}_std": np.std(values),
        f"{prefix}_range": np.max(values) - np.min(values),
    }


def get_composition_features(formula):

    try:

        composition = Composition(
            formula
        )

        elements = list(
            composition.elements
        )

        amounts = np.array(
            [
                composition.get_atomic_fraction(
                    element
                )
                for element in elements
            ]
        )

        # ----------------------------------------------------
        # Atomic mass
        # ----------------------------------------------------

        atomic_masses = [
            float(element.atomic_mass)
            for element in elements
        ]

        # ----------------------------------------------------
        # Electronegativity
        # ----------------------------------------------------

        electronegativities = []

        for element in elements:

            value = element.X

            if value is not None:

                electronegativities.append(
                    float(value)
                )

        # ----------------------------------------------------
        # Atomic radius
        # ----------------------------------------------------

        atomic_radii = []

        for element in elements:

            value = element.atomic_radius

            if value is not None:

                atomic_radii.append(
                    float(value)
                )

        # ----------------------------------------------------
        # Weighted averages
        # ----------------------------------------------------

        mass_weighted = np.average(
            atomic_masses,
            weights=amounts
        )

        if len(electronegativities) == len(elements):

            electronegativity_weighted = np.average(
                electronegativities,
                weights=amounts
            )

        else:

            electronegativity_weighted = np.nan

        if len(atomic_radii) == len(elements):

            radius_weighted = np.average(
                atomic_radii,
                weights=amounts
            )

        else:

            radius_weighted = np.nan

        features = {

            "composition_nelements":
                len(elements),

            "avg_atomic_mass":
                mass_weighted,

            "avg_electronegativity":
                electronegativity_weighted,

            "avg_atomic_radius":
                radius_weighted,
        }

        # ----------------------------------------------------
        # Statistical descriptors
        # ----------------------------------------------------

        features.update(
            safe_stats(
                atomic_masses,
                "atomic_mass"
            )
        )

        if electronegativities:

            features.update(
                safe_stats(
                    electronegativities,
                    "electronegativity"
                )
            )

        else:

            features.update({
                "electronegativity_mean": np.nan,
                "electronegativity_min": np.nan,
                "electronegativity_max": np.nan,
                "electronegativity_std": np.nan,
                "electronegativity_range": np.nan,
            })

        if atomic_radii:

            features.update(
                safe_stats(
                    atomic_radii,
                    "atomic_radius"
                )
            )

        else:

            features.update({
                "atomic_radius_mean": np.nan,
                "atomic_radius_min": np.nan,
                "atomic_radius_max": np.nan,
                "atomic_radius_std": np.nan,
                "atomic_radius_range": np.nan,
            })

        return features

    except Exception:

        return {}


def main():

    print("=" * 60)
    print(
        "MATTERGEN V1.2 — COMPOSITION FEATURE ENGINEERING"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(
        INPUT_PATH,
        keep_default_na=False
    )

    print(
        f"Materials loaded: {len(df)}"
    )

    # --------------------------------------------------------
    # Generate features
    # --------------------------------------------------------

    print(
        "\nGenerating composition features..."
    )

    feature_rows = []

    failed = 0

    for index, formula in enumerate(
        df["formula"]
    ):

        features = get_composition_features(
            formula
        )

        if not features:

            failed += 1

        feature_rows.append(
            features
        )

        if (index + 1) % 10000 == 0:

            print(
                f"  Processed: {index + 1}"
            )

    feature_df = pd.DataFrame(
        feature_rows
    )

    # --------------------------------------------------------
    # Existing structural features
    # --------------------------------------------------------

    feature_df["density"] = df[
        "density"
    ]

    feature_df["volume"] = df[
        "volume"
    ]

    feature_df["nsites"] = df[
        "nsites"
    ]

    feature_df["nelements"] = df[
        "nelements"
    ]

    feature_df["volume_per_atom"] = (
        df["volume"] /
        df["nsites"]
    )

    # --------------------------------------------------------
    # IDs and targets
    # --------------------------------------------------------

    output_df = pd.concat(
        [
            df[
                [
                    "material_id",
                    "formula",
                    "band_gap_ev",
                    "formation_energy_per_atom",
                    "energy_above_hull",
                    "is_stable",
                ]
            ],
            feature_df,
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Remove duplicate columns
    # --------------------------------------------------------

    output_df = output_df.loc[
        :,
        ~output_df.columns.duplicated()
    ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)

    print(
        f"\nMaterials: {len(output_df)}"
    )

    print(
        f"Features: {len(output_df.columns)}"
    )

    print(
        f"Failed formulas: {failed}"
    )

    print("\nGenerated columns:")

    for column in output_df.columns:

        print(
            f"  - {column}"
        )

    print(
        f"\nOutput: {OUTPUT_PATH}"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()