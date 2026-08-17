import pandas as pd
import numpy as np

from pathlib import Path
from pymatgen.core import Composition


# ============================================================
# MATTERGEN V1.2 — ADVANCED COMPOSITION DESCRIPTORS
# ============================================================

INPUT_PATH = Path(
    "data/materials_project_clean.csv"
)

OUTPUT_PATH = Path(
    "data/materials_project_advanced_features.csv"
)


def composition_features(formula):

    try:

        comp = Composition(
            formula
        )

        amounts = np.array(
            list(
                comp.get_el_amt_dict().values()
            ),
            dtype=float
        )

        fractions = (
            amounts / amounts.sum()
        )

        # ----------------------------------------------------
        # Stoichiometric descriptors
        # ----------------------------------------------------

        max_fraction = fractions.max()

        min_fraction = fractions.min()

        fraction_std = fractions.std()

        fraction_range = (
            max_fraction - min_fraction
        )

        # Shannon entropy
        entropy = -np.sum(
            fractions *
            np.log(
                fractions
            )
        )

        # Normalized entropy
        if len(fractions) > 1:

            normalized_entropy = (
                entropy /
                np.log(len(fractions))
            )

        else:

            normalized_entropy = 0.0

        # Dominant-element fraction
        dominant_fraction = max_fraction

        # Number of elements
        num_elements = len(
            fractions
        )

        # Effective number of elements
        effective_elements = (
            np.exp(entropy)
        )

        return {
            "stoich_max_fraction":
                max_fraction,

            "stoich_min_fraction":
                min_fraction,

            "stoich_fraction_std":
                fraction_std,

            "stoich_fraction_range":
                fraction_range,

            "stoich_entropy":
                entropy,

            "stoich_normalized_entropy":
                normalized_entropy,

            "dominant_element_fraction":
                dominant_fraction,

            "effective_element_count":
                effective_elements,

            "composition_element_count":
                num_elements,
        }

    except Exception:

        return {
            "stoich_max_fraction": np.nan,
            "stoich_min_fraction": np.nan,
            "stoich_fraction_std": np.nan,
            "stoich_fraction_range": np.nan,
            "stoich_entropy": np.nan,
            "stoich_normalized_entropy": np.nan,
            "dominant_element_fraction": np.nan,
            "effective_element_count": np.nan,
            "composition_element_count": np.nan,
        }


def main():

    print("=" * 60)
    print(
        "MATTERGEN V1.2 — ADVANCED COMPOSITION DESCRIPTORS"
    )
    print("=" * 60)

    print(
        "\nLoading clean dataset..."
    )

    df = pd.read_csv(
        INPUT_PATH,
        keep_default_na=False
    )

    print(
        f"Materials loaded: {len(df)}"
    )

    print(
        "\nGenerating advanced descriptors..."
    )

    feature_rows = []

    failed = 0

    for index, formula in enumerate(
        df["formula"]
    ):

        features = composition_features(
            formula
        )

        if all(
            pd.isna(value)
            for value in features.values()
        ):

            failed += 1

        feature_rows.append(
            features
        )

        if (
            index + 1
        ) % 10000 == 0:

            print(
                f"  Processed: {index + 1}"
            )

    advanced_df = pd.DataFrame(
        feature_rows
    )

    # --------------------------------------------------------
    # Combine with material IDs
    # --------------------------------------------------------

    output_df = pd.concat(
        [
            df[
                [
                    "material_id",
                    "formula",
                ]
            ].reset_index(
                drop=True
            ),

            advanced_df.reset_index(
                drop=True
            ),
        ],
        axis=1
    )

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

    print("\n" + "=" * 60)
    print(
        "ADVANCED DESCRIPTOR GENERATION COMPLETE"
    )
    print("=" * 60)

    print(
        f"\nMaterials: {len(output_df)}"
    )

    print(
        f"New features: {len(advanced_df.columns)}"
    )

    print(
        f"Failed formulas: {failed}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print("\nGenerated columns:")

    for column in advanced_df.columns:

        print(
            f"  - {column}"
        )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()