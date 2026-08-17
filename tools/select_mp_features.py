import pandas as pd
from pathlib import Path


# ============================================================
# MATTERGEN V1.2 — FEATURE REDUCTION
# ============================================================

INPUT_PATH = Path(
    "data/materials_project_ml.csv"
)

OUTPUT_PATH = Path(
    "data/materials_project_ml_reduced.csv"
)


# Exact/redundant representations identified during
# V1.2 feature analysis.
REMOVE_FEATURES = [
    "composition_nelements",
    "structure_volume",
    "structure_nsites",
    "structure_volume_per_atom",
]


def main():

    print("=" * 60)
    print("MATTERGEN V1.2 — FEATURE REDUCTION")
    print("=" * 60)

    print("\nLoading ML dataset...")

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False
    )

    print(
        f"Original rows: {len(df)}"
    )

    print(
        f"Original columns: {len(df.columns)}"
    )

    print(
        f"Removing {len(REMOVE_FEATURES)} redundant features..."
    )

    for feature in REMOVE_FEATURES:

        if feature in df.columns:

            df = df.drop(
                columns=[feature]
            )

            print(
                f"  Removed: {feature}"
            )

        else:

            print(
                f"  WARNING: {feature} not found"
            )

    # --------------------------------------------------------
    # Save reduced dataset
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
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FEATURE REDUCTION COMPLETE")
    print("=" * 60)

    print(
        f"\nRows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"Features removed: {len(REMOVE_FEATURES)}"
    )

    print(
        f"Remaining features: "
        f"{len(df.columns) - 6}"
    )

    print(
        f"\nOutput: {OUTPUT_PATH}"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()