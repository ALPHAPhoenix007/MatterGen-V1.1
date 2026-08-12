import pandas as pd
from rdkit import Chem

DATASET_PATH = "data/materials_dataset.csv"

TARGETS = [
    "band_gap_ev",
    "formation_energy",
    "stability_score",
    "melting_point_k"
]


def main():
    df = pd.read_csv(DATASET_PATH)

    print("\n========== MATTERGEN DATASET AUDIT ==========\n")

    # Basic information
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")

    # Duplicate rows
    duplicate_rows = df.duplicated().sum()
    duplicate_smiles = df["smiles"].duplicated().sum()

    print("\n--- Duplicates ---")
    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Duplicate SMILES: {duplicate_smiles}")

    # Missing values
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    # Validate SMILES
    valid_smiles = 0
    invalid_smiles = 0

    for smiles in df["smiles"]:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            invalid_smiles += 1
        else:
            valid_smiles += 1

    print("\n--- SMILES Validation ---")
    print(f"Valid SMILES: {valid_smiles}")
    print(f"Invalid SMILES: {invalid_smiles}")

    # Unique molecules
    print("\n--- Unique Values ---")
    print(f"Unique SMILES: {df['smiles'].nunique()}")
    print(f"Unique formulas: {df['formula'].nunique()}")
    print(f"Unique names: {df['name'].nunique()}")

    # Target statistics
    print("\n--- Target Statistics ---")

    for target in TARGETS:
        if target in df.columns:
            values = df[target]

            print(f"\n{target}")
            print(f"  Min:    {values.min():.4f}")
            print(f"  Max:    {values.max():.4f}")
            print(f"  Mean:   {values.mean():.4f}")
            print(f"  Std:    {values.std():.4f}")
            print(f"  Unique: {values.nunique()}")

    print("\n==============================================\n")


if __name__ == "__main__":
    main()