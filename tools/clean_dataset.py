import pandas as pd
from rdkit import Chem

INPUT_PATH = "data/materials_dataset.csv"
OUTPUT_PATH = "data/materials_dataset.csv"

INVALID_SMILES_FIXES = {
    "c1ccc(cc1)NO2": "c1ccc(cc1)[N+](=O)[O-]"
}


def main():
    df = pd.read_csv(INPUT_PATH)

    original_rows = len(df)

    # Remove exact duplicate records
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = original_rows - len(df)

    # Fix known invalid SMILES
    smiles_fixed = 0

    for old_smiles, new_smiles in INVALID_SMILES_FIXES.items():
        mask = df["smiles"] == old_smiles
        smiles_fixed += mask.sum()
        df.loc[mask, "smiles"] = new_smiles

    # Validate all remaining SMILES
    invalid_rows = []

    for index, smiles in df["smiles"].items():
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            invalid_rows.append(index)

    if invalid_rows:
        print("ERROR: Invalid SMILES still remain:")
        for index in invalid_rows:
            print(f"  Row {index}: {df.loc[index, 'smiles']}")
        return

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n========== DATASET CLEANING ==========\n")
    print(f"Original rows:       {original_rows}")
    print(f"Duplicates removed:  {duplicates_removed}")
    print(f"SMILES fixed:        {smiles_fixed}")
    print(f"Final rows:          {len(df)}")
    print(f"Invalid SMILES:      {len(invalid_rows)}")
    print("\nClean dataset saved successfully.")
    print("======================================\n")


if __name__ == "__main__":
    main()