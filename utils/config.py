"""
Configuration constants for MatterGen
"""

# Color Palette - Professional scientific theme
COLORS = {
    'primary': '#1f77b4',      # Deep blue
    'secondary': '#2ca02c',    # Green
    'accent': '#ff7f0e',       # Orange
    'background': '#f8f9fa',   # Light gray
    'text': '#212529',         # Dark gray
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545'
}

# ML Model Parameters
ML_CONFIG = {
    'n_neighbors': 5,           # For similarity matching
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5
}

# Molecular Fingerprint Settings
FINGERPRINT_CONFIG = {
    'radius': 2,                # Morgan fingerprint radius
    'n_bits': 2048,            # Fingerprint length
    'use_features': True
}

# Property Prediction Targets
PROPERTIES = [
    'band_gap_ev',              # Band gap in eV
    'formation_energy',         # Formation energy (eV/atom)
    'stability_score',          # Stability metric (0-1)
    'melting_point_k'           # Melting point in Kelvin
]

# Molecular Descriptors to Calculate
DESCRIPTORS = [
    'MolWt',
    'MolLogP',
    'NumHDonors',
    'NumHAcceptors',
    'TPSA',
    'NumRotatableBonds',
    'NumAromaticRings',
    'FractionCSP3',
    'HeavyAtomCount',
    'NumHeteroatoms',
    'RingCount',
    'NumAliphaticRings',
    'NumSaturatedRings',
    'NumAromaticCarbocycles',
    'NumAromaticHeterocycles',
    'HeavyAtomMolWt',
    'ExactMolWt',
    'MolMR',
    'LabuteASA',
    'BalabanJ',
    'BertzCT'
]