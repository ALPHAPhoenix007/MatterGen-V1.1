# MatterGen V1.1

## AI-Assisted Material Property Prediction

MatterGen is a machine learning and cheminformatics system for predicting material properties from molecular structure.

Version 1.1 focuses on improving the original implementation through dataset validation, expanded molecular descriptors, property-specific machine learning models, and cross-validation-based model selection.

The current system implements the **Material → Property** direction of MatterGen.

---

## Overview

Given a molecular structure represented as a SMILES string, MatterGen processes the structure using RDKit, extracts molecular descriptors, and predicts four material properties:

- Band Gap
- Formation Energy
- Stability Score
- Melting Point

The system also provides molecular similarity analysis using Morgan fingerprints and Tanimoto similarity, along with interactive 3D molecular visualization.

The project is currently an experimental research platform for development, benchmarking, and demonstration.

---

## Current Architecture

```text
                    Molecular Structure
                           │
                    SMILES / Formula
                           │
                           ▼
                         RDKit
                           │
                Molecular Validation
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
     Molecular Descriptors        Morgan Fingerprint
             │                           │
             │                           ▼
             │                  Tanimoto Similarity
             │                           │
             ▼                           │
      21 Molecular Features              │
             │                           │
     ┌───────┼────────┬────────┐         │
     ▼       ▼        ▼        ▼         │
   Band    Formation Stability Melting   │
   Gap     Energy    Score     Point     │
     │       │        │        │         │
     ▼       ▼        ▼        ▼         │
 Gradient  Random   Random   Gradient    │
 Boosting  Forest   Forest   Boosting    │
```

---

## Machine Learning Pipeline

### Feature Engineering

MatterGen V1.1 uses 21 molecular descriptors generated using RDKit.

The feature set includes descriptors related to:

- Molecular weight
- Molecular lipophilicity
- Hydrogen-bond donors and acceptors
- Topological polar surface area
- Rotatable bonds
- Aromaticity
- Carbon hybridization
- Molecular topology
- Molecular complexity
- Other structural properties

Morgan fingerprints are maintained separately from the prediction feature set and are used by the similarity engine.

---

## Property-Specific Models

Instead of using one model for every property, V1.1 evaluates multiple machine learning algorithms and selects the better-performing approach for each target.

| Property | Selected Model |
|---|---|
| Band Gap | Gradient Boosting |
| Formation Energy | Random Forest |
| Stability Score | Random Forest |
| Melting Point | Gradient Boosting |

The models evaluated during development include:

- Random Forest Regressor
- Gradient Boosting Regressor
- Ridge Regression

Model selection was based on 5-fold cross-validation.

---

## Dataset

The current cleaned dataset contains:

- **144 unique compounds**
- **7 columns**
- **21 molecular features**
- **0 invalid SMILES**
- **0 duplicate SMILES**
- **0 missing values**

Dataset columns:

```text
smiles
formula
band_gap_ev
formation_energy
stability_score
melting_point_k
name
```

The original dataset contained duplicate records and one invalid SMILES entry. A dedicated dataset cleaning and auditing pipeline was introduced to identify and correct these issues.

The original dataset is preserved separately as:

```text
data/materials_dataset_raw.csv
```

The cleaned dataset used by the current system is:

```text
data/materials_dataset.csv
```

---

## Dataset Validation

MatterGen includes tools for inspecting and validating the dataset before model training.

The audit process checks:

- Dataset dimensions
- Duplicate records
- Duplicate SMILES
- Missing values
- SMILES validity
- Unique molecular structures
- Target-property distributions
- Statistical ranges of prediction targets

---

## Model Validation

V1.1 uses 5-fold cross-validation to evaluate model stability.

Current cross-validation results:

| Property | Model | R² |
|---|---|---:|
| Band Gap | Gradient Boosting | 0.9461 ± 0.0348 |
| Formation Energy | Random Forest | 0.9621 ± 0.0155 |
| Stability Score | Random Forest | 0.9106 ± 0.0870 |
| Melting Point | Gradient Boosting | 0.7493 ± 0.0935 |

These results are based on the current 144-compound dataset and should not be interpreted as general performance on unseen real-world materials.

---

## Baseline vs Expanded Features

The original MatterGen implementation used 8 molecular descriptors.

V1.1 expands the representation to 21 descriptors.

| Property | 8 Features | 21 Features |
|---|---:|---:|
| Band Gap | 0.9198 | 0.9525 |
| Formation Energy | 0.9493 | 0.9540 |
| Stability Score | 0.9228 | 0.9510 |
| Melting Point | 0.6338 | 0.6342 |

---

## Molecular Similarity

MatterGen uses Morgan fingerprints for structural similarity analysis.

The similarity engine calculates the **Tanimoto similarity coefficient** between the input molecule and compounds in the dataset.

The system can return structurally similar compounds along with their known properties.

---

## Molecular Visualization

MatterGen provides interactive 3D visualization of molecular structures using:

- RDKit
- 3D conformer generation
- Molecular geometry optimization
- Py3Dmol

---

## Application

The current application is built using Streamlit.

### Predictions

Displays:

- Molecular information
- Predicted material properties
- Molecular descriptors
- Prediction methodology

### Similarity Analysis

Displays:

- Structurally similar compounds
- Tanimoto similarity scores
- Property comparisons

### 3D Structure

Provides an interactive representation of the input molecule.

### Model Insights

Provides:

- Property-specific model information
- Cross-validation performance
- Feature importance
- Training-data distributions

---

## Project Structure

```text
MatterGen-V1.1/
│
├── app.py
├── README.md
├── requirements.txt
├── runtime.txt
│
├── backend/
│   ├── chemistry.py
│   ├── features.py
│   ├── ml_models.py
│   └── similarity.py
│
├── data/
│   ├── materials_dataset.csv
│   ├── materials_dataset_raw.csv
│   └── sample_inputs.csv
│
├── tools/
│   ├── audit_dataset.py
│   ├── clean_dataset.py
│   ├── evaluate_baseline.py
│   ├── feature_experiment.py
│   ├── model_experiment.py
│   └── cross_validation.py
│
└── utils/
    ├── config.py
    └── visualizer.py
```

---

## Installation

### Requirements

- Python 3.10
- pip
- Git

### Clone the repository

```bash
git clone https://github.com/ALPHAPhoenix007/MatterGen-V1.1.git
cd MatterGen-V1.1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## Development and Evaluation Tools

### Dataset audit

```bash
python tools/audit_dataset.py
```

### Dataset cleaning

```bash
python tools/clean_dataset.py
```

### Baseline evaluation

```bash
python tools/evaluate_baseline.py
```

### Feature experiment

```bash
python tools/feature_experiment.py
```

### Model comparison

```bash
python tools/model_experiment.py
```

### Cross-validation

```bash
python tools/cross_validation.py
```

---

## Current Limitations

MatterGen V1.1 is still an experimental system.

### Dataset Size

The current dataset contains only 144 compounds. This is insufficient for broad claims about material-property prediction performance.

### Dataset Composition

A larger, externally sourced materials dataset is required for meaningful scientific evaluation.

### Molecular Representation

The current prediction pipeline is based on molecular descriptors. More complex material structures, crystal lattices, elemental compositions, and periodic structures require representations beyond conventional molecular descriptors.

### Melting Point Prediction

Melting point currently has the weakest predictive performance among the four targets, indicating that additional data and more appropriate structural features are required.

### Generalization

The reported metrics are measured on the current dataset and should not be interpreted as proof of generalization to arbitrary unseen materials.

---

## Roadmap

MatterGen is planned as a two-directional material intelligence system.

### Direction 1 — Material → Property

The current development focuses on predicting material properties from molecular structure.

```text
Material Structure
       ↓
Feature Extraction
       ↓
Machine Learning
       ↓
Material Properties
```

### Direction 2 — Property → Material

The planned inverse system will begin with desired material properties and search for candidate materials that satisfy those requirements.

```text
Desired Properties
       ↓
Candidate Generation
       ↓
Property Prediction
       ↓
Candidate Ranking
       ↓
Potential Materials
```

Future development may include:

- Larger externally sourced materials datasets
- Materials Project integration
- Crystal and composition-based representations
- Uncertainty estimation
- Multi-objective optimization
- Candidate material generation
- Property-constrained search
- More advanced machine learning models
- Experimental candidate ranking

The long-term goal is to evolve MatterGen from a property prediction system into a broader **AI-assisted materials discovery platform**.

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Machine Learning | scikit-learn |
| Chemistry | RDKit |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Web Application | Streamlit |
| Molecular Visualization | Py3Dmol |
| Model Persistence | Joblib |

---

## Version History

### V1.0

Original hackathon implementation.

Focused on:

- Molecular property prediction
- Random Forest models
- 8 molecular descriptors
- Molecular similarity
- Streamlit interface

### V1.1

ML pipeline upgrade.

Introduced:

- Dataset cleaning and validation
- 144-compound cleaned dataset
- 21 molecular descriptors
- Property-specific model selection
- Random Forest and Gradient Boosting evaluation
- 5-fold cross-validation
- Improved model evaluation tools
- Updated production prediction pipeline

---

## Disclaimer

MatterGen is a research and educational project.

Predictions generated by the current system should not be treated as experimentally validated material properties or as a substitute for laboratory measurements or established computational materials-science workflows.

---

## Project

MatterGen is being developed as an ongoing exploration of machine learning, chemistry, mathematical modeling, and AI-assisted materials discovery.
