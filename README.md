# MatterGen

## AI-Powered Material Property Prediction

MatterGen is a machine learning application for predicting material properties from molecular structure. The system accepts molecular representations such as SMILES strings and chemical formulas, processes them using RDKit, and applies scikit-learn models to estimate selected material properties.

The project combines molecular feature extraction, supervised learning, molecular similarity analysis, and 3D structure visualization in a single Streamlit application.

## Features

- Molecular input through SMILES strings and chemical formulas
- Prediction of band gap, formation energy, stability score, and melting point
- Molecular descriptor-based machine learning
- Morgan fingerprint generation for molecular similarity analysis
- Tanimoto similarity matching against reference compounds
- Interactive 3D molecular visualization
- Model feature importance and prediction insights
- Modular separation of chemistry, machine learning, similarity, and visualization components

## Architecture

```text
Molecular Input
      |
      v
RDKit Molecular Processing
      |
      v
Feature Extraction
  |              |
  |              +--> Morgan Fingerprints
  |
  +-----------------> Molecular Descriptors
      |
      +----------------------+
      |                      |
      v                      v
ML Prediction         Similarity Analysis
      |                      |
      +----------+-----------+
                 |
                 v
          Streamlit Interface
                 |
       +---------+---------+
       |         |         |
       v         v         v
 Predictions  Similarity  3D Structure
```

## Machine Learning Pipeline

### 1. Molecular Processing

The application validates and parses molecular inputs using RDKit. SMILES strings are converted into molecular structures before feature extraction.

### 2. Feature Extraction

The system uses two main molecular representations:

- **Molecular descriptors** for material property prediction, including molecular weight, LogP, TPSA, and other structural descriptors.
- **Morgan fingerprints (ECFP4)** with 2048 bits for molecular similarity analysis.

### 3. Property Prediction

Random Forest regression models are used to predict the following properties:

| Property | Unit |
|---|---|
| Band Gap | eV |
| Formation Energy | eV/atom |
| Stability Score | 0–1 |
| Melting Point | K |

Separate regression models are used for the target properties.

### 4. Similarity Analysis

Morgan fingerprints are compared using the Tanimoto coefficient to identify structurally similar compounds in the reference dataset. The application returns the most similar compounds and their similarity scores.

### 5. Visualization

Molecular structures are visualized interactively in three dimensions using Py3Dmol. RDKit is used for molecular processing and 3D coordinate generation.

## Model Configuration

- **Algorithm:** Random Forest Regressor
- **Number of estimators:** 100
- **Maximum depth:** 10
- **Features:** RDKit molecular descriptors
- **Training split:** 80/20
- **Cross-validation:** 5-fold
- **Evaluation metrics:** MAE, RMSE, R²

The current prototype uses a dataset containing 50 compounds. The dataset is intended to demonstrate the complete prediction workflow rather than serve as a production-scale scientific benchmark.

## Project Structure

```text
MatterGen/
│
├── app.py
├── requirements.txt
├── README.md
│
├── backend/
│   ├── chemistry.py
│   ├── features.py
│   ├── ml_models.py
│   └── similarity.py
│
├── data/
│   ├── materials_dataset.csv
│   └── sample_inputs.csv
│
├── models/
│   └── trained_model.pkl
│
└── utils/
    ├── config.py
    └── visualizer.py
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development and ML pipeline |
| Scikit-learn | Machine learning and model evaluation |
| RDKit | Molecular processing and cheminformatics |
| Streamlit | Web application interface |
| Pandas | Dataset processing |
| NumPy | Numerical computation |
| Plotly | Interactive data visualization |
| Py3Dmol | 3D molecular visualization |

## Dataset

The prototype uses `data/materials_dataset.csv`, containing molecular representations and corresponding material properties for 50 compounds.

The current dataset is intended for development and demonstration. Because of its limited size and composition, predictions should not be treated as experimentally validated material properties.

Potential data sources for future development include:

- Materials Project
- PubChem
- Cambridge Structural Database

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd MatterGen
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## Example Workflow

1. Enter a SMILES string or molecular formula.
2. Validate and process the molecular structure using RDKit.
3. Generate molecular descriptors and fingerprints.
4. Run the trained Random Forest models.
5. Display predicted material properties.
6. Compare the input molecule with structurally similar compounds.
7. Explore the molecular structure and model insights.

## Limitations

MatterGen is currently a research and hackathon prototype. The training dataset is relatively small and includes synthetic or curated values. Consequently, the predictions are intended for demonstration and exploration rather than direct experimental or industrial decision-making.

The accuracy and generalization of the models can be improved substantially with larger, higher-quality, experimentally validated datasets.

## Future Development

- Expand the training dataset to thousands of compounds
- Integrate Materials Project data through its API
- Evaluate additional models such as Gradient Boosting and neural networks
- Add prediction uncertainty estimates
- Implement multi-objective material optimization
- Improve validation using experimentally reported properties
- Provide downloadable prediction reports
- Deploy the application for wider access

## Live Application

https://mattergen.streamlit.app/

## License

This project is released under the MIT License.

## Acknowledgements

MatterGen uses the following open-source technologies:

- RDKit
- Scikit-learn
- Streamlit
- Pandas
- NumPy
- Plotly
- Py3Dmol
