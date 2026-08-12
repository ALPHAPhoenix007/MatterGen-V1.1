import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os

os.environ["RDKIT_NO_IMPORT_WARNING"] = "1"


def normalize_formula(formula: str) -> str:
    if formula is None:
        return ""
    return formula.replace(" ", "").upper()


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.chemistry import (
    smiles_to_mol,
    formula_to_smiles,
    validate_molecule,
    get_molecular_formula,
    calculate_basic_properties
)

from backend.features import (
    extract_features_from_smiles,
    get_feature_names
)

from backend.ml_models import (
    MultiPropertyPredictor,
    PROPERTY_MODEL_TYPES
)

from backend.similarity import SimilarityEngine

from utils.visualizer import render_3d_molecule

from utils.config import (
    COLORS,
    PROPERTIES
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MatterGen - Material Property Predictor",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# UI COLORS
# =========================================================

COLORS = {
    "primary": "#1f77b4",
    "background": "#000000",
    "text": "#12D2DF"
}


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    f"""
<style>

.main {{
    background-color: {COLORS['background']};
    color: {COLORS['text']};
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 24px;
}}

.stTabs [data-baseweb="tab"] {{
    height: 50px;
    background-color: white;
    color: {COLORS['text']};
    border-radius: 8px;
    padding: 0px 24px;
    font-weight: 500;
}}

.stTabs [aria-selected="true"] {{
    background-color: {COLORS['primary']};
    color: white;
}}

.metric-card {{
    background-color: white;
    color: {COLORS['text']};
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 10px 0;
}}

h1 {{
    color: {COLORS['primary']};
}}

h2, h3 {{
    color: {COLORS['text']};
}}

p, span, label, div {{
    color: {COLORS['text']};
}}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# MODEL INITIALIZATION
# =========================================================

@st.cache_resource
def load_models():

    data_path = (
        Path(__file__).parent
        / "data"
        / "materials_dataset.csv"
    )

    df = pd.read_csv(
        data_path,
        on_bad_lines="skip"
    )

    # Train the production MatterGen pipeline
    predictor = MultiPropertyPredictor(
        PROPERTIES
    )

    training_results = predictor.train_all(
        df,
        smiles_col="smiles"
    )

    # Similarity database
    sim_engine = SimilarityEngine()

    sim_engine.load_database(
        df,
        smiles_col="smiles"
    )

    return (
        predictor,
        sim_engine,
        df,
        training_results
    )


with st.spinner("Initializing MatterGen..."):

    (
        predictor,
        similarity_engine,
        dataset,
        training_results
    ) = load_models()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.image(
    "https://via.placeholder.com/300x100/1f77b4/ffffff?text=MatterGen"
)

st.sidebar.title("MatterGen")

st.sidebar.markdown(
    "### AI powered-Material Property Predictor"
)

st.sidebar.markdown("---")


# =========================================================
# INPUT METHOD
# =========================================================

input_method = st.sidebar.radio(
    "Input Method",
    [
        "SMILES String",
        "Molecular Formula",
        "Example Molecules"
    ]
)


molecule_input = None


# ---------------------------------------------------------
# SMILES
# ---------------------------------------------------------

if input_method == "SMILES String":

    molecule_input = st.sidebar.text_input(
        "Enter SMILES",
        value="c1ccccc1",
        help="Example: c1ccccc1 (benzene)"
    )


# ---------------------------------------------------------
# Molecular Formula
# ---------------------------------------------------------

elif input_method == "Molecular Formula":

    user_formula = normalize_formula(
        st.sidebar.text_input(
            "Enter Formula",
            placeholder="e.g. C2H4O2"
        )
    )

    molecule_input = None

    if user_formula:

        dataset["formula_norm"] = (
            dataset["formula"]
            .apply(normalize_formula)
        )

        match = dataset[
            dataset["formula_norm"]
            == user_formula
        ]

        if match.empty:

            st.sidebar.warning(
                "Formula not found in database."
            )

        else:

            molecule_input = (
                match.iloc[0]["smiles"]
            )

            compound_name = (
                match.iloc[0]["name"]
            )

            st.sidebar.success(
                f"Found: {compound_name}"
            )


# ---------------------------------------------------------
# Examples
# ---------------------------------------------------------

else:

    examples = pd.read_csv(
        Path(__file__).parent
        / "data"
        / "sample_inputs.csv"
    )

    selected = st.sidebar.selectbox(
        "Select Example",
        examples["name"].tolist()
    )

    molecule_input = (
        examples[
            examples["name"] == selected
        ]["smiles"]
        .values[0]
    )

    st.sidebar.info(
        f"**{selected}**: {molecule_input}"
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.sidebar.markdown("---")

predict_button = st.sidebar.button(
    "🔮 Predict Properties",
    type="primary",
    use_container_width=True
)


# =========================================================
# ABOUT
# =========================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
### About MatterGen

MatterGen predicts material properties
from molecular structure using machine
learning and cheminformatics.

**Property Models**

- Band Gap → Gradient Boosting
- Formation Energy → Random Forest
- Stability → Random Forest
- Melting Point → Gradient Boosting

**Feature Engineering**

- 21 RDKit molecular descriptors
- Morgan fingerprints for similarity
- Tanimoto similarity matching
"""
)


# =========================================================
# MAIN HEADER
# =========================================================

st.title(
    "MatterGen: AI Material Property Predictor"
)

st.markdown(
    "### Predict material properties using "
    "molecular structure and machine learning"
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Predictions",
        "Similarity Analysis",
        "3D Structure",
        "Model Insights"
    ]
)


# =========================================================
# TAB 1 — PREDICTIONS
# =========================================================

with tab1:

    if predict_button and molecule_input:

        # -------------------------------------------------
        # Validate molecule
        # -------------------------------------------------

        mol = smiles_to_mol(
            molecule_input
        )

        is_valid, msg = validate_molecule(
            mol
        )

        if not is_valid:

            st.error(
                f"Invalid molecule: {msg}"
            )

        else:

            st.success(
                f"Valid molecule: {msg}"
            )

            # -------------------------------------------------
            # Molecular information
            # -------------------------------------------------

            st.markdown(
                "### Molecular Information"
            )

            col1, col2, col3 = st.columns(3)

            formula = get_molecular_formula(
                mol
            )

            basic_props = calculate_basic_properties(
                mol
            )

            with col1:

                st.markdown(
                    '<div class="metric-card">',
                    unsafe_allow_html=True
                )

                st.metric(
                    "Molecular Formula",
                    formula
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown(
                    '<div class="metric-card">',
                    unsafe_allow_html=True
                )

                st.metric(
                    "Molecular Weight",
                    f"{basic_props['molecular_weight']:.2f} g/mol"
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

            with col3:

                st.markdown(
                    '<div class="metric-card">',
                    unsafe_allow_html=True
                )

                st.metric(
                    "Heavy Atoms",
                    int(
                        basic_props["heavy_atoms"]
                    )
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

            # -------------------------------------------------
            # Predictions
            # -------------------------------------------------

            st.markdown(
                "### Predicted Material Properties"
            )

            predictions = (
                predictor.predict_all(
                    molecule_input
                )
            )

            property_info = {

                "band_gap_ev": (
                    "Band Gap",
                    "eV",
                    "🔋"
                ),

                "formation_energy": (
                    "Formation Energy",
                    "eV/atom",
                    "⚡"
                ),

                "stability_score": (
                    "Stability Score",
                    "0-1 scale",
                    "🛡️"
                ),

                "melting_point_k": (
                    "Melting Point",
                    "K",
                    "🌡️"
                )
            }

            col1, col2 = st.columns(2)

            for idx, (prop, value) in enumerate(
                predictions.items()
            ):

                if prop not in property_info:
                    continue

                name, unit, icon = (
                    property_info[prop]
                )

                target_col = (
                    col1
                    if idx % 2 == 0
                    else col2
                )

                with target_col:

                    st.markdown(
                        '<div class="metric-card">',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"**{icon} {name}**"
                    )

                    st.markdown(
                        f"""
                        <h2 style='margin:0;'>
                        {value:.3f} {unit}
                        </h2>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

            # -------------------------------------------------
            # Molecular descriptors
            # -------------------------------------------------

            st.markdown(
                "### Molecular Descriptors"
            )

            descriptor_values = (
                extract_features_from_smiles(
                    molecule_input
                )
            )

            feature_names = (
                get_feature_names()
            )

            if descriptor_values is not None:

                desc_df = pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "Value": descriptor_values
                    }
                )

                st.dataframe(
                    desc_df,
                    hide_index=True,
                    use_container_width=True
                )

            # -------------------------------------------------
            # Explanation
            # -------------------------------------------------

            st.markdown(
                "### Prediction Explanation"
            )

            st.info(
                f"""
**How MatterGen works:**

- RDKit validates and processes the molecular structure.
- {len(get_feature_names())} molecular descriptors are extracted.
- Each property is predicted using its validated model.
- Morgan fingerprints are used separately for molecular similarity.
- The production dataset contains {len(dataset)} validated compounds.
"""
            )

    else:

        st.info(
            "Enter a molecule in the sidebar and "
            "click 'Predict Properties' to begin"
        )

        # Dataset statistics

        st.markdown(
            "### Dataset Statistics"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Compounds",
                len(dataset)
            )

        with col2:
            st.metric(
                "Average Band Gap",
                f"{dataset['band_gap_ev'].mean():.2f} eV"
            )

        with col3:
            st.metric(
                "Avg Stability",
                f"{dataset['stability_score'].mean():.2f}"
            )

        with col4:
            st.metric(
                "Avg Melting Point",
                f"{dataset['melting_point_k'].mean():.0f} K"
            )


# =========================================================
# TAB 2 — SIMILARITY ANALYSIS
# =========================================================

with tab2:

    if predict_button and molecule_input:

        st.markdown(
            "### Similar Compounds in Database"
        )

        st.markdown(
            "Finding structurally similar known materials..."
        )

        similar = (
            similarity_engine.find_similar(
                molecule_input,
                n_results=5
            )
        )

        if similar:

            # Predictions are needed for comparison
            predictions = (
                predictor.predict_all(
                    molecule_input
                )
            )

            for rank, (
                idx,
                similarity,
                info
            ) in enumerate(
                similar,
                1
            ):

                with st.expander(
                    f"#{rank}: "
                    f"{info.get('name', 'Unknown')} "
                    f"- Similarity: {similarity:.3f}"
                ):

                    col1, col2 = st.columns(
                        [1, 2]
                    )

                    with col1:

                        st.markdown(
                            f"**SMILES:** `{info['smiles']}`"
                        )

                        st.markdown(
                            f"**Formula:** "
                            f"{info.get('formula', 'N/A')}"
                        )

                        st.markdown(
                            f"**Tanimoto Similarity:** "
                            f"{similarity:.4f}"
                        )

                    with col2:

                        comparison_data = []

                        for prop in PROPERTIES:

                            if (
                                prop in info
                                and prop in predictions
                            ):

                                comparison_data.append(
                                    {
                                        "Property": prop,
                                        "Query Molecule":
                                            predictions[prop],
                                        "Similar Molecule":
                                            info[prop]
                                    }
                                )

                        if comparison_data:

                            comp_df = pd.DataFrame(
                                comparison_data
                            )

                            st.dataframe(
                                comp_df,
                                hide_index=True,
                                use_container_width=True
                            )

            # Similarity chart

            st.markdown(
                "### Similarity Score Distribution"
            )

            similarities = [
                s[1]
                for s in similar
            ]

            names = [
                s[2].get(
                    "name",
                    f"Compound {s[0]}"
                )
                for s in similar
            ]

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=names,
                        y=similarities,
                        marker_color=
                            COLORS["primary"]
                    )
                ]
            )

            fig.update_layout(
                title="Tanimoto Similarity Scores",
                xaxis_title="Compound",
                yaxis_title="Similarity Score",
                yaxis_range=[0, 1]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "No similar compounds found in database"
            )

    else:

        st.info(
            "Submit a molecule to see similarity analysis"
        )


# =========================================================
# TAB 3 — 3D STRUCTURE
# =========================================================

with tab3:

    if predict_button and molecule_input:

        st.markdown(
            "### Interactive 3D Molecular Structure"
        )

        viz_style = st.radio(
            "Visualization Style",
            [
                "stick",
                "sphere",
                "both"
            ],
            horizontal=True
        )

        style_map = {
            "stick": "stick",
            "sphere": "sphere",
            "both": "stick"
        }

        try:

            render_3d_molecule(
                molecule_input,
                style=style_map[viz_style]
            )

            st.caption(
                "Use mouse to rotate, scroll to zoom"
            )

        except Exception as e:

            st.error(
                f"Could not render 3D structure: {str(e)}"
            )

            st.info(
                "The molecule may be too complex "
                "for 3D rendering"
            )

    else:

        st.info(
            "Submit a molecule to see 3D structure"
        )


# =========================================================
# TAB 4 — MODEL INSIGHTS
# =========================================================

with tab4:

    st.markdown(
        "### Model Performance & Feature Importance"
    )

    # -----------------------------------------------------
    # Select property
    # -----------------------------------------------------

    selected_property = st.selectbox(
        "Select Property",
        PROPERTIES,
        format_func=lambda x: {
            "band_gap_ev": "Band Gap",
            "formation_energy": "Formation Energy",
            "stability_score": "Stability Score",
            "melting_point_k": "Melting Point"
        }.get(x, x)
    )

    selected_model = (
        predictor.models[
            selected_property
        ]
    )

    model_type = (
        PROPERTY_MODEL_TYPES[
            selected_property
        ]
    )

    model_display_name = {
        "random_forest":
            "Random Forest Regressor",
        "gradient_boosting":
            "Gradient Boosting Regressor"
    }.get(
        model_type,
        model_type
    )

    # -----------------------------------------------------
    # Model information
    # -----------------------------------------------------

    st.markdown(
        "#### Model Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Model",
            model_display_name
        )

    with col2:

        cv_result = training_results[
            selected_property
        ]

        st.metric(
            "5-Fold CV R²",
            f"{cv_result['cv_mean']:.4f}"
        )

    with col3:

        st.metric(
            "CV Std. Dev.",
            f"{cv_result['cv_std']:.4f}"
        )

    st.markdown(
        f"""
**Training Data:** {len(dataset)} validated compounds

**Input Features:** {len(get_feature_names())} RDKit molecular descriptors

**Model Selection:** Property-specific model selected through 5-fold cross-validation
"""
    )

    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------

    importance = (
        selected_model.get_feature_importance()
    )

    if importance:

        st.markdown(
            "#### Feature Importance"
        )

        imp_df = pd.DataFrame(
            list(
                importance.items()
            )[:10],
            columns=[
                "Feature",
                "Importance"
            ]
        )

        fig = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues"
        )

        fig.update_layout(
            showlegend=False,
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Feature importance is not available "
            "for this model."
        )

    # -----------------------------------------------------
    # Property distribution
    # -----------------------------------------------------

    st.markdown(
        "#### Property Distribution in Training Data"
    )

    fig = px.histogram(
        dataset,
        x=selected_property,
        nbins=30,
        color_discrete_sequence=[
            COLORS["primary"]
        ]
    )

    fig.update_layout(
        title=f"Distribution of {selected_property}",
        xaxis_title=selected_property,
        yaxis_title="Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
<div style='text-align: center; color: #666;'>
    <p><strong>MatterGen</strong> -
    AI-Powered Material Property Prediction</p>

    <p>
    Built with Streamlit, RDKit, and scikit-learn |
    Hackathon Project 2026
    </p>
</div>
""",
    unsafe_allow_html=True
)