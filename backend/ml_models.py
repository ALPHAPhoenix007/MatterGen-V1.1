"""
Machine Learning models for MatterGen.

Uses property-specific models selected through 5-fold cross-validation:
- Band gap: Gradient Boosting
- Formation energy: Random Forest
- Stability: Random Forest
- Melting point: Gradient Boosting
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.model_selection import (
    cross_val_score,
    KFold
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from typing import Dict, Optional, List

import joblib

from backend.features import (
    extract_features_from_smiles,
    get_feature_names
)

from utils.config import ML_CONFIG


# ---------------------------------------------------------
# Property-specific model configuration
# ---------------------------------------------------------

PROPERTY_MODEL_TYPES = {
    "band_gap_ev": "gradient_boosting",
    "formation_energy": "random_forest",
    "stability_score": "random_forest",
    "melting_point_k": "gradient_boosting"
}


class PropertyPredictor:
    """
    ML model for predicting one material property
    from molecular structure.
    """

    def __init__(self, property_name: str = "band_gap_ev"):

        self.property_name = property_name

        self.model = None

        self.feature_mean = None
        self.feature_std = None

        self.feature_names = get_feature_names()

        self.model_type = PROPERTY_MODEL_TYPES.get(
            property_name,
            "random_forest"
        )

        self.trained = False

    # -----------------------------------------------------
    # Create model
    # -----------------------------------------------------

    def _create_model(self, model_type: str):

        if model_type == "gradient_boosting":

            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=ML_CONFIG["random_state"]
            )

        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=ML_CONFIG["random_state"],
            n_jobs=-1
        )

    # -----------------------------------------------------
    # Train model
    # -----------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: Optional[str] = None
    ):

        if model_type is None:
            model_type = self.model_type

        self.model_type = model_type

        # Normalize features
        self.feature_mean = np.mean(X, axis=0)

        self.feature_std = np.std(X, axis=0)

        self.feature_std = np.where(
            self.feature_std == 0,
            1,
            self.feature_std
        )

        X_normalized = (
            X - self.feature_mean
        ) / self.feature_std

        # Create model
        self.model = self._create_model(
            model_type
        )

        # Train on complete training dataset
        self.model.fit(
            X_normalized,
            y
        )

        self.trained = True

        # -------------------------------------------------
        # Use the SAME shuffled 5-fold CV methodology
        # used during our model-selection experiment.
        # -------------------------------------------------

        cv = KFold(
            n_splits=ML_CONFIG["cv_folds"],
            shuffle=True,
            random_state=ML_CONFIG["random_state"]
        )

        cv_scores = cross_val_score(
            self.model,
            X_normalized,
            y,
            cv=cv,
            scoring="r2"
        )

        return {
            "cv_mean": float(
                np.mean(cv_scores)
            ),
            "cv_std": float(
                np.std(cv_scores)
            ),
            "model_type": model_type
        }

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:

        if not self.trained or self.model is None:
            raise ValueError(
                "Model not trained yet"
            )

        X_normalized = (
            X - self.feature_mean
        ) / self.feature_std

        return self.model.predict(
            X_normalized
        )

    # -----------------------------------------------------
    # Predict directly from SMILES
    # -----------------------------------------------------

    def predict_from_smiles(
        self,
        smiles: str
    ) -> Optional[float]:

        features = extract_features_from_smiles(
            smiles
        )

        if features is None:
            return None

        features = features.reshape(
            1,
            -1
        )

        prediction = self.predict(
            features
        )

        return float(
            prediction[0]
        )

    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------

    def get_feature_importance(
        self
    ) -> Dict[str, float]:

        if not self.trained:
            return {}

        if hasattr(
            self.model,
            "feature_importances_"
        ):

            importances = (
                self.model.feature_importances_
            )

            importance_dict = {
                name: float(importance)
                for name, importance in zip(
                    self.feature_names,
                    importances
                )
            }

            return dict(
                sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            )

        return {}

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    def save(
        self,
        filepath: str
    ):

        if not self.trained:
            raise ValueError(
                "Cannot save untrained model"
            )

        model_data = {
            "model": self.model,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "feature_names": self.feature_names,
            "property_name": self.property_name,
            "model_type": self.model_type
        }

        joblib.dump(
            model_data,
            filepath
        )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    def load(
        self,
        filepath: str
    ):

        model_data = joblib.load(
            filepath
        )

        self.model = model_data["model"]

        self.feature_mean = (
            model_data["feature_mean"]
        )

        self.feature_std = (
            model_data["feature_std"]
        )

        self.feature_names = (
            model_data["feature_names"]
        )

        self.property_name = (
            model_data["property_name"]
        )

        self.model_type = model_data.get(
            "model_type",
            PROPERTY_MODEL_TYPES.get(
                self.property_name,
                "random_forest"
            )
        )

        self.trained = True


class MultiPropertyPredictor:
    """
    Predict multiple material properties.

    Each property automatically uses the
    model selected through cross-validation.
    """

    def __init__(
        self,
        property_names: List[str]
    ):

        self.property_names = property_names

        self.models = {
            prop: PropertyPredictor(
                property_name=prop
            )
            for prop in property_names
        }

    # -----------------------------------------------------
    # Train all property models
    # -----------------------------------------------------

    def train_all(
        self,
        df: pd.DataFrame,
        smiles_col: str = "smiles"
    ):

        features_list = []
        valid_indices = []

        for idx, smiles in enumerate(
            df[smiles_col]
        ):

            features = (
                extract_features_from_smiles(
                    smiles
                )
            )

            if features is not None:

                features_list.append(
                    features
                )

                valid_indices.append(
                    idx
                )

        if not features_list:
            raise ValueError(
                "No valid molecular features found."
            )

        X = np.vstack(
            features_list
        )

        results = {}

        for prop in self.property_names:

            if prop not in df.columns:
                continue

            y = (
                df[prop]
                .iloc[valid_indices]
                .values
            )

            model = self.models[prop]

            score = model.train(
                X,
                y
            )

            results[prop] = score

        return results

    # -----------------------------------------------------
    # Predict all properties
    # -----------------------------------------------------

    def predict_all(
        self,
        smiles: str
    ) -> Dict[str, float]:

        predictions = {}

        for prop, model in self.models.items():

            prediction = (
                model.predict_from_smiles(
                    smiles
                )
            )

            if prediction is not None:
                predictions[prop] = prediction

        return predictions


# ---------------------------------------------------------
# Evaluation utility
# ---------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate regression evaluation metrics.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }