# ============================================================
# tests/test_models.py
# ============================================================
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import generate_student_dataset
from src.data_cleaning import encode_categorical_variables, get_feature_matrix
from src.models import RegressionLinéaire


@pytest.fixture
def prepared_data():
    """Données préparées pour les tests des modèles."""
    df = generate_student_dataset(n_samples=200, random_state=42)
    df_enc = encode_categorical_variables(df)
    X, y_reg, y_clf = get_feature_matrix(df_enc)
    feature_cols = [
        'temps_etude_hebdo', 'nb_devoirs_rendus',
        'exercices_completes_pct', 'score_motivation'
    ]
    return X[feature_cols], y_reg, y_clf


class TestRegressionLinéaire:
    """Tests du modèle de régression linéaire."""
    
    def test_fit_runs_without_error(self, prepared_data):
        """Le fit ne doit pas lever d'exception."""
        X, y, _ = prepared_data
        reg = RegressionLinéaire(feature_names=X.columns.tolist())
        reg.fit(X, y)
        assert reg.is_fitted
    
    def test_r2_positive(self, prepared_data):
        """Le R² doit être positif sur des données corrélées."""
        X, y, _ = prepared_data
        reg = RegressionLinéaire(feature_names=X.columns.tolist())
        reg.fit(X, y)
        assert reg.metrics['R²'] > 0
    
    def test_predictions_in_range(self, prepared_data):
        """Les prédictions ne doivent pas être absurdes."""
        X, y, _ = prepared_data
        reg = RegressionLinéaire(feature_names=X.columns.tolist())
        reg.fit(X, y)
        assert reg.y_pred.max() < 30
        assert reg.y_pred.min() > -10
    
    def test_residuals_length(self, prepared_data):
        """Les résidus doivent avoir la même taille que y_test."""
        X, y, _ = prepared_data
        reg = RegressionLinéaire(feature_names=X.columns.tolist())
        reg.fit(X, y)
        assert len(reg.residuals) == len(reg.y_test)
    
    def test_metrics_keys(self, prepared_data):
        """Toutes les métriques attendues doivent être présentes."""
        X, y, _ = prepared_data
        reg = RegressionLinéaire(feature_names=X.columns.tolist())
        reg.fit(X, y)
        assert all(k in reg.metrics for k in ['R²', 'R²_adj', 'RMSE', 'MAE', 'MSE'])
    
    def test_rmse_less_than_std(self, prepared_data):
        """Le RMSE doit être inférieur à l'écart-type brut des y."""
        X, y, _ = prepared_data
        reg = RegressionLinéaire(feature_names=X.columns.tolist())
        reg.fit(X, y)
        assert reg.metrics['RMSE'] <= float(y.std())
