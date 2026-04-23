# ============================================================
# tests/test_data_cleaning.py
# ============================================================
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import generate_student_dataset
from src.data_cleaning import (
    report_missing_values, handle_missing_values,
    encode_categorical_variables, remove_outliers_iqr,
    get_feature_matrix
)


@pytest.fixture
def sample_df():
    """Dataset de test (50 obs.) partagé entre les tests."""
    return generate_student_dataset(n_samples=50, random_state=0)


@pytest.fixture
def dirty_df(sample_df):
    """Dataset avec valeurs manquantes injectées."""
    df = sample_df.copy()
    df.loc[0, 'temps_etude_hebdo'] = np.nan
    df.loc[5, 'score_motivation'] = np.nan
    return df


class TestDataGeneration:
    """Tests de la génération du dataset."""
    
    def test_dimensions(self, sample_df):
        """Le dataset doit avoir 50 lignes et 17 colonnes."""
        assert sample_df.shape == (50, 17)
    
    def test_no_missing_generated(self, sample_df):
        """Le dataset généré ne doit pas contenir de NaN."""
        assert sample_df.isnull().sum().sum() == 0
    
    def test_note_finale_range(self, sample_df):
        """Les notes finales doivent être entre 0 et 20."""
        assert sample_df['note_finale'].between(0, 20).all()
    
    def test_reussite_binary(self, sample_df):
        """La variable réussite doit être binaire (0 ou 1)."""
        assert set(sample_df['reussite'].unique()).issubset({0, 1})
    
    def test_reussite_consistent_with_note(self, sample_df):
        """La réussite doit correspondre à note >= 10."""
        expected = (sample_df['note_finale'] >= 10).astype(int)
        pd.testing.assert_series_equal(
            sample_df['reussite'],
            expected,
            check_names=False
        )
    
    def test_genre_values(self, sample_df):
        """Le genre doit être uniquement M ou F."""
        assert set(sample_df['genre'].unique()).issubset({'M', 'F'})
    
    def test_age_range(self, sample_df):
        """L'âge doit être entre 17 et 45."""
        assert sample_df['age'].between(17, 45).all()


class TestMissingValues:
    """Tests du traitement des valeurs manquantes."""
    
    def test_report_detects_missing(self, dirty_df):
        """Le rapport doit détecter les NaN injectés."""
        report = report_missing_values(dirty_df)
        assert len(report) > 0
        assert 'temps_etude_hebdo' in report.index or \
               'score_motivation' in report.index
    
    def test_handle_missing_removes_nans(self, dirty_df):
        """Après traitement, il ne doit plus y avoir de NaN."""
        df_clean = handle_missing_values(dirty_df)
        num_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
        assert df_clean[num_cols].isnull().sum().sum() == 0
    
    def test_handle_missing_preserves_length(self, dirty_df):
        """Le nettoyage ne doit pas supprimer de lignes."""
        df_clean = handle_missing_values(dirty_df)
        assert len(df_clean) == len(dirty_df)


class TestEncoding:
    """Tests de l'encodage des variables catégorielles."""
    
    def test_genre_encoded(self, sample_df):
        """Le genre doit être encodé en 0/1."""
        df_enc = encode_categorical_variables(sample_df)
        assert 'genre_num' in df_enc.columns
        assert set(df_enc['genre_num'].unique()).issubset({0, 1})
    
    def test_niveau_etudes_ordinal(self, sample_df):
        """Le niveau d'études doit être encodé ordinalement (1-5)."""
        df_enc = encode_categorical_variables(sample_df)
        assert 'niveau_etudes_num' in df_enc.columns
        assert df_enc['niveau_etudes_num'].between(1, 5).all()
    
    def test_revenu_ordinal(self, sample_df):
        """Le revenu famille doit être encodé (0-2)."""
        df_enc = encode_categorical_variables(sample_df)
        assert 'revenu_famille_num' in df_enc.columns
        assert df_enc['revenu_famille_num'].between(0, 2).all()


class TestOutlierRemoval:
    """Tests de la suppression des outliers."""
    
    def test_removes_extremes(self):
        """L'IQR doit supprimer les valeurs extrêmes."""
        df = pd.DataFrame({'values': [10, 12, 13, 11, 12, 200, 9, 11]})
        df_clean = remove_outliers_iqr(df, 'values')
        assert df_clean['values'].max() < 200
    
    def test_preserves_normal_data(self):
        """L'IQR ne doit pas supprimer les valeurs normales."""
        rng = np.random.default_rng(42)
        data = rng.normal(loc=50, scale=5, size=100)
        df = pd.DataFrame({'values': data})
        df_clean = remove_outliers_iqr(df, 'values')
        assert len(df_clean) >= 0.95 * len(df)


class TestFeatureMatrix:
    """Tests de la construction de la matrice de features."""
    
    def test_feature_matrix_shape(self, sample_df):
        """La matrice X doit avoir le bon nombre de colonnes."""
        df_enc = encode_categorical_variables(sample_df)
        X, y_reg, y_clf = get_feature_matrix(df_enc)
        assert X.shape[1] == 14  # 14 features définies
    
    def test_y_reg_range(self, sample_df):
        """Les cibles de régression doivent être dans [0, 20]."""
        df_enc = encode_categorical_variables(sample_df)
        _, y_reg, _ = get_feature_matrix(df_enc)
        assert y_reg.between(0, 20).all()
    
    def test_y_clf_binary(self, sample_df):
        """Les cibles de classification doivent être binaires."""
        df_enc = encode_categorical_variables(sample_df)
        _, _, y_clf = get_feature_matrix(df_enc)
        assert set(y_clf.unique()).issubset({0, 1})
