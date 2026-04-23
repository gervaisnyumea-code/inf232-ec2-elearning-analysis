# ============================================================
# src/models.py
# INF 232 EC2 — Modèles de régression et classification
# ============================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score, confusion_matrix, roc_curve, auc, classification_report
import joblib
from pathlib import Path
from typing import Tuple, Optional
from .regression import RegressionLinéaire


# ============================================================
# RÉGRESSION LINÉAIRE
# ============================================================

class RegressionModel:
    """Modèle de régression linéaire avec diagnostics."""

    def __init__(self, model_type: str = 'linear'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = self._get_model()
        self.feature_names = None

    def _get_model(self):
        if self.model_type == 'ridge':
            return Ridge(alpha=1.0)
        elif self.model_type == 'lasso':
            return Lasso(alpha=0.1)
        return LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RegressionModel':
        """Entraîne le modèle."""
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prédit les valeurs."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Évalue le modèle et retourne les métriques."""
        y_pred = self.predict(X)
        return {
            'R2': r2_score(y, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'MAE': mean_absolute_error(y, y_pred)
        }

    def get_residuals(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Calcule les résidus."""
        y_pred = self.predict(X)
        return y - y_pred

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
        """Validation croisée."""
        X_scaled = self.scaler.fit_transform(X)
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='r2')
        return {
            'mean_r2': scores.mean(),
            'std_r2': scores.std(),
            'scores': scores
        }

    def learning_curve_data(self, X: pd.DataFrame, y: pd.Series,
                            train_sizes: np.ndarray = None) -> dict:
        """Calcule les données pour la courbe d'apprentissage."""
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        X_scaled = self.scaler.fit_transform(X)
        train_sizes_abs, train_scores, test_scores = learning_curve(
            self.model, X_scaled, y, train_sizes=train_sizes, cv=5, scoring='r2'
        )

        return {
            'train_sizes': train_sizes_abs,
            'train_scores': train_scores.mean(axis=1),
            'test_scores': test_scores.mean(axis=1)
        }

    def save(self, filepath: str) -> None:
        """Sauvegarde le modèle."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'feature_names': self.feature_names}, filepath)
        print(f"✅ Modèle sauvegardé : {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'RegressionModel':
        """Charge un modèle."""
        data = joblib.load(filepath)
        model = cls(data['model'].__class__.__name__.lower())
        model.model = data['model']
        model.scaler = data['scaler']
        model.feature_names = data['feature_names']
        return model


# ============================================================
# CLASSIFICATION
# ============================================================

class ClassificationModel:
    """Modèle de classification binaire."""

    def __init__(self, model_type: str = 'rf'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = self._get_model()
        self.feature_names = None

    def _get_model(self):
        if self.model_type == 'knn':
            return KNeighborsClassifier(n_neighbors=5)
        return RandomForestClassifier(n_estimators=100, random_state=42)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ClassificationModel':
        """Entraîne le modèle."""
        # Store feature names to allow safe prediction later
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prédit les classes."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Prédit les probabilités."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Évalue le modèle."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)

        return {
            'accuracy': (y_pred == y).mean(),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'classification_report': classification_report(y, y_pred)
        }

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
        """Validation croisée."""
        X_scaled = self.scaler.fit_transform(X)
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
        return {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores
        }

    def feature_importance(self, X: pd.DataFrame) -> np.ndarray:
        """Retourne l'importance des features."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None

    def save(self, filepath: str) -> None:
        """Sauvegarde le modèle."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        # Save model along with scaler and feature names for safe loading/prediction
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'feature_names': getattr(self, 'feature_names', None)}, filepath)
        print(f"✅ Modèle sauvegardé : {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'ClassificationModel':
        """Charge un modèle de classification (si sauvegardé sous forme dict)."""
        data = joblib.load(filepath)
        inst = cls()
        inst.model = data.get('model')
        inst.scaler = data.get('scaler')
        inst.feature_names = data.get('feature_names', None)
        return inst


# ============================================================
# CLUSTERING
# ============================================================

class ClusteringModel:
    """Modèles de clustering non-supervisé."""

    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cah = AgglomerativeClustering(n_clusters=n_clusters)

    def fit_kmeans(self, X: pd.DataFrame) -> np.ndarray:
        """K-Means clustering."""
        self.kmeans.fit(X)
        return self.kmeans.labels_

    def fit_cah(self, X: pd.DataFrame) -> np.ndarray:
        """CAH clustering."""
        return self.cah.fit_predict(X)

    def get_silhouette(self, X: pd.DataFrame, labels: np.ndarray) -> float:
        """Calcule le score silhouette."""
        return silhouette_score(X, labels)

    def elbow_method(self, X: pd.DataFrame, max_k: int = 10) -> list:
        """Méthode du coude pour déterminer k optimal."""
        inertias = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        return inertias


# ============================================================
# RÉDUCTION DE DIMENSION
# ============================================================

class DimensionalityReduction:
    """Réduction de dimension (ACP, t-SNE, LDA)."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.tsne = TSNE(n_components=n_components, random_state=42)
        self.lda = LDA(n_components=n_components)

    def fit_pca(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """ACP - retourne les composantes et la variance expliquée."""
        X_scaled = StandardScaler().fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        return X_pca, self.pca.explained_variance_ratio_

    def fit_tsne(self, X: pd.DataFrame, perplexity: int = 30) -> np.ndarray:
        """t-SNE."""
        X_scaled = StandardScaler().fit_transform(X)
        self.tsne = TSNE(n_components=self.n_components, random_state=42, perplexity=perplexity)
        return self.tsne.fit_transform(X_scaled)

    def fit_lda(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """LDA supervisée."""
        X_scaled = StandardScaler().fit_transform(X)
        return self.lda.fit_transform(X_scaled, y)


# ============================================================
# PIPELINE COMPLÈT
# ============================================================

def train_regression_pipeline(X: pd.DataFrame, y: pd.Series,
                               model_type: str = 'linear') -> Tuple[RegressionModel, dict]:
    """Entraîne le pipeline de régression complet."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RegressionModel(model_type)
    model.fit(X_train, y_train)

    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)
    cv_metrics = model.cross_validate(X_train, y_train)

    return model, {
        'train': train_metrics,
        'test': test_metrics,
        'cv': cv_metrics
    }


def train_classification_pipeline(X: pd.DataFrame, y: pd.Series,
                                   model_type: str = 'rf') -> Tuple[ClassificationModel, dict]:
    """Entraîne le pipeline de classification complet."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = ClassificationModel(model_type)
    model.fit(X_train, y_train)

    eval_results = model.evaluate(X_test, y_test)
    cv_metrics = model.cross_validate(X_train, y_train)

    return model, {
        'test': eval_results,
        'cv': cv_metrics
    }


if __name__ == "__main__":
    from src.data_cleaning import load_raw_data, full_pipeline, get_feature_matrix

    print("🔄 Chargement et nettoyage des données...")
    df = full_pipeline()
    X, y_reg, y_clf = get_feature_matrix(df)

    print("\n📈 Entraînement modèle de régression...")
    reg_model, reg_metrics = train_regression_pipeline(X, y_reg)
    print(f"   R2 Test: {reg_metrics['test']['R2']:.4f}")
    print(f"   RMSE Test: {reg_metrics['test']['RMSE']:.4f}")

    print("\n🎯 Entraînement modèle de classification...")
    clf_model, clf_metrics = train_classification_pipeline(X, y_clf)
    print(f"   Accuracy Test: {clf_metrics['test']['accuracy']:.4f}")
    print(f"   AUC: {clf_metrics['test']['auc']:.4f}")