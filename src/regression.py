# ============================================================
# src/regression.py
# RegressionLinéaire minimal adapté pour les tests
# ============================================================

from typing import List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class RegressionLinéaire:
    """Classe légère de régression linéaire compatible avec les tests.

    Fournit : fit(), metrics dict avec clés 'R²','R²_adj','RMSE','MAE','MSE',
    attributs y_pred, y_test, residuals, is_fitted.
    """

    def __init__(
        self,
        feature_names: List[str],
        target_name: str = 'note_finale',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> None:
        self.feature_names = feature_names
        self.target_name = target_name
        self.test_size = test_size
        self.random_state = random_state
        self.model = LinearRegression()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RegressionLinéaire':
        """Entraîne le modèle sur X/y et conserve les prédictions sur le test set."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.residuals = self.y_test.values - self.y_pred
        self.is_fitted = True
        self._compute_metrics()
        return self

    def _compute_metrics(self) -> None:
        n = len(self.y_test)
        k = len(self.feature_names) if self.feature_names is not None else 0
        r2 = float(r2_score(self.y_test, self.y_pred))
        mse = float(mean_squared_error(self.y_test, self.y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(self.y_test, self.y_pred))

        if n - k - 1 > 0:
            r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        else:
            r2_adj = r2

        self.metrics = {
            'R²':     round(r2, 4),
            'R²_adj': round(r2_adj, 4),
            'RMSE':   round(rmse, 4),
            'MAE':    round(mae, 4),
            'MSE':    round(mse, 4)
        }


if __name__ == "__main__":
    # Petit test local
    from src.data_generation import generate_student_dataset
    from src.data_cleaning import encode_categorical_variables, get_feature_matrix

    df = generate_student_dataset(n_samples=200)
    df_enc = encode_categorical_variables(df)
    X, y_reg, _ = get_feature_matrix(df_enc)
    cols = ['temps_etude_hebdo', 'nb_devoirs_rendus', 'exercices_completes_pct']
    reg = RegressionLinéaire(feature_names=cols)
    reg.fit(X[cols], y_reg)
    print(reg.metrics)