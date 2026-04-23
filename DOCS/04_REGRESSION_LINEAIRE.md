# INF 232 EC2 — RÉGRESSION LINÉAIRE SIMPLE & MULTIPLE
## Implémentation, Diagnostics et Diagrammes

---

## 1. CADRE THÉORIQUE FORMEL

### 1.1 Modèle de Régression Linéaire Multiple

Le modèle général s'écrit :

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₖxₖ + ε
```

Où :
- `y` = note_finale (variable expliquée)
- `xᵢ` = variables explicatives (temps d'étude, connexions, etc.)
- `βᵢ` = coefficients de régression (estimés par OLS)
- `ε ~ N(0, σ²)` = terme d'erreur

### 1.2 Estimation par Moindres Carrés Ordinaires (OLS)

L'estimateur OLS minimise :

```
SSR = Σᵢ (yᵢ - ŷᵢ)² = Σᵢ εᵢ²
```

La solution matricielle est : **β̂ = (XᵀX)⁻¹ Xᵀy**

### 1.3 Hypothèses de Gauss-Markov (à vérifier !)

| Hypothèse | Description | Test associé | Visualisation |
|-----------|-------------|--------------|---------------|
| H1 — Linéarité | E[ε] = 0 | Résidus vs valeurs ajustées | Scatter résidus |
| H2 — Homoscédasticité | Var(ε) = σ² constante | Breusch-Pagan | Scale-Location plot |
| H3 — Indépendance | Cov(εᵢ, εⱼ) = 0 | Durbin-Watson | ACF résidus |
| H4 — Normalité des résidus | ε ~ N(0, σ²) | Shapiro-Wilk | QQ-plot |
| H5 — Non-colinéarité | VIF < 10 | VIF | Heatmap corrélation |

### 1.4 Métriques d'Évaluation

```
R²  = 1 - (SS_res / SS_tot)          # Coefficient de détermination
R²_adj = 1 - (1-R²)(n-1)/(n-K-1)    # R² ajusté (pénalise les variables inutiles)
MSE = (1/n) Σ (yᵢ - ŷᵢ)²             # Erreur quadratique moyenne
RMSE = √MSE                           # Racine de l'erreur quadratique
MAE = (1/n) Σ |yᵢ - ŷᵢ|              # Erreur absolue moyenne
```

---

## 2. MODULE `src/models.py` — CODE COMPLET

```python
# ============================================================
# src/models.py
# INF 232 EC2 — Régression Linéaire Simple et Multiple
# ============================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import shapiro, probplot
import joblib
from pathlib import Path
from typing import Dict, Tuple, Any


class RegressionLinéaire:
    """
    Classe encapsulant la régression linéaire simple et multiple.
    Implémente l'entraînement, l'évaluation, les diagnostics et
    toutes les visualisations associées.
    
    Attributes
    ----------
    model : LinearRegression
        Modèle sklearn entraîné.
    X_train, X_test, y_train, y_test : arrays
        Données de train et test.
    y_pred : array
        Prédictions sur le test set.
    metrics : dict
        Métriques d'évaluation (R², RMSE, MAE).
    """
    
    def __init__(
        self,
        feature_names: list,
        target_name: str = 'note_finale',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> None:
        self.feature_names = feature_names
        self.target_name = target_name
        self.test_size = test_size
        self.random_state = random_state
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    # ── Entraînement ─────────────────────────────────────────
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RegressionLinéaire':
        """
        Entraîne le modèle de régression linéaire.
        
        Parameters
        ----------
        X : pd.DataFrame
            Matrice des features.
        y : pd.Series
            Vecteur cible (note_finale).
        
        Returns
        -------
        self
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.residuals = self.y_test.values - self.y_pred
        self.is_fitted = True
        self._compute_metrics()
        return self
    
    def _compute_metrics(self) -> None:
        """Calcule et stocke les métriques d'évaluation."""
        n = len(self.y_test)
        k = len(self.feature_names)
        r2 = r2_score(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        
        self.metrics = {
            'R²':       round(r2, 4),
            'R²_adj':   round(1 - (1 - r2) * (n - 1) / (n - k - 1), 4),
            'RMSE':     round(np.sqrt(mse), 4),
            'MAE':      round(mean_absolute_error(self.y_test, self.y_pred), 4),
            'MSE':      round(mse, 4),
        }
    
    def print_summary(self) -> None:
        """Affiche un résumé complet du modèle."""
        if not self.is_fitted:
            raise RuntimeError("Modèle non entraîné. Appelez fit() d'abord.")
        
        print("=" * 55)
        print("      RÉSUMÉ DE LA RÉGRESSION LINÉAIRE MULTIPLE")
        print("=" * 55)
        print(f"  Cible      : {self.target_name}")
        print(f"  Nb features: {len(self.feature_names)}")
        print(f"  Train size : {len(self.X_train)} obs.")
        print(f"  Test size  : {len(self.X_test)} obs.")
        print("-" * 55)
        print("  COEFFICIENTS :")
        print(f"  Intercept (β₀) = {self.model.intercept_:.4f}")
        for name, coef in zip(self.feature_names, self.model.coef_):
            print(f"  β({name:30s}) = {coef:+.4f}")
        print("-" * 55)
        print("  MÉTRIQUES :")
        for k, v in self.metrics.items():
            print(f"  {k:10s} = {v}")
        print("=" * 55)
    
    def get_coefficients_df(self) -> pd.DataFrame:
        """Retourne les coefficients sous forme de DataFrame trié."""
        coefs = pd.DataFrame({
            'Feature':      self.feature_names,
            'Coefficient':  self.model.coef_,
            'Abs_Coef':     np.abs(self.model.coef_)
        }).sort_values('Abs_Coef', ascending=False)
        return coefs
    
    # ── Diagnostics des hypothèses ───────────────────────────
    
    def test_normality(self) -> Dict[str, float]:
        """
        Test de Shapiro-Wilk pour la normalité des résidus.
        H0 : Les résidus suivent une loi normale.
        H1 rejetée si p-value < 0.05.
        """
        stat, p_value = shapiro(self.residuals[:200])  # Shapiro max 200 obs
        result = {
            'statistic': round(stat, 4),
            'p_value':   round(p_value, 4),
            'normal':    p_value > 0.05
        }
        print(f"Shapiro-Wilk : W={stat:.4f}, p={p_value:.4f}")
        print(f"→ Normalité {'vérifiée <img src=app/static/icons/check.svg alt=check width=18/>' if result['normal'] else 'douteuse <img src=app/static/icons/warning.svg alt=warning width=18/>'}")
        return result
    
    def test_durbin_watson(self) -> float:
        """
        Calcule le statistique de Durbin-Watson pour l'autocorrélation.
        DW ≈ 2 → pas d'autocorrélation
        DW < 1.5 → autocorrélation positive
        DW > 2.5 → autocorrélation négative
        """
        from statsmodels.stats.stattools import durbin_watson
        dw = durbin_watson(self.residuals)
        print(f"Durbin-Watson = {dw:.3f}")
        if 1.5 <= dw <= 2.5:
            print("→ Pas d'autocorrélation détectée <img src=app/static/icons/check.svg alt=check width=18/>")
        else:
            print("→ Autocorrélation possible <img src=app/static/icons/warning.svg alt=warning width=18/>")
        return dw
    
    # ── Visualisations ───────────────────────────────────────
    
    def plot_regression_line(self, x_col_idx: int = 0) -> plt.Figure:
        """
        DIAGRAMME 7 — Droite de régression simple.
        Nuage de points + droite + IC 95%.
        """
        fig, ax = plt.subplots(figsize=(9, 6))
        
        x_vals = self.X_test.iloc[:, x_col_idx].values
        sort_idx = np.argsort(x_vals)
        
        ax.scatter(x_vals, self.y_test, alpha=0.5, color='#2196F3',
                   label='Observations', s=40, zorder=2)
        ax.scatter(x_vals, self.y_pred, alpha=0.4, color='#FF5722',
                   marker='^', label='Prédictions', s=40, zorder=3)
        
        # Droite
        m = self.model.coef_[x_col_idx]
        b = self.model.intercept_
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, m * x_line + b, 'darkblue', linewidth=2,
                label=f'ŷ = {b:.2f} + {m:.2f}·x  (R²={self.metrics["R²"]:.3f})')
        
        # Résidus (traits de connexion)
        for xi, yi, yp in zip(x_vals[:30], self.y_test.values[:30], self.y_pred[:30]):
            ax.plot([xi, xi], [yi, yp], color='gray', alpha=0.3, linewidth=0.8)
        
        ax.set_xlabel(self.feature_names[x_col_idx], fontsize=11)
        ax.set_ylabel(self.target_name, fontsize=11)
        ax.set_title('Régression Linéaire — Ajustement sur données test', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.axhline(10, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        return fig
    
    def plot_residuals_vs_fitted(self) -> plt.Figure:
        """
        DIAGRAMME 9 — Résidus vs Valeurs ajustées.
        Permet de vérifier l'homoscédasticité et la linéarité.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(self.y_pred, self.residuals, alpha=0.5, color='#2196F3', s=30)
        ax.axhline(0, color='red', linewidth=1.5, linestyle='--')
        
        # Courbe lissée (LOWESS approximation)
        from scipy.ndimage import uniform_filter1d
        sorted_idx = np.argsort(self.y_pred)
        smooth = uniform_filter1d(self.residuals[sorted_idx], size=30)
        ax.plot(self.y_pred[sorted_idx], smooth, color='orange',
                linewidth=2, label='Tendance (lissée)')
        
        ax.set_xlabel('Valeurs ajustées (ŷ)', fontsize=11)
        ax.set_ylabel('Résidus (y - ŷ)', fontsize=11)
        ax.set_title('Diagnostic — Résidus vs Valeurs Ajustées', fontsize=13, fontweight='bold')
        ax.legend()
        ax.text(0.02, 0.97, "→ Un nuage aléatoire autour de 0 confirme l'homoscédasticité",
                transform=ax.transAxes, fontsize=9, color='gray', va='top')
        plt.tight_layout()
        return fig
    
    def plot_qq_plot(self) -> plt.Figure:
        """
        DIAGRAMME 10 — QQ-Plot des résidus.
        Vérifie la normalité des résidus.
        """
        fig, ax = plt.subplots(figsize=(7, 7))
        (osm, osr), (slope, intercept, r) = probplot(self.residuals, dist="norm")
        
        ax.scatter(osm, osr, color='#2196F3', alpha=0.6, s=25, label='Quantiles empiriques')
        ax.plot(osm, slope * np.array(osm) + intercept, 'r-', linewidth=2, label='Droite théorique N(0,1)')
        
        ax.set_xlabel('Quantiles théoriques N(0,1)', fontsize=11)
        ax.set_ylabel('Quantiles empiriques des résidus', fontsize=11)
        ax.set_title('QQ-Plot — Vérification de la normalité des résidus', fontsize=12, fontweight='bold')
        ax.legend()
        
        # Annotation R²
        ax.text(0.05, 0.95, f'R (normalité) = {r:.3f}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        return fig
    
    def plot_learning_curve(self) -> plt.Figure:
        """
        DIAGRAMME 11 — Courbe d'apprentissage.
        Évalue le surapprentissage (overfitting) vs sous-apprentissage.
        """
        import numpy as np
        from sklearn.model_selection import learning_curve
        
        X_all = pd.concat([self.X_train, self.X_test])
        y_all = pd.concat([pd.Series(self.y_train), pd.Series(self.y_test)])
        
        train_sizes, train_scores, val_scores = learning_curve(
            LinearRegression(), X_all, y_all,
            cv=5, scoring='r2',
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1
        )
        
        fig, ax = plt.subplots(figsize=(9, 5))
        
        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)
        
        ax.plot(train_sizes, train_mean, 'o-', color='#2196F3', label='Score entraînement')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#2196F3')
        
        ax.plot(train_sizes, val_mean, 'o-', color='#FF5722', label='Score validation croisée (CV-5)')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#FF5722')
        
        ax.set_xlabel("Taille de l'ensemble d'entraînement", fontsize=11)
        ax.set_ylabel('Score R²', fontsize=11)
        ax.set_title('Courbe d\'apprentissage — Régression Linéaire', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1.05])
        plt.tight_layout()
        return fig
    
    def plot_coefficients_bar(self) -> plt.Figure:
        """
        DIAGRAMME — Importance des coefficients (barres horizontales).
        """
        coefs = self.get_coefficients_df()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#4CAF50' if c >= 0 else '#FF5722' for c in coefs['Coefficient']]
        ax.barh(coefs['Feature'], coefs['Coefficient'], color=colors, edgecolor='white')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Coefficient (βᵢ)', fontsize=11)
        ax.set_title('Coefficients de Régression Linéaire Multiple', fontsize=13, fontweight='bold')
        
        # Annotation
        for i, (v, row) in enumerate(zip(coefs['Coefficient'], coefs.itertuples())):
            ax.text(v + 0.01 * np.sign(v), i, f'{v:+.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def save(self, filepath: str = "data/models/regression_model.pkl") -> None:
        """Sérialise le modèle entraîné."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"<img src=app/static/icons/check.svg alt=check width=18/> Modèle sauvegardé : {filepath}")


# ════════════════════════════════════════════════════════════
# SCRIPT D'EXÉCUTION PRINCIPAL
# ════════════════════════════════════════════════════════════

def run_regression_pipeline(df_clean: pd.DataFrame) -> RegressionLinéaire:
    """
    Pipeline complet de régression linéaire.
    
    Parameters
    ----------
    df_clean : pd.DataFrame
        Dataset nettoyé avec variables encodées.
    
    Returns
    -------
    RegressionLinéaire
        Modèle entraîné et évalué.
    """
    from src.data_cleaning import get_feature_matrix
    
    print("<img src=app/static/icons/loop.svg alt=loop width=18/> Démarrage pipeline régression linéaire...")
    
    X, y_reg, _ = get_feature_matrix(df_clean)
    
    # Régression simple (une seule variable)
    print("\n--- RÉGRESSION SIMPLE : temps_etude_hebdo → note_finale ---")
    reg_simple = RegressionLinéaire(
        feature_names=['temps_etude_hebdo'],
        target_name='note_finale'
    )
    reg_simple.fit(X[['temps_etude_hebdo']], y_reg)
    reg_simple.print_summary()
    
    # Régression multiple (toutes les features)
    print("\n--- RÉGRESSION MULTIPLE ---")
    feature_cols = [
        'temps_etude_hebdo', 'nb_devoirs_rendus', 'exercices_completes_pct',
        'nb_connexions_semaine', 'score_motivation', 'nombre_absences',
        'videos_vues_pct', 'note_mi_parcours'
    ]
    reg_multi = RegressionLinéaire(
        feature_names=feature_cols,
        target_name='note_finale'
    )
    reg_multi.fit(X[feature_cols], y_reg)
    reg_multi.print_summary()
    
    # Diagnostics
    print("\n--- DIAGNOSTICS ---")
    reg_multi.test_normality()
    reg_multi.test_durbin_watson()
    
    # Validation croisée
    scores_cv = cross_val_score(
        LinearRegression(), X[feature_cols], y_reg,
        cv=5, scoring='r2'
    )
    print(f"\nCV-5 R² : {scores_cv.mean():.3f} ± {scores_cv.std():.3f}")
    
    # Sauvegarder
    reg_multi.save("data/models/regression_model.pkl")
    
    return reg_multi


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/processed/elearning_clean.csv")
    model = run_regression_pipeline(df)
```

---

## 3. TABLEAU DE RÉSULTATS ATTENDUS

| Métrique | Régression Simple | Régression Multiple |
|----------|-------------------|---------------------|
| R² | ~0.45 – 0.55 | **~0.72 – 0.82** |
| R²_adj | ~0.44 – 0.54 | ~0.71 – 0.81 |
| RMSE | ~2.2 – 2.5 pts | **~1.4 – 1.8 pts** |
| MAE | ~1.8 – 2.2 pts | ~1.1 – 1.5 pts |

**Interprétation :** La régression multiple explique environ 75% de la variance de la note finale, avec un gain significatif par rapport à la régression simple. Le RMSE de ~1.6 points signifie que nos prédictions sont en moyenne à ±1.6/20 de la vraie note.

---

## 4. ÉQUATION DE RÉGRESSION ESTIMÉE (exemple théorique)

```
note_finale = 1.50
            + 0.40 × temps_etude_hebdo
            + 0.30 × nb_devoirs_rendus
            + 0.07 × exercices_completes_pct
            + 0.12 × nb_connexions_semaine
            + 0.18 × score_motivation
            − 0.20 × nombre_absences
            + 0.04 × videos_vues_pct
            + 0.60 × note_mi_parcours
            + ε
```

**Lecture des coefficients :**
- 1 heure d'étude supplémentaire par semaine → +0.40 point à la note finale
- 1 absence supplémentaire → −0.20 point à la note finale
- 1 devoir rendu en plus → +0.30 point à la note finale

---

*Document 5/10 — INF 232 EC2*
