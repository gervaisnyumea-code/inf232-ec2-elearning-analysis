# ============================================================
# src/analysis.py
# INF 232 EC2 — Analyse exploratoire (EDA)
# ============================================================

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any


def get_descriptive_stats(df: pd.DataFrame, cols: Optional[list] = None) -> pd.DataFrame:
    """Retourne les statistiques descriptives complètes."""
    if cols:
        return df[cols].describe()
    return df.describe()


def get_distribution_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Retourne la distribution d'une variable catégorielle."""
    counts = df[col].value_counts()
    pct = (counts / len(df) * 100).round(2)
    return pd.DataFrame({'Effectif': counts, 'Pourcentage': pct})


def get_correlation_with_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Retourne les corrélations avec la variable cible."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]
    correlations = df[numeric_cols].corr()[target_col].sort_values(ascending=False)
    return correlations.to_frame('Corrélation')


def get_skewness_kurtosis(df: pd.DataFrame, col: str) -> dict:
    """Retourne l'asymétrie et l'aplatissement."""
    from scipy.stats import skew, kurtosis
    return {
        'skewness': skew(df[col].dropna()),
        'kurtosis': kurtosis(df[col].dropna())
    }


def get_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
    """Détecte les outliers avec la méthode IQR."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    return outliers


def get_summary_by_group(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Retourne un résumé groupé par catégorie."""
    return df.groupby(group_col)[value_col].agg(['mean', 'std', 'min', 'max']).round(2)


def get_missing_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse complète des valeurs manquantes."""
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)
    unique = df.nunique()

    report = pd.DataFrame({
        'nb_manquant': missing,
        'pct_manquant': pct,
        'nb_valeurs_uniques': unique,
        'type': df.dtypes
    })
    return report.sort_values('pct_manquant', ascending=False)


def compute_descriptive_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcule l'ensemble des statistiques descriptives du dataset.
    
    Returns
    -------
    dict
        Dictionnaire avec clés : summary, skewness, kurtosis, 
        correlations, categorical_counts.
    """
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    stats = {
        'summary':    df[num_cols].describe().round(3),
        'skewness':   df[num_cols].skew().round(3),
        'kurtosis':   df[num_cols].kurtosis().round(3),
        'correlations': df[num_cols].corr().round(3),
        'categorical_counts': {
            col: df[col].value_counts(normalize=True).round(3)
            for col in cat_cols
        }
    }
    return stats


def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Détecte les outliers via le Z-score.
    
    Returns
    -------
    pd.DataFrame
        Rapport des outliers par colonne numérique.
    """
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    results = []
    for col in num_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        n_outliers = (z_scores > threshold).sum()
        if n_outliers > 0:
            results.append({'colonne': col, 'nb_outliers': int(n_outliers),
                           'pct': round(n_outliers/len(df)*100, 2)})
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test with cleaned data
    from src.data_cleaning import load_raw_data, full_pipeline

    df = full_pipeline()
    print("📊 Statistiques descriptives :")
    print(get_descriptive_stats(df).to_string())
    print("\n📈 Corrélations avec note_finale :")
    print(get_correlation_with_target(df, 'note_finale').to_string())