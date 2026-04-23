# ============================================================
# src/data_cleaning.py
# INF 232 EC2 — Nettoyage et prétraitement des données
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from pathlib import Path
import joblib
from typing import Tuple


def load_raw_data(filepath: str = "data/raw/elearning_dataset.csv") -> pd.DataFrame:
    """Charge les données brutes depuis un fichier CSV."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")
    df = pd.read_csv(path, encoding='utf-8')
    print(f"✅ Données chargées : {df.shape}")
    return df


def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Génère un rapport des valeurs manquantes."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({
        'nb_manquant': missing,
        'pct_manquant': pct
    }).query('nb_manquant > 0').sort_values('pct_manquant', ascending=False)
    return report


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Traite les valeurs manquantes."""
    df_clean = df.copy()

    num_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

    # Exclure les colonnes cibles et identifiants
    num_cols = [c for c in num_cols if c not in ['etudiant_id', 'reussite']]

    # Imputation numérique
    num_imputer = SimpleImputer(strategy='median')
    df_clean[num_cols] = num_imputer.fit_transform(df_clean[num_cols])

    # Imputation catégorielle
    for col in cat_cols:
        if df_clean[col].isnull().any():
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

    return df_clean


def remove_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
    """Supprime les outliers selon la méthode IQR."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    mask = df[col].between(lower, upper)
    n_removed = (~mask).sum()
    if n_removed > 0:
        print(f"  [{col}] {n_removed} outliers supprimés")
    return df[mask].copy()


def encode_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Encode les variables catégorielles."""
    df_enc = df.copy()

    # Encodage ordinal niveau d'études
    niveau_map = {'L1': 1, 'L2': 2, 'L3': 3, 'M1': 4, 'M2': 5}
    df_enc['niveau_etudes_num'] = df_enc['niveau_etudes'].map(niveau_map)

    # Encodage genre
    df_enc['genre_num'] = (df_enc['genre'] == 'M').astype(int)

    # Encodage revenu famille
    revenu_map = {'Bas': 0, 'Moyen': 1, 'Élevé': 2}
    df_enc['revenu_famille_num'] = df_enc['revenu_famille'].map(revenu_map)

    # Encodage accès internet
    internet_map = {'Limité': 0, 'Instable': 1, 'Stable': 2}
    df_enc['acces_internet_num'] = df_enc['acces_internet'].map(internet_map)

    return df_enc


def get_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Retourne la matrice de features X et les vecteurs cibles."""
    feature_cols = [
        'age', 'genre_num', 'niveau_etudes_num',
        'nb_connexions_semaine', 'temps_etude_hebdo',
        'exercices_completes_pct', 'videos_vues_pct',
        'participation_forums', 'nb_devoirs_rendus',
        'revenu_famille_num', 'score_motivation',
        'nombre_absences', 'acces_internet_num',
        'note_mi_parcours'
    ]
    X = df[feature_cols].copy()
    y_reg = df['note_finale'].copy()
    y_clf = df['reussite'].copy()
    return X, y_reg, y_clf


def full_pipeline(filepath: str = "data/raw/elearning_dataset.csv") -> pd.DataFrame:
    """Exécute le pipeline complet de nettoyage."""
    print("🔧 Démarrage du pipeline de nettoyage...")
    df = load_raw_data(filepath)

    print("\n📊 Rapport valeurs manquantes :")
    missing_report = report_missing_values(df)
    if len(missing_report) > 0:
        print(missing_report.to_string())
    else:
        print("Aucune valeur manquante détectée.")

    df = handle_missing_values(df)
    df = encode_categorical_variables(df)

    # Supprimer les doublons
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"\n🔍 Doublons supprimés : {n_before - len(df)}")

    # Sauvegarder
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/elearning_clean.csv", index=False)
    print(f"\n✅ Dataset nettoyé sauvegardé ({df.shape})")

    return df


def save_processed_data(df: pd.DataFrame, scaler: StandardScaler = None) -> None:
    """Sauvegarde les données traitées et le scaler."""
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/elearning_clean.csv", index=False)

    if scaler is not None:
        Path("data/models").mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, "data/models/scaler.pkl")
        print("✅ Scaler sauvegardé : data/models/scaler.pkl")

    print(f"✅ Données traitées sauvegardées : data/processed/elearning_clean.csv")


if __name__ == "__main__":
    df = full_pipeline()
    print("\n📊 Aperçu des données nettoyées :")
    print(df.head().to_string())