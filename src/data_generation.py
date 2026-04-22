# ============================================================
# src/data_generation.py
# INF 232 EC2 — Génération automatique du jeu de données
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path


def generate_student_dataset(
    n_samples: int = 500,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Génère un jeu de données synthétique réaliste de comportements
    d'apprentissage d'étudiants sur une plateforme e-learning.

    Parameters
    ----------
    n_samples : int
        Nombre d'observations à générer (défaut : 500)
    random_state : int
        Graine aléatoire pour la reproductibilité

    Returns
    -------
    pd.DataFrame
        DataFrame avec 15 variables et n_samples observations.
    """
    rng = np.random.default_rng(random_state)

    # 1. Identifiants
    etudiant_id = np.arange(1, n_samples + 1)

    # 2. Variables socio-démographiques
    age = np.clip(
        rng.normal(loc=22, scale=5, size=n_samples).astype(int),
        17, 45
    )

    genre = rng.choice(['M', 'F'], size=n_samples, p=[0.52, 0.48])

    niveau_etudes = rng.choice(
        ['L1', 'L2', 'L3', 'M1', 'M2'],
        size=n_samples,
        p=[0.30, 0.30, 0.20, 0.15, 0.05]
    )

    # 3. Variables comportementales
    nb_connexions_semaine = np.clip(
        rng.normal(loc=8, scale=3, size=n_samples),
        1.0, 21.0
    ).round(1)

    temps_etude_hebdo = np.clip(
        rng.normal(loc=12, scale=5, size=n_samples),
        0.0, 35.0
    ).round(1)

    exercices_completes_pct = np.clip(
        rng.beta(a=5, b=2, size=n_samples) * 100,
        0.0, 100.0
    ).round(1)

    videos_vues_pct = np.clip(
        rng.beta(a=4, b=2, size=n_samples) * 100,
        0.0, 100.0
    ).round(1)

    participation_forums = rng.poisson(lam=5, size=n_samples)

    nb_devoirs_rendus = rng.binomial(n=10, p=0.75, size=n_samples)

    # 4. Variables socio-économiques
    revenu_famille = rng.choice(
        ['Bas', 'Moyen', 'Élevé'],
        size=n_samples,
        p=[0.30, 0.50, 0.20]
    )

    score_motivation = np.clip(
        rng.normal(loc=6.5, scale=1.8, size=n_samples),
        1.0, 10.0
    ).round(1)

    # 5. Variables contextuelles
    nombre_absences = rng.poisson(lam=3, size=n_samples)

    acces_internet = rng.choice(
        ['Stable', 'Instable', 'Limité'],
        size=n_samples,
        p=[0.60, 0.25, 0.15]
    )

    # 6. Génération des variables cibles
    # Note mi-parcours
    note_mi_parcours = (
        2.0
        + 0.35 * temps_etude_hebdo
        + 0.08 * exercices_completes_pct
        + 0.10 * score_motivation
        - 0.15 * nombre_absences
        + rng.normal(0, 1.5, n_samples)
    )
    note_mi_parcours = np.clip(note_mi_parcours, 0.0, 20.0).round(2)

    # Note finale
    note_finale = (
        -7.5
        + 0.22 * temps_etude_hebdo
        + 0.03 * exercices_completes_pct
        + 0.04 * nb_connexions_semaine
        + 0.12 * nb_devoirs_rendus
        + 0.40 * score_motivation
        - 0.50 * nombre_absences
        + 0.01 * videos_vues_pct
        + 0.90 * note_mi_parcours
        + rng.normal(0, 4.0, n_samples)
    )
    note_finale = np.clip(note_finale, 0.0, 20.0).round(2)

    # Variable cible binaire : réussite
    reussite = (note_finale >= 10.0).astype(int)

    # 7. Construction du DataFrame
    df = pd.DataFrame({
        'etudiant_id': etudiant_id,
        'age': age,
        'genre': genre,
        'niveau_etudes': niveau_etudes,
        'nb_connexions_semaine': nb_connexions_semaine,
        'temps_etude_hebdo': temps_etude_hebdo,
        'exercices_completes_pct': exercices_completes_pct,
        'videos_vues_pct': videos_vues_pct,
        'participation_forums': participation_forums,
        'nb_devoirs_rendus': nb_devoirs_rendus,
        'revenu_famille': revenu_famille,
        'score_motivation': score_motivation,
        'nombre_absences': nombre_absences,
        'acces_internet': acces_internet,
        'note_mi_parcours': note_mi_parcours,
        'note_finale': note_finale,
        'reussite': reussite,
    })

    return df


def inject_missing_values(
    df: pd.DataFrame,
    missing_rate: float = 0.03,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Injecte des valeurs manquantes pour simuler un cas réel.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset original propre.
    missing_rate : float
        Taux de valeurs manquantes à injecter.
    random_state : int
        Graine pour la reproductibilité.

    Returns
    -------
    pd.DataFrame
        Dataset avec valeurs manquantes.
    """
    rng = np.random.default_rng(random_state)
    df_dirty = df.copy()

    cols_to_corrupt = [
        'temps_etude_hebdo', 'score_motivation',
        'nb_connexions_semaine', 'participation_forums'
    ]

    for col in cols_to_corrupt:
        n_missing = int(missing_rate * len(df_dirty))
        missing_indices = rng.choice(len(df_dirty), size=n_missing, replace=False)
        df_dirty.loc[missing_indices, col] = np.nan

    return df_dirty


def save_dataset(df: pd.DataFrame, output_dir: str = "data/raw") -> None:
    """
    Sauvegarde le dataset dans un fichier CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Le dataset à sauvegarder.
    output_dir : str
        Répertoire de destination.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "elearning_dataset.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Dataset sauvegardé : {output_path}")
    print(f"   Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"   Taux de réussite : {df['reussite'].mean():.1%}")


if __name__ == "__main__":
    df = generate_student_dataset(n_samples=500, random_state=42)
    save_dataset(df)
    print("\n📊 Aperçu des 5 premières observations :")
    print(df.head().to_string())
    print("\n📈 Statistiques descriptives :")
    print(df.describe().round(2).to_string())