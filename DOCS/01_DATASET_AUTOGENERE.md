# INF 232 EC2 — REGISTRE DE DONNÉES & GÉNÉRATION AUTOMATIQUE
## Spécifications Complètes du Jeu de Données

---

## 1. CONTEXTE ET JUSTIFICATION

Le jeu de données simule les **comportements d'apprentissage de 500 étudiants** sur une plateforme e-learning sur une période d'un semestre académique (16 semaines). Chaque observation représente un étudiant avec ses caractéristiques socio-démographiques, ses comportements numériques et ses résultats académiques.

**Source :** Données auto-générées avec `numpy.random` et `faker` — distribution réaliste inspirée des études empiriques sur l'apprentissage numérique.

---

## 2. REGISTRE COMPLET DES VARIABLES

### 2.1 Variables Socio-démographiques (4 variables)

| Variable | Type | Plage / Modalités | Distribution | Rôle |
|----------|------|-------------------|--------------|------|
| `etudiant_id` | int | 1 → 500 | Séquentiel | Identifiant |
| `age` | int | 17 → 45 ans | Normal μ=22, σ=5 (tronquée) | Explicative |
| `genre` | str | `M` / `F` | Bernoulli p=0.52 (M) | Explicative catégorielle |
| `niveau_etudes` | str | `L1` / `L2` / `L3` / `M1` / `M2` | Multinomial [0.3, 0.3, 0.2, 0.15, 0.05] | Explicative catégorielle |

### 2.2 Variables Comportementales Numériques (6 variables)

| Variable | Type | Plage | Distribution | Rôle | Description |
|----------|------|-------|--------------|------|-------------|
| `nb_connexions_semaine` | float | 1 → 21 | Normal μ=8, σ=3 | Explicative | Nombre moyen de connexions/semaine |
| `temps_etude_hebdo` | float | 0 → 35h | Normal μ=12, σ=5 | **Explicative principale** | Heures d'étude/semaine |
| `exercices_completes_pct` | float | 0 → 100% | Beta(5,2) × 100 | Explicative | % d'exercices réalisés |
| `videos_vues_pct` | float | 0 → 100% | Beta(4,2) × 100 | Explicative | % de vidéos regardées |
| `participation_forums` | int | 0 → 50 | Poisson(λ=5) | Explicative | Nb de messages posté |
| `nb_devoirs_rendus` | int | 0 → 10 | Binomial(10, p=0.75) | Explicative | Devoirs soumis sur 10 |

### 2.3 Variables Socio-économiques (2 variables)

| Variable | Type | Plage / Modalités | Distribution | Rôle |
|----------|------|-------------------|--------------|------|
| `revenu_famille` | str | `Bas` / `Moyen` / `Élevé` | Multinomial [0.3, 0.5, 0.2] | Explicative catégorielle |
| `score_motivation` | float | 1.0 → 10.0 | Normal μ=6.5, σ=1.8 | Explicative |

### 2.4 Variables Contextuelles (2 variables)

| Variable | Type | Plage | Distribution | Rôle |
|----------|------|-------|--------------|------|
| `nombre_absences` | int | 0 → 20 | Poisson(λ=3) | Explicative |
| `acces_internet` | str | `Stable` / `Instable` / `Limité` | Multinomial [0.6, 0.25, 0.15] | Explicative catégorielle |

### 2.5 Variables Cibles (3 variables)

| Variable | Type | Plage / Modalités | Rôle |
|----------|------|-------------------|------|
| `note_mi_parcours` | float | 0.0 → 20.0 | Cible intermédiaire (régression simple) |
| `note_finale` | float | 0.0 → 20.0 | **Cible principale (régression multiple)** |
| `reussite` | int | `0` (Échec) / `1` (Succès) | **Cible binaire (classification)** |

> **Règle de génération des cibles :**
> - `note_finale` = β₀ + β₁·temps_etude + β₂·exercices_pct + β₃·connexions + ε
> - `reussite` = 1 si note_finale ≥ 10 else 0

---

## 3. SCHÉMA DES RELATIONS ENTRE VARIABLES

```
┌────────────────────────────────────────────────────────────────┐
│                   CARTE DES RELATIONS                          │
│                                                                │
│  [Socio-démographique]    [Comportemental]    [Socio-économ.]  │
│   age ──────────────────►                                      │
│   genre ────────────────►  nb_connexions ──►                   │
│   niveau_etudes ────────►  temps_etude ─────►  note_finale ◄── │
│                            exercices_pct ───►  ↓               │
│                            videos_vues ─────►  reussite        │
│   score_motivation ─────►  participation ──►                   │
│   revenu_famille ───────►  nb_devoirs ─────►                   │
│   acces_internet ───────►                                      │
│   nombre_absences ──────►  note_mi_parcours►                   │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. CODE PYTHON COMPLET DE GÉNÉRATION

```python
# ============================================================
# src/data_generation.py
# INF 232 EC2 — Génération automatique du jeu de données
# Auteur : [Votre nom]
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
    
    Notes
    -----
    Les distributions sont choisies pour être réalistes :
    - Les variables comportementales suivent des distributions
      appropriées à leur nature (Beta pour les pourcentages,
      Poisson pour les comptages, Normale pour les continus).
    - La note finale est générée par une combinaison linéaire
      des variables comportementales avec bruit gaussien.
    """
    rng = np.random.default_rng(random_state)
    
    # ── 1. Identifiants ──────────────────────────────────────
    etudiant_id = np.arange(1, n_samples + 1)
    
    # ── 2. Variables socio-démographiques ────────────────────
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
    
    # ── 3. Variables comportementales ────────────────────────
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
    
    # ── 4. Variables socio-économiques ───────────────────────
    revenu_famille = rng.choice(
        ['Bas', 'Moyen', 'Élevé'],
        size=n_samples,
        p=[0.30, 0.50, 0.20]
    )
    
    score_motivation = np.clip(
        rng.normal(loc=6.5, scale=1.8, size=n_samples),
        1.0, 10.0
    ).round(1)
    
    # ── 5. Variables contextuelles ───────────────────────────
    nombre_absences = rng.poisson(lam=3, size=n_samples)
    
    acces_internet = rng.choice(
        ['Stable', 'Instable', 'Limité'],
        size=n_samples,
        p=[0.60, 0.25, 0.15]
    )
    
    # ── 6. Génération des variables cibles ───────────────────
    # Note mi-parcours : influence principale = temps d'étude + exercices
    note_mi_parcours = (
        2.0
        + 0.35 * temps_etude_hebdo
        + 0.08 * exercices_completes_pct
        + 0.10 * score_motivation
        - 0.15 * nombre_absences
        + rng.normal(0, 1.5, n_samples)
    )
    note_mi_parcours = np.clip(note_mi_parcours, 0.0, 20.0).round(2)
    
    # Note finale : combinaison linéaire réaliste
    # y = β0 + β1*temps + β2*exercices + β3*connexions + β4*devoirs
    #       + β5*motivation - β6*absences + β7*videos + ε
    note_finale = (
        1.5
        + 0.40 * temps_etude_hebdo
        + 0.07 * exercices_completes_pct
        + 0.12 * nb_connexions_semaine
        + 0.30 * nb_devoirs_rendus
        + 0.18 * score_motivation
        - 0.20 * nombre_absences
        + 0.04 * videos_vues_pct
        + 0.60 * note_mi_parcours
        + rng.normal(0, 1.2, n_samples)
    )
    note_finale = np.clip(note_finale, 0.0, 20.0).round(2)
    
    # Variable cible binaire : réussite
    reussite = (note_finale >= 10.0).astype(int)
    
    # ── 7. Construction du DataFrame ─────────────────────────
    df = pd.DataFrame({
        'etudiant_id':             etudiant_id,
        'age':                     age,
        'genre':                   genre,
        'niveau_etudes':           niveau_etudes,
        'nb_connexions_semaine':   nb_connexions_semaine,
        'temps_etude_hebdo':       temps_etude_hebdo,
        'exercices_completes_pct': exercices_completes_pct,
        'videos_vues_pct':         videos_vues_pct,
        'participation_forums':    participation_forums,
        'nb_devoirs_rendus':       nb_devoirs_rendus,
        'revenu_famille':          revenu_famille,
        'score_motivation':        score_motivation,
        'nombre_absences':         nombre_absences,
        'acces_internet':          acces_internet,
        'note_mi_parcours':        note_mi_parcours,
        'note_finale':             note_finale,
        'reussite':                reussite,
    })
    
    return df


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
    print(f"<img src=app/static/icons/check.svg alt=check width=18/> Dataset sauvegardé : {output_path}")
    print(f"   Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"   Taux de réussite : {df['reussite'].mean():.1%}")


if __name__ == "__main__":
    df = generate_student_dataset(n_samples=500, random_state=42)
    save_dataset(df)
    print("\n<img src=app/static/icons/chart.svg alt=chart width=18/> Aperçu des 5 premières observations :")
    print(df.head().to_string())
    print("\n<img src=app/static/icons/up.svg alt=up width=18/> Statistiques descriptives :")
    print(df.describe().round(2).to_string())
```

---

## 5. STATISTIQUES THÉORIQUES ATTENDUES

| Variable | Moyenne (μ) | Écart-type (σ) | Min | Max | Type |
|----------|-------------|----------------|-----|-----|------|
| `age` | 22 | 5 | 17 | 45 | int |
| `nb_connexions_semaine` | 8.0 | 3.0 | 1.0 | 21.0 | float |
| `temps_etude_hebdo` | 12.0 | 5.0 | 0.0 | 35.0 | float |
| `exercices_completes_pct` | ~71% | ~18% | 0% | 100% | float |
| `videos_vues_pct` | ~67% | ~20% | 0% | 100% | float |
| `participation_forums` | 5 | ~2.2 | 0 | ~18 | int |
| `nb_devoirs_rendus` | 7.5 | ~1.37 | 0 | 10 | int |
| `score_motivation` | 6.5 | 1.8 | 1.0 | 10.0 | float |
| `nombre_absences` | 3 | ~1.7 | 0 | ~12 | int |
| `note_mi_parcours` | ~10.5 | ~2.8 | 0.0 | 20.0 | float |
| `note_finale` | ~11.0 | ~3.0 | 0.0 | 20.0 | float |
| `reussite` (taux) | ~60% | - | 0 | 1 | int |

---

## 6. VARIABLES CATÉGORIELLES — DISTRIBUTIONS ATTENDUES

### `genre`
```
M : ~52%  ████████████████████████████████████████████████████
F : ~48%  ████████████████████████████████████████████████
```

### `niveau_etudes`
```
L1 : ~30%  ████████████████████████████████
L2 : ~30%  ████████████████████████████████
L3 : ~20%  ████████████████████████
M1 : ~15%  ████████████████████
M2 :  ~5%  ████████
```

### `revenu_famille`
```
Bas    : ~30%  ████████████████████████████████
Moyen  : ~50%  ████████████████████████████████████████████████████
Élevé  : ~20%  ████████████████████████
```

### `acces_internet`
```
Stable   : ~60%  ████████████████████████████████████████████████████████████████
Instable : ~25%  ████████████████████████████
Limité   : ~15%  ████████████████████
```

---

## 7. CORRÉLATIONS THÉORIQUES ATTENDUES (avec note_finale)

| Variable | Corrélation Pearson (r) théorique | Interprétation |
|----------|----------------------------------|----------------|
| `temps_etude_hebdo` | **+0.65 à +0.75** | Forte corrélation positive |
| `nb_devoirs_rendus` | **+0.55 à +0.65** | Corrélation positive significative |
| `note_mi_parcours` | **+0.60 à +0.70** | Bonne corrélation |
| `exercices_completes_pct` | **+0.45 à +0.55** | Corrélation modérée positive |
| `score_motivation` | **+0.35 à +0.45** | Corrélation modérée |
| `nb_connexions_semaine` | **+0.30 à +0.40** | Corrélation faible-modérée |
| `nombre_absences` | **-0.40 à -0.55** | Corrélation négative significative |
| `age` | **-0.05 à +0.10** | Corrélation quasi-nulle |

---

## 8. PROBLÈMES POTENTIELS ET STRATÉGIES DE NETTOYAGE

| Problème | Fréquence simulée | Traitement |
|----------|-------------------|------------|
| Valeurs manquantes (NaN) | ~2-5% aléatoirement injectées | `SimpleImputer(strategy='median')` pour numériques, `mode` pour catégorielles |
| Outliers (scores > 3σ) | ~1% des observations | IQR method + Winsorization |
| Doublons | 0% (dataset propre) | `df.drop_duplicates()` préventif |
| Variables catégorielles non encodées | 4 variables | `OneHotEncoder` / `LabelEncoder` |

---

## 9. INJECTION DE VALEURS MANQUANTES (Pour la démonstration du nettoyage)

```python
def inject_missing_values(
    df: pd.DataFrame,
    missing_rate: float = 0.03,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Injecte des valeurs manquantes de manière aléatoire dans le
    dataset pour simuler un cas réel et démontrer le nettoyage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset original propre.
    missing_rate : float
        Taux de valeurs manquantes à injecter (défaut : 3%)
    random_state : int
        Graine pour la reproductibilité.
    
    Returns
    -------
    pd.DataFrame
        Dataset avec valeurs manquantes injectées.
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
    
    print(f"<img src=app/static/icons/chart.svg alt=chart width=18/> Valeurs manquantes injectées :")
    print(df_dirty[cols_to_corrupt].isnull().sum().to_string())
    return df_dirty
```

---

*Document 2/10 — INF 232 EC2*
