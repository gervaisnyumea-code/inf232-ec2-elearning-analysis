# INF 232 EC2 — ARCHITECTURE TECHNIQUE COMPLÈTE
## Structure du Projet, Modules et Dépendances

---

## 1. ARBORESCENCE CHIRURGICALE DU PROJET

```
inf232_elearning/
│
├── 📁 data/
│   ├── raw/
│   │   └── elearning_dataset.csv          ← Dataset brut généré
│   ├── processed/
│   │   └── elearning_clean.csv            ← Dataset nettoyé
│   └── models/
│       ├── regression_model.pkl           ← Modèle régression sérialisé
│       ├── classifier_model.pkl           ← Modèle classification sérialisé
│       └── scaler.pkl                     ← StandardScaler sérialisé
│
├── 📁 src/
│   ├── __init__.py
│   ├── data_generation.py                 ← Génération des données
│   ├── data_cleaning.py                   ← Nettoyage + feature engineering
│   ├── analysis.py                        ← Statistiques descriptives
│   ├── models.py                          ← Classes régression + classification
│   ├── clustering.py                      ← K-Means + CAH
│   ├── dimension_reduction.py             ← ACP + t-SNE + LDA
│   ├── visualization.py                   ← Toutes les fonctions graphiques
│   └── utils.py                           ← Fonctions utilitaires
│
├── 📁 app/
│   ├── __init__.py
│   ├── main.py                            ← Point d'entrée Streamlit
│   └── pages/
│       ├── 1_Collecte.py                  ← Page formulaire de collecte
│       ├── 2_Exploration.py               ← Page EDA + statistiques
│       ├── 3_Modelisation.py              ← Page modèles prédictifs
│       └── 4_Visualisation.py             ← Page réduction de dimension
│
├── 📁 notebooks/
│   ├── 01_exploration.ipynb               ← EDA interactive
│   ├── 02_modelisation.ipynb              ← Modélisation pas à pas
│   └── 03_visualisation.ipynb             ← Toutes les visualisations
│
├── 📁 tests/
│   ├── __init__.py
│   ├── test_data_cleaning.py
│   ├── test_models.py
│   └── test_utils.py
│
├── 📁 rapport/
│   └── rapport_inf232.pdf
│
├── requirements.txt
├── README.md
├── .gitignore
└── setup.py
```

---

## 2. DIAGRAMME D'ARCHITECTURE (Vue en Couches)

```
┌─────────────────────────────────────────────────────────────────┐
│                     COUCHE PRÉSENTATION                         │
│                   app/  (Streamlit)                             │
│   ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────────┐  │
│   │ Collecte │ │   EDA    │ │ Modélisation │ │  Réduction   │  │
│   │  forms   │ │ dashbrd  │ │  prédiction  │ │  ACP/t-SNE   │  │
│   └────┬─────┘ └────┬─────┘ └──────┬───────┘ └──────┬───────┘  │
└────────┼────────────┼──────────────┼────────────────┼──────────┘
         │            │              │                │
┌────────▼────────────▼──────────────▼────────────────▼──────────┐
│                    COUCHE LOGIQUE MÉTIER                         │
│                        src/                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ analysis │ │  models  │ │clustering│ │dim_reduction     │   │
│  │          │ │          │ │          │ │ PCA, t-SNE, LDA  │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────────┘   │
│       │            │            │              │                │
│  ┌────▼────────────▼────────────▼──────────────▼───────────┐   │
│  │               data_cleaning  +  visualization           │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
         │
┌────────▼───────────────────────────────────────────────────────┐
│                    COUCHE DONNÉES                               │
│  data/raw/         data/processed/       data/models/           │
│  elearning_        elearning_            *.pkl                   │
│  dataset.csv       clean.csv                                    │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. MODULES — SPÉCIFICATIONS ET INTERFACES

### 3.1 `src/data_cleaning.py`

```python
# ============================================================
# src/data_cleaning.py
# ============================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from pathlib import Path
import joblib
from typing import Tuple


def load_raw_data(filepath: str = "data/raw/elearning_dataset.csv") -> pd.DataFrame:
    """
    Charge les données brutes depuis un fichier CSV.
    
    Parameters
    ----------
    filepath : str
        Chemin vers le fichier CSV.
    
    Returns
    -------
    pd.DataFrame
        DataFrame des données brutes.
    
    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas au chemin spécifié.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")
    df = pd.read_csv(path, encoding='utf-8')
    print(f"<img src=app/static/icons/check.svg alt=check width=18/> Données chargées : {df.shape}")
    return df


def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère un rapport des valeurs manquantes.
    
    Returns
    -------
    pd.DataFrame
        Rapport avec colonnes [colonne, nb_manquant, pct_manquant].
    """
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({
        'nb_manquant': missing,
        'pct_manquant': pct
    }).query('nb_manquant > 0').sort_values('pct_manquant', ascending=False)
    return report


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Traite les valeurs manquantes :
    - Variables numériques → imputation par la médiane
    - Variables catégorielles → imputation par le mode
    """
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
        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    return df_clean


def remove_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
    """
    Supprime les outliers selon la méthode IQR.
    
    Parameters
    ----------
    col : str
        Colonne sur laquelle appliquer le filtrage.
    factor : float
        Multiplicateur de l'IQR (défaut : 1.5).
    
    Returns
    -------
    pd.DataFrame
        DataFrame sans les outliers.
    """
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
    """
    Encode les variables catégorielles :
    - genre → LabelEncoder (M=1, F=0)
    - niveau_etudes → ordinal mapping
    - revenu_famille → ordinal mapping
    - acces_internet → ordinal mapping
    """
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
    """
    Retourne la matrice de features X et les vecteurs cibles y_reg et y_clf.
    
    Returns
    -------
    X : pd.DataFrame
        Matrice des variables explicatives (numériques encodées)
    y_reg : pd.Series
        Variable cible pour la régression (note_finale)
    y_clf : pd.Series
        Variable cible pour la classification (reussite)
    """
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
    """
    Exécute le pipeline complet de nettoyage.
    """
    print("🔧 Démarrage du pipeline de nettoyage...")
    df = load_raw_data(filepath)
    
    print("\n<img src=app/static/icons/chart.svg alt=chart width=18/> Rapport valeurs manquantes :")
    print(report_missing_values(df).to_string())
    
    df = handle_missing_values(df)
    df = encode_categorical_variables(df)
    
    # Supprimer les doublons
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"\n<img src=app/static/icons/search.svg alt=search width=18/> Doublons supprimés : {n_before - len(df)}")
    
    # Sauvegarder
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/elearning_clean.csv", index=False)
    print(f"\n<img src=app/static/icons/check.svg alt=check width=18/> Dataset nettoyé sauvegardé ({df.shape})")
    
    return df
```

---

### 3.2 `src/utils.py`

```python
# ============================================================
# src/utils.py
# ============================================================
import pathlib
import joblib
import pandas as pd
from typing import Any


def get_project_root() -> pathlib.Path:
    """Retourne le chemin racine du projet."""
    return pathlib.Path(__file__).parent.parent


def save_model(model: Any, filepath: str) -> None:
    """
    Sérialise un modèle sklearn avec joblib.
    
    Parameters
    ----------
    model : Any
        Objet modèle sklearn.
    filepath : str
        Chemin de sauvegarde (.pkl).
    """
    pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"<img src=app/static/icons/check.svg alt=check width=18/> Modèle sauvegardé : {filepath}")


def load_model(filepath: str) -> Any:
    """
    Charge un modèle sérialisé avec joblib.
    
    Raises
    ------
    FileNotFoundError
        Si le fichier .pkl n'existe pas.
    """
    path = pathlib.Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {filepath}")
    return joblib.load(filepath)


def compute_vif(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Calcule le Variance Inflation Factor pour détecter
    la multicolinéarité.
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec colonnes [feature, VIF].
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = df[features].dropna()
    vif_data = pd.DataFrame()
    vif_data['feature'] = features
    vif_data['VIF'] = [
        variance_inflation_factor(X.values, i)
        for i in range(len(features))
    ]
    return vif_data.sort_values('VIF', ascending=False)
```

---

## 4. REQUIREMENTS.TXT

```
# INF 232 EC2 — Dépendances Python
# Installation : pip install -r requirements.txt

# ── Manipulation de données ───────────────────────────────
pandas==1.5.3
numpy==1.23.5

# ── Visualisation ─────────────────────────────────────────
matplotlib==3.7.0
seaborn==0.12.2
plotly==5.13.0

# ── Machine Learning ──────────────────────────────────────
scikit-learn==1.2.2

# ── Application web ───────────────────────────────────────
streamlit==1.20.0

# ── Sauvegarde modèles ────────────────────────────────────
joblib==1.2.0

# ── Tests unitaires ───────────────────────────────────────
pytest==7.2.1

# ── Statistiques avancées ─────────────────────────────────
scipy==1.10.0
statsmodels==0.13.5

# ── Génération de données ─────────────────────────────────
faker==18.3.0
```

---

## 5. .GITIGNORE

```gitignore
# Environnement Python
venv/
__pycache__/
*.pyc
*.pyo
.env

# Données volumineuses
data/raw/*.csv
data/processed/*.parquet

# Modèles sérialisés lourds
data/models/*.pkl

# IDE
.vscode/
.idea/
*.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
```

---

## 6. README.MD

```markdown
# INF 232 EC2 — Analyse de Performance Académique en E-Learning

Application web de collecte et d'analyse des comportements d'apprentissage des étudiants.

## Installation

```bash
git clone <url-repo>
cd inf232_elearning
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OU : venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

## Génération des données

```bash
python src/data_generation.py
```

## Lancement de l'application

```bash
streamlit run app/main.py
```

## Tests

```bash
pytest tests/ -v --tb=short
```

## Auteur

[Votre nom] — Licence 2 Génie Informatique et Data Sciences  
Université de Yaoundé I — INF 232 EC2
```

---

*Document 3/10 — INF 232 EC2*
