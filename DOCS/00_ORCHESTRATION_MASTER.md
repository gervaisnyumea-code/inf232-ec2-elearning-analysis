# INF 232 EC2 — ORCHESTRATION MASTER
## Plan Général & Feuille de Route Chirurgicale

---

## 🎯 VISION DU PROJET

**Titre :** Application web de collecte et d'analyse de la performance académique des étudiants en ligne  
**Secteur choisi :** Éducation numérique & E-learning  
**Problématique :** *"Quels comportements d'apprentissage influencent la réussite académique des étudiants sur une plateforme e-learning ?"*  
**Stack :** Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Plotly · Streamlit · SQLite · Pytest

---

## 📐 ARCHITECTURE DES LIVRABLES

```
INF232_PROJET/
├── 00_ORCHESTRATION_MASTER.md        ← CE DOCUMENT
├── 01_DATASET_AUTOGENERE.md          ← Registre + code de génération de données
├── 02_ARCHITECTURE_TECHNIQUE.md      ← Arborescence + modules + dépendances
├── 03_EDA_DIAGRAMMES.md              ← Analyse exploratoire + catalogue complet de diagrammes
├── 04_REGRESSION_LINEAIRE.md         ← Régression simple/multiple + diagnostics + diagrammes
├── 05_CLASSIFICATION.md              ← Supervisée + K-Means + CAH + diagrammes
├── 06_REDUCTION_DIMENSION.md         ← ACP + t-SNE + LDA + diagrammes
├── 07_APPLICATION_STREAMLIT.md       ← Code complet commenté de l'app
├── 08_RAPPORT_TECHNIQUE.md           ← Template de rédaction du rapport final PDF
└── 09_CHECKLIST_FINALE.md            ← Checklist de rendu avant soumission
```

---

## 🗺️ CARTE MENTALE DU PROJET (Vue d'ensemble)

```
                    ┌─────────────────────────────────────────────────────┐
                    │               INF 232 EC2 — PROJET GLOBAL           │
                    └──────────────────────┬──────────────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         │                                 │                                 │
         ▼                                 ▼                                 ▼
  ┌─────────────┐                 ┌────────────────┐                ┌───────────────┐
  │  DONNÉES    │                 │  MODÉLISATION  │                │  APPLICATION  │
  └──────┬──────┘                 └───────┬────────┘                └───────┬───────┘
         │                                │                                 │
    Auto-génération              ┌────────┴────────┐                 Streamlit
    500 observations             │                 │                 Multi-pages
    15 variables                 ▼                 ▼
                          Supervisé          Non-supervisé
                       ┌────────────┐      ┌──────────────┐
                       │ Régression │      │   K-Means    │
                       │ Linéaire   │      │   CAH        │
                       │ Classif.   │      │              │
                       └────────────┘      └──────────────┘
                              │
                    ┌─────────┴────────┐
                    ▼                  ▼
               Réduction          Diagrammes
               ACP / t-SNE / LDA  (12 types)
```

---

## 📅 PLANNING DÉTAILLÉ PAR PHASE

| Phase | Nom | Durée | Documents associés | Livrables |
|-------|-----|-------|--------------------|-----------|
| **0** | Choix du secteur + setup | Jour 1 | 00, 02 | Git init, venv, arborescence |
| **1** | Génération des données | Jour 1-2 | 01 | dataset.csv (500 lignes) |
| **2** | Nettoyage + EDA | Jour 2-3 | 03 | notebook_exploration.ipynb |
| **3** | Régression linéaire | Jour 3-4 | 04 | models/regression.pkl |
| **4** | Classification | Jour 4-5 | 05 | models/classifier.pkl |
| **5** | Réduction de dimension | Jour 5-6 | 06 | visualisations ACP/t-SNE/LDA |
| **6** | Application Streamlit | Jour 6-8 | 07 | app/ fonctionnelle |
| **7** | Tests unitaires | Jour 8 | 02, 09 | pytest passing |
| **8** | Rapport technique | Jour 8-10 | 08 | rapport_inf232.pdf |
| **9** | Rendu final | Jour 10 | 09 | GitHub propre + PDF |

---

## 🔗 FLUX DE DONNÉES (Pipeline Complet)

```
┌──────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE DONNÉES                           │
│                                                                  │
│  [Génération]  →  [Nettoyage]  →  [EDA]  →  [Modélisation]      │
│       │               │            │              │              │
│  numpy.random    dropna()      describe()    LinearReg           │
│  faker           IQR            heatmap      KNN/RF              │
│  500 obs.        StandardScaler pairplot     KMeans              │
│  15 vars.        OneHotEncoder  boxplot      PCA/TSNE/LDA        │
│       │               │            │              │              │
│       └───────────────┴────────────┴──────────────┘              │
│                          │                                        │
│                    [Application Streamlit]                        │
│                    - Formulaire collecte                          │
│                    - Tableau de bord EDA                          │
│                    - Prédiction interactive                       │
│                    - Visualisations réduction                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📊 CATALOGUE COMPLET DES DIAGRAMMES ATTENDUS

| # | Type de Diagramme | Technique | Bibliothèque | Section du rapport |
|---|-------------------|-----------|--------------|-------------------|
| 1 | Histogramme + KDE | EDA | Seaborn | Analyse descriptive |
| 2 | **Camembert (pie)** | Distribution catégorielles | Matplotlib | EDA |
| 3 | **Diagramme en barres** | Comparaison catégories | Plotly | EDA |
| 4 | Boxplot | Outliers | Seaborn | EDA |
| 5 | Heatmap corrélation | Relations linéaires | Seaborn | EDA |
| 6 | Pairplot | Relations multivariées | Seaborn | EDA |
| 7 | **Nuage de points** (scatter) | Régression simple | Matplotlib | Régression |
| 8 | **Droite de régression** | Régression simple | Matplotlib | Régression |
| 9 | **Résidus vs Fitted** | Diagnostic régression | Matplotlib | Régression |
| 10 | **QQ-plot** | Normalité des résidus | Scipy | Régression |
| 11 | **Courbe d'apprentissage** | Performance modèle | Scikit-learn | Modélisation |
| 12 | **Matrice de confusion** | Classification | Seaborn | Classification |
| 13 | **Courbe ROC** | Classification binaire | Scikit-learn | Classification |
| 14 | **Importance des variables** (barres) | Random Forest | Matplotlib | Classification |
| 15 | **Méthode du coude** | K-Means | Matplotlib | Clustering |
| 16 | **Silhouette plot** | Qualité clusters | Scikit-learn | Clustering |
| 17 | **Dendrogramme** | CAH | Scipy | Clustering |
| 18 | **Biplot ACP** | Réduction dim. | Matplotlib | ACP |
| 19 | **Variance expliquée** (barres) | ACP | Matplotlib | ACP |
| 20 | **t-SNE scatter** coloré | Clustering visuel | Matplotlib | t-SNE |
| 21 | **LDA projection** | Réduction supervisée | Matplotlib | LDA |
| 22 | **Radar/Toile d'araignée** | Profil des clusters | Plotly | Clustering |
| 23 | **Graphe en chaîne** (pipeline) | Architecture | Diagramme | Architecture |
| 24 | **Graphe oscillant** (série temporelle) | Évolution connexions | Plotly | EDA |

> **Total : 24 types de diagrammes** couvrant 100% des exigences institutionnelles

---

## 🔑 CRITÈRES DE NOTATION (Mapping)

| Critère | Poids | Documents concernés |
|---------|-------|---------------------|
| Qualité du code (modularité, docstrings, tests) | 30% | 02, 07, 09 |
| Rapport technique (structure, rigueur, équations) | 30% | 08 |
| Application fonctionnelle (formulaire, modèles) | 25% | 07 |
| Visualisations et interprétation | 15% | 03, 04, 05, 06 |

---

## ⚡ COMMANDES DE DÉMARRAGE RAPIDE

```bash
# 1. Créer l'environnement
python -m venv venv && source venv/bin/activate  # Linux/Mac
# OU : venv\Scripts\activate  # Windows

# 2. Installer les dépendances
pip install pandas numpy matplotlib seaborn plotly scikit-learn streamlit pytest joblib faker

# 3. Générer les données
python src/data_generation.py

# 4. Lancer l'application
streamlit run app/main.py

# 5. Lancer les tests
pytest tests/ -v
```

---

## 📌 JUSTIFICATION DU SECTEUR CHOISI

Le secteur de **l'éducation numérique** est retenu pour les raisons suivantes :

1. **Richesse des variables** : comportementales (connexions, temps d'étude), académiques (notes), socio-démographiques (âge, genre, niveau), permettant d'appliquer *toutes* les techniques requises.
2. **Pertinence pédagogique** : en tant qu'étudiant en Licence 2, l'analyse de la réussite académique présente un intérêt direct et une légitimité forte.
3. **Potentiel de visualisation** : les données de progression, de connexion et de performance génèrent naturellement des séries temporelles, des clusters de profils et des modèles prédictifs.
4. **Originalité** : évite les thèmes classiques (Titanic, Iris) tout en restant accessible.
5. **Faisabilité** : données auto-générées avec `numpy.random` et `faker` de manière réaliste.

---

*Document 1/10 — INF 232 EC2 — Daryl Gervais MASTER-KING*
