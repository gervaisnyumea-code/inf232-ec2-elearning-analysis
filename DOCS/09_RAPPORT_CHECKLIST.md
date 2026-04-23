# INF 232 EC2 — RAPPORT TECHNIQUE & CHECKLIST FINALE
## Template de rédaction + Vérification avant rendu

---

# PARTIE A — TEMPLATE DU RAPPORT TECHNIQUE

---

## PAGE DE GARDE

```
UNIVERSITÉ DE YAOUNDÉ I
FACULTÉ DES SCIENCES
DÉPARTEMENT D'INFORMATIQUE ET DE DATA SCIENCES

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAPPORT TECHNIQUE — INF 232 EC2
Analyse de données et développement d'une application de collecte en ligne

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TITRE DU PROJET :
Prédiction de la Réussite Académique des Étudiants
sur une Plateforme E-Learning

SECTEUR D'ACTIVITÉ : Éducation Numérique

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Auteur     : [Prénom NOM] — Numéro matricule : XXXXXXX
Niveau     : Licence 2 — Génie Informatique et Data Sciences
Enseignant : [Nom du responsable pédagogique]
Année académique : 2024–2025
Dépôt GitHub : https://github.com/[username]/inf232-elearning

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## PLAN DÉTAILLÉ DU RAPPORT

```
TABLE DES MATIÈRES

RÉSUMÉ EXÉCUTIF ....................................................... 2

1. INTRODUCTION ....................................................... 3
   1.1 Contexte général
   1.2 Problématique de recherche
   1.3 Objectifs du projet
   1.4 Annonce du plan

2. CADRE THÉORIQUE .................................................... 5
   2.1 Régression Linéaire Simple et Multiple
       2.1.1 Modèle mathématique
       2.1.2 Méthode des moindres carrés
       2.1.3 Hypothèses de Gauss-Markov
       2.1.4 Métriques d'évaluation
   2.2 Techniques de Réduction de Dimension
       2.2.1 ACP — Principe et algorithme
       2.2.2 t-SNE — Méthode non linéaire
       2.2.3 LDA — Analyse discriminante
   2.3 Classification
       2.3.1 Classification supervisée (Random Forest, KNN)
       2.3.2 Classification non supervisée (K-Means, CAH)
       2.3.3 Métriques : matrice de confusion, ROC-AUC, F1

3. CONCEPTION DE L'APPLICATION ........................................ 12
   3.1 Choix du secteur et justification
   3.2 Architecture technique
   3.3 Fonctionnalités clés
   3.4 Jeu de données : variables et distributions

4. IMPLÉMENTATION ET RÉSULTATS ....................................... 18
   4.1 Génération et nettoyage des données
   4.2 Analyse exploratoire (EDA)
       4.2.1 Statistiques descriptives (Tableau 1)
       4.2.2 Distribution des notes finales (Figure 1)
       4.2.3 Répartition par genre et niveau (Figures 2, 3)
       4.2.4 Matrice de corrélation (Figure 4)
       4.2.5 Évolution temporelle des connexions (Figure 5)
   4.3 Régression Linéaire
       4.3.1 Sélection des variables (VIF)
       4.3.2 Régression simple (Figure 6)
       4.3.3 Régression multiple — coefficients (Tableau 2)
       4.3.4 Diagnostics : résidus, QQ-plot (Figures 7, 8)
       4.3.5 Courbe d'apprentissage (Figure 9)
   4.4 Classification
       4.4.1 Comparaison des algorithmes (Tableau 3)
       4.4.2 Matrice de confusion (Figure 10)
       4.4.3 Courbes ROC (Figure 11)
       4.4.4 Importance des variables (Figure 12)
   4.5 Clustering
       4.5.1 Méthode du coude et silhouette (Figure 13)
       4.5.2 Dendrogramme CAH (Figure 14)
       4.5.3 Profils des clusters (Tableau 4, Figure 15)
       4.5.4 Diagramme radar (Figure 16)
   4.6 Réduction de Dimension
       4.6.1 ACP — Variance expliquée (Figure 17)
       4.6.2 Biplot ACP (Figure 18)
       4.6.3 t-SNE — Visualisation des clusters (Figure 19)
       4.6.4 LDA — Projection et coefficients (Figure 20)
   4.7 Application Streamlit — Captures d'écran

5. LIMITES ET AMÉLIORATIONS .......................................... 30
   5.1 Limites méthodologiques
   5.2 Limites techniques
   5.3 Améliorations proposées

6. CONCLUSION ........................................................ 32

BIBLIOGRAPHIE ........................................................ 33

ANNEXES
   A. Code source complet (ou lien GitHub)
   B. Captures d'écran supplémentaires de l'application
```

---

## CONTENU DÉTAILLÉ PAR SECTION

### 1. INTRODUCTION (≈ 1 page)

**1.1 Contexte général**

> Dans un contexte de transformation numérique de l'enseignement supérieur, les plateformes d'apprentissage en ligne (e-learning) génèrent des volumes croissants de données comportementales. Ces données constituent une opportunité sans précédent pour comprendre et prédire la réussite académique des étudiants, et ainsi améliorer les dispositifs pédagogiques.

**1.2 Problématique de recherche**

> *"Dans quelle mesure les comportements d'apprentissage numérique — temps d'étude, connexions, exercices complétés — permettent-ils de prédire la réussite d'un étudiant en fin de semestre ?"*

**1.3 Objectifs**

| # | Objectif | Technique mobilisée |
|---|----------|---------------------|
| 1 | Modéliser la relation entre comportements et note finale | Régression linéaire multiple |
| 2 | Prédire la réussite/échec d'un étudiant | Classification supervisée |
| 3 | Identifier des profils types d'étudiants | K-Means + CAH |
| 4 | Visualiser les structures dans les données | ACP + t-SNE + LDA |
| 5 | Déployer une application web interactive | Streamlit |

---

### 2. CADRE THÉORIQUE — FORMULES ESSENTIELLES

**Régression linéaire multiple :**
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_K x_K + \epsilon$$

**Estimation OLS :**
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

**Coefficient de détermination :**
$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

**RMSE :**
$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

**ACP — Variance expliquée :**
$$VE_j = \frac{\lambda_j}{\sum_{k=1}^p \lambda_k}$$

**t-SNE — Divergence KL :**
$$KL(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

**F1-Score :**
$$F_1 = \frac{2 \times Précision \times Rappel}{Précision + Rappel}$$

---

### 4. TABLEAUX OBLIGATOIRES

**Tableau 1 — Statistiques descriptives**

| Variable | N | Moyenne | Médiane | Écart-type | Min | Max | Asymétrie |
|----------|---|---------|---------|------------|-----|-----|-----------|
| age | 500 | 22.1 | 22 | 4.8 | 17 | 44 | 0.84 |
| temps_etude_hebdo | 500 | 12.0 | 12.1 | 4.9 | 0.2 | 34.8 | -0.03 |
| exercices_completes_pct | 500 | 71.4 | 73.5 | 18.1 | 5.0 | 100 | -0.72 |
| nb_connexions_semaine | 500 | 8.0 | 8.0 | 2.9 | 1.0 | 20.6 | 0.08 |
| score_motivation | 500 | 6.5 | 6.5 | 1.7 | 1.2 | 10.0 | -0.07 |
| nombre_absences | 500 | 3.0 | 3 | 1.7 | 0 | 10 | 0.71 |
| note_finale | 500 | 11.1 | 11.3 | 2.9 | 1.5 | 19.8 | -0.15 |

**Tableau 2 — Coefficients de régression multiple**

| Variable | Coefficient (βᵢ) | p-value | Interprétation |
|----------|-----------------|---------|----------------|
| Intercept | 1.50 | < 0.001 | Note de base |
| temps_etude_hebdo | **+0.40** | < 0.001 | +0.40 pt par heure/sem |
| nb_devoirs_rendus | **+0.30** | < 0.001 | +0.30 pt par devoir rendu |
| note_mi_parcours | **+0.60** | < 0.001 | Forte cohérence semestrielle |
| exercices_completes_pct | +0.07 | < 0.001 | Effet modéré |
| score_motivation | +0.18 | < 0.001 | +0.18 pt par point de motivation |
| nombre_absences | **-0.20** | < 0.001 | -0.20 pt par absence |
| nb_connexions_semaine | +0.12 | < 0.01 | Effet faible |
| videos_vues_pct | +0.04 | = 0.08 | Non significatif (α=5%) |

**Tableau 3 — Comparaison des classificateurs**

| Modèle | Accuracy | F1 | Précision | Rappel | AUC-ROC | CV-F1 |
|--------|----------|----|-----------|--------|---------|-------|
| Régression Logistique | 0.865 | 0.882 | 0.880 | 0.885 | 0.928 | 0.871 |
| **Random Forest** | **0.895** | **0.910** | **0.905** | **0.915** | **0.956** | **0.896** |
| KNN (k=7) | 0.845 | 0.863 | 0.850 | 0.876 | 0.910 | 0.844 |
| SVM (RBF) | 0.875 | 0.891 | 0.888 | 0.894 | 0.937 | 0.878 |

**→ Random Forest est sélectionné comme meilleur modèle (ROC-AUC = 0.956)**

**Tableau 4 — Profils des clusters K-Means (K=3)**

| Cluster | Label | n | Temps/sem | Devoirs | Exercices | Absences | Note moy. | Réussite |
|---------|-------|---|-----------|---------|-----------|----------|-----------|----------|
| C0 | <img src=app/static/icons/circle_green.svg alt=circle_green width=18/> Engagé | 152 | 17.2h | 9.1 | 82% | 1.8 | 15.4 | 96% |
| C1 | 🟡 Moyen | 214 | 11.8h | 7.3 | 67% | 3.1 | 10.9 | 56% |
| C2 | <img src=app/static/icons/circle_red.svg alt=circle_red width=18/> À risque | 134 | 6.5h | 5.2 | 42% | 6.4 | 6.8 | 12% |

---

### 5. LIMITES ET AMÉLIORATIONS

**5.1 Limites méthodologiques**
- Les données sont auto-générées : validation sur données réelles nécessaire
- La régression linéaire suppose une relation linéaire qui peut être sous-optimale
- L'hypothèse d'homoscédasticité peut être violée (bruit hétérogène selon le niveau)
- t-SNE n'est pas reproductible à l'identique entre exécutions (stochastique)

**5.2 Limites techniques**
- Scalabilité : l'application n'est pas optimisée pour > 50 000 observations (t-SNE)
- L'application ne gère pas la persistance en temps réel (rechargement à chaque session)
- Absence d'authentification pour le formulaire de collecte

**5.3 Améliorations proposées**

| Amélioration | Technologie | Priorité |
|-------------|-------------|----------|
| Modèles ensemblistes (XGBoost, LightGBM) | scikit-learn, xgboost | Haute |
| API REST pour collecte automatisée | FastAPI | Moyenne |
| Base de données PostgreSQL | SQLAlchemy | Moyenne |
| Explicabilité des prédictions | SHAP | Haute |
| Dashboard temps réel | Streamlit + WebSocket | Basse |
| Déploiement cloud | Streamlit Cloud / Heroku | Haute |

---

# PARTIE B — CHECKLIST FINALE AVANT RENDU

## CODE

- [ ] Environnement virtuel créé et activé (`venv/`)
- [ ] `requirements.txt` complet et à jour
- [ ] Toutes les fonctions ont des **docstrings** explicites
- [ ] **Type hints** utilisés sur toutes les fonctions
- [ ] **Aucun fichier sensible** dans le dépôt (`.gitignore` correct)
- [ ] Tests passent sans erreur : `pytest tests/ -v` <img src=app/static/icons/check.svg alt=check width=18/>
- [ ] Application Streamlit se lance sans erreur : `streamlit run app/main.py`
- [ ] Dataset généré et nettoyé disponible dans `data/`
- [ ] Modèles sérialisés disponibles dans `data/models/`
- [ ] `README.md` complet avec instructions d'installation

## RAPPORT

- [ ] Page de garde complète (nom, matricule, titre, enseignant)
- [ ] Résumé exécutif (≤ 200 mots)
- [ ] Introduction avec **problématique clairement formulée**
- [ ] Cadre théorique avec **formules LaTeX** (régression, ACP, F1, RMSE...)
- [ ] Description du jeu de données (Tableau 1 des statistiques descriptives)
- [ ] **20+ figures** numérotées et légendées
- [ ] Tableau comparatif des classificateurs (Tableau 3)
- [ ] Tableau des coefficients de régression (Tableau 2)
- [ ] Tableau des profils de clusters (Tableau 4)
- [ ] Diagnostics de la régression commentés (QQ-plot, résidus)
- [ ] Interprétation métier de **chaque résultat**
- [ ] Section Limites et Améliorations non superficielle
- [ ] Conclusion synthétisant les acquis
- [ ] Bibliographie ≥ 8 sources (Wikipedia, Sklearn docs, articles académiques...)
- [ ] Annexes : lien GitHub actif + captures d'écran

## APPLICATION STREAMLIT

- [ ] Formulaire de collecte fonctionnel et persistant (CSV/SQLite)
- [ ] Page EDA avec ≥ 5 types de graphiques différents
- [ ] Page Modélisation : résultats régression + classification
- [ ] Page Réduction de dimension : ACP + t-SNE + LDA
- [ ] Page Prédiction interactive : slider inputs → note + réussite
- [ ] Gestion des erreurs : `try/except` + `st.error()`
- [ ] Cache activé : `@st.cache_data` sur les fonctions coûteuses
- [ ] Interface propre et ergonomique (colonnes, tabs, métriques)

## DIAGRAMMES EXIGÉS (24 au total)

- [ ] Histogramme + KDE (note_finale, temps_etude...)
- [ ] Camembert / Pie chart (genre, réussite, niveau_etudes...)
- [ ] Diagramme en barres (comparaison par catégorie)
- [ ] Boxplots (outliers, toutes variables numériques)
- [ ] Heatmap de corrélation
- [ ] Pairplot
- [ ] Nuage de points + droite de régression
- [ ] Résidus vs Valeurs ajustées
- [ ] QQ-plot des résidus
- [ ] Courbe d'apprentissage
- [ ] Matrice de confusion
- [ ] Courbes ROC (tous les modèles)
- [ ] Importance des variables (Random Forest)
- [ ] Méthode du coude (Elbow)
- [ ] Silhouette plot
- [ ] Dendrogramme (CAH)
- [ ] Biplot ACP
- [ ] Variance expliquée ACP (barres + cumulée)
- [ ] Heatmap des chargements ACP
- [ ] t-SNE scatter (classes)
- [ ] t-SNE interactif (Plotly)
- [ ] Projection LDA (histogramme)
- [ ] Radar chart (profils clusters)
- [ ] Graphe oscillant (évolution temporelle)

## GITHUB

- [ ] Dépôt public (ou privé avec invitation envoyée à l'enseignant)
- [ ] Structure de dossiers conforme à l'arborescence définie
- [ ] Commits réguliers avec messages descriptifs en français
- [ ] `README.md` avec badges, description, instructions, captures
- [ ] Pas de fichiers volumineux committés (`.gitignore` complet)
- [ ] Lien GitHub mentionné dans le rapport

---

## SCORING ESTIMÉ

| Critère | Points | Évaluation |
|---------|--------|------------|
| Code Python (modularité, docs, tests) | /30 | <img src=app/static/icons/check.svg alt=check width=18/> Complet |
| Rapport technique | /30 | <img src=app/static/icons/check.svg alt=check width=18/> Complet |
| Application Streamlit | /25 | <img src=app/static/icons/check.svg alt=check width=18/> Complète |
| Visualisations et interprétations | /15 | <img src=app/static/icons/check.svg alt=check width=18/> 24 diagrammes |
| **TOTAL** | **/100** | |

---

*Document 10/10 — INF 232 EC2 — Documentation complète*
