# INF232 EC2 — Projet E-Learning Performance Analysis

## 📚 Projet

Application web de collecte et d'analyse de la performance académique des étudiants sur une plateforme e-learning.

**Problématique:** *"Quels comportements d'apprentissage influencent la réussite académique des étudiants sur une plateforme e-learning ?"*

## 🛠️ Stack Technique

- **Langage:** Python
- **Data:** Pandas, NumPy
- **ML:** Scikit-learn
- **Visualisation:** Matplotlib, Seaborn, Plotly
- **Application:** Streamlit
- **Base de données:** SQLite
- **Tests:** Pytest

## 📁 Structure du Projet

```
INF232_PROJET/
├── src/              # Code source (génération données, preprocessing)
├── data/             # Dataset CSV
├── models/           # Modèles entrainés (.pkl)
├── app/              # Application Streamlit
├── tests/            # Tests unitaires
├── docs/             # Documentation du projet
└── README.md
```

## 🚀 Installation

```bash
# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install pandas numpy matplotlib seaborn plotly scikit-learn streamlit pytest joblib faker

# Générer les données
python src/data_generation.py

# Lancer l'application
streamlit run app/main.py

# Lancer les tests
pytest tests/ -v
```

## <img src=app/static/icons/chart.svg alt=chart width=18/> Phases du Projet

| Phase | Description |
|-------|-------------|
| 0 | Setup initial (git, venv, arborescence) |
| 1 | Génération des données (500 obs, 15 variables) |
| 2 | Nettoyage + EDA |
| 3 | Régression linéaire |
| 4 | Classification |
| 5 | Réduction de dimension |
| 6 | Application Streamlit |
| 7 | Tests unitaires |
| 8 | Rapport technique |
| 9 | Rendu final |

---

*Projet INF232 EC2 — Université*