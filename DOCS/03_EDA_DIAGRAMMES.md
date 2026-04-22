# INF 232 EC2 — ANALYSE EXPLORATOIRE & CATALOGUE COMPLET DES DIAGRAMMES
## EDA — Statistiques Descriptives + 12 Types de Visualisations

---

## 1. MODULE `src/analysis.py` — STATISTIQUES DESCRIPTIVES

```python
# ============================================================
# src/analysis.py
# ============================================================
import pandas as pd
import numpy as np
from typing import Dict, Any


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
            results.append({'colonne': col, 'nb_outliers': n_outliers,
                           'pct': round(n_outliers/len(df)*100, 2)})
    return pd.DataFrame(results)
```

---

## 2. MODULE `src/visualization.py` — TOUTES LES FONCTIONS GRAPHIQUES

```python
# ============================================================
# src/visualization.py
# INF 232 EC2 — Catalogue complet des visualisations
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, Tuple

# Palette cohérente sur tout le projet
PALETTE = sns.color_palette("husl", 8)
COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11
})


# ════════════════════════════════════════════════════════════
# DIAGRAMME 1 — HISTOGRAMME + KDE
# ════════════════════════════════════════════════════════════

def plot_histogram_kde(
    df: pd.DataFrame,
    col: str,
    title: Optional[str] = None,
    color: str = '#2196F3'
) -> plt.Figure:
    """
    Diagramme : Distribution d'une variable numérique (histogramme + courbe KDE).
    Objectif  : Visualiser la forme de la distribution (symétrie, asymétrie).
    
    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        df[col].dropna(), kde=True, color=color,
        edgecolor='white', linewidth=0.5, ax=ax
    )
    mean_val = df[col].mean()
    median_val = df[col].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Moyenne : {mean_val:.2f}')
    ax.axvline(median_val, color='orange', linestyle=':', linewidth=1.5, label=f'Médiane : {median_val:.2f}')
    ax.set_title(title or f'Distribution de {col}', fontsize=13, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Fréquence')
    ax.legend()
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════
# DIAGRAMME 2 — CAMEMBERT (PIE CHART)
# ════════════════════════════════════════════════════════════

def plot_pie_chart(
    df: pd.DataFrame,
    col: str,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Diagramme : Répartition d'une variable catégorielle (camembert).
    Objectif  : Visualiser les proportions de chaque modalité.
    
    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    counts = df[col].value_counts()
    explode = [0.04] * len(counts)
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct='%1.1f%%',
        explode=explode,
        colors=COLORS[:len(counts)],
        startangle=140,
        shadow=True
    )
    for text in autotexts:
        text.set_fontweight('bold')
    ax.set_title(title or f'Répartition de {col}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════
# DIAGRAMME 3 — DIAGRAMME EN BARRES (Plotly interactif)
# ════════════════════════════════════════════════════════════

def plot_bar_chart_interactive(
    df: pd.DataFrame,
    col: str,
    target: str = 'note_finale',
    title: Optional[str] = None
) -> go.Figure:
    """
    Diagramme : Moyenne de la variable cible par catégorie (barres interactives Plotly).
    Objectif  : Comparer les performances moyennes entre groupes.
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    agg = df.groupby(col)[target].mean().reset_index().sort_values(target, ascending=False)
    fig = px.bar(
        agg, x=col, y=target,
        text=agg[target].round(2),
        color=col,
        color_discrete_sequence=px.colors.qualitative.Set2,
        title=title or f'Moyenne {target} par {col}',
        labels={col: col, target: f'Moyenne {target}'}
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        yaxis=dict(range=[0, max(agg[target]) * 1.15])
    )
    return fig


# ════════════════════════════════════════════════════════════
# DIAGRAMME 4 — BOXPLOT (Outliers)
# ════════════════════════════════════════════════════════════

def plot_boxplot_multi(
    df: pd.DataFrame,
    cols: list,
    title: str = 'Détection des outliers — Boxplots'
) -> plt.Figure:
    """
    Diagramme : Boîtes à moustaches pour plusieurs variables.
    Objectif  : Détecter les outliers et comparer les dispersions.
    """
    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()
    
    for i, col in enumerate(cols):
        sns.boxplot(y=df[col], ax=axes[i], color=COLORS[i % len(COLORS)],
                    flierprops=dict(marker='o', color='red', markersize=4))
        axes[i].set_title(col, fontweight='bold')
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════
# DIAGRAMME 5 — HEATMAP DE CORRÉLATION
# ════════════════════════════════════════════════════════════

def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = 'Matrice de Corrélation de Pearson'
) -> plt.Figure:
    """
    Diagramme : Heatmap des corrélations entre variables numériques.
    Objectif  : Identifier les relations linéaires entre variables.
    """
    num_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = num_df.corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt='.2f',
        cmap='RdYlGn', center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
        cbar_kws={'label': 'Coefficient de Pearson'}
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════
# DIAGRAMME 6 — SÉRIE TEMPORELLE / GRAPHE OSCILLANT
# ════════════════════════════════════════════════════════════

def plot_oscillating_progression(
    df: pd.DataFrame,
    col: str = 'nb_connexions_semaine',
    n_students: int = 5
) -> go.Figure:
    """
    Diagramme : Évolution simulée des connexions sur 16 semaines (oscillant).
    Objectif  : Visualiser la progression dans le temps d'un étudiant type.
    
    Note : Simule une progression hebdomadaire à partir des données.
    """
    rng = np.random.default_rng(42)
    weeks = list(range(1, 17))
    
    fig = go.Figure()
    
    for i in range(n_students):
        base = df[col].iloc[i]
        # Oscillation réaliste autour de la valeur de base
        values = base + 2 * np.sin(np.linspace(0, 3*np.pi, 16)) + rng.normal(0, 0.8, 16)
        values = np.clip(values, 0, 21)
        
        fig.add_trace(go.Scatter(
            x=weeks, y=values,
            mode='lines+markers',
            name=f'Étudiant {i+1}',
            line=dict(width=2),
            marker=dict(size=5)
        ))
    
    fig.update_layout(
        title='Évolution des connexions hebdomadaires (16 semaines)',
        xaxis_title='Semaine',
        yaxis_title='Nb connexions',
        plot_bgcolor='white',
        hovermode='x unified'
    )
    return fig


# ════════════════════════════════════════════════════════════
# DIAGRAMME 7 — NUAGE DE POINTS (scatter) + droite de régression
# ════════════════════════════════════════════════════════════

def plot_scatter_regression(
    df: pd.DataFrame,
    x_col: str = 'temps_etude_hebdo',
    y_col: str = 'note_finale',
    hue: Optional[str] = 'reussite'
) -> plt.Figure:
    """
    Diagramme : Nuage de points avec droite de régression.
    Objectif  : Visualiser la relation linéaire entre deux variables.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    palette = {0: '#FF5722', 1: '#4CAF50'}
    labels = {0: 'Échec (< 10)', 1: 'Réussite (≥ 10)'}
    
    if hue:
        for val in sorted(df[hue].unique()):
            mask = df[hue] == val
            ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                      c=palette[val], label=labels[val], alpha=0.6, s=40)
    else:
        ax.scatter(df[x_col], df[y_col], alpha=0.5, color='#2196F3', s=40)
    
    # Droite de régression
    slope, intercept, r_value, p_value, _ = stats.linregress(
        df[x_col].dropna(), df[y_col].dropna()
    )
    x_line = np.linspace(df[x_col].min(), df[x_col].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color='darkblue', linewidth=2,
            label=f'Régression : y={slope:.2f}x+{intercept:.2f}  (R²={r_value**2:.3f})')
    
    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(y_col, fontsize=11)
    ax.set_title(f'Relation entre {x_col} et {y_col}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(10, color='gray', linestyle='--', alpha=0.5, label='Seuil réussite')
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════
# DIAGRAMME 8 — RADAR CHART (Profil des clusters)
# ════════════════════════════════════════════════════════════

def plot_radar_clusters(
    cluster_means: pd.DataFrame,
    features: list,
    title: str = 'Profil moyen des clusters'
) -> go.Figure:
    """
    Diagramme : Toile d'araignée (radar) pour comparer les profils des clusters.
    Objectif  : Caractériser et nommer chaque groupe (K-Means ou CAH).
    """
    fig = go.Figure()
    
    colors_radar = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
    
    for i, (_, row) in enumerate(cluster_means.iterrows()):
        values = row[features].tolist()
        values += values[:1]  # Fermer le radar
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=features + [features[0]],
            fill='toself',
            opacity=0.6,
            name=f'Cluster {i}',
            line=dict(color=colors_radar[i % len(colors_radar)])
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title=title,
        showlegend=True
    )
    return fig


# ════════════════════════════════════════════════════════════
# DIAGRAMME 9 — PAIRPLOT (Seaborn)
# ════════════════════════════════════════════════════════════

def plot_pairplot(
    df: pd.DataFrame,
    cols: list,
    hue: str = 'reussite',
    title: str = 'Pairplot — Relations multivariées'
) -> plt.Figure:
    """
    Diagramme : Matrice de nuages de points croisés.
    Objectif  : Aperçu global des relations entre variables sélectionnées.
    """
    palette = {0: '#FF5722', 1: '#4CAF50'}
    g = sns.pairplot(
        df[cols + [hue]],
        hue=hue,
        palette=palette,
        diag_kind='kde',
        plot_kws={'alpha': 0.5, 's': 20},
        height=2.2
    )
    g.fig.suptitle(title, y=1.01, fontsize=13, fontweight='bold')
    return g.fig
```

---

## 3. CATALOGUE DES DIAGRAMMES — GUIDE D'INTERPRÉTATION

### Diagramme 1 — Histogramme + KDE
```
   Fréquence
     │
  60 ┤         ████
  50 ┤       ████████         ← KDE (courbe lissée)
  40 ┤     ████████████
  30 ┤  ██████████████████
  20 ┤ ████████████████████
  10 ┤███████████████████████
     └────────────────────────→ note_finale
     0    5    10   15   20
          ↑ Médiane  ↑ Moyenne
```
**Interprétation :** Permet de vérifier si la distribution est normale (hypothèse de la régression). L'écart entre la moyenne et la médiane révèle l'asymétrie.

---

### Diagramme 2 — Camembert
```
        ┌─────────────────┐
        │   genre M 52%   │
        │  ╭──────────╮   │
        │  │ ████████ │   │  ← Secteur F 48%
        │  │████  ████│   │
        │  │  ████    │   │
        │  ╰──────────╯   │
        └─────────────────┘
```
**Utilisation :** `genre`, `niveau_etudes`, `revenu_famille`, `acces_internet`, `reussite`

---

### Diagramme 3 — Barres (comparaison)
```
Note moyenne
    │
 14 ┤  ████
 12 ┤  ████  ████
 10 ┤  ████  ████  ████
  8 ┤  ████  ████  ████  ████
    └──────────────────────────
      Stable Instable Limité
              acces_internet
```
**Interprétation :** L'accès internet stable est corrélé à de meilleures performances moyennes.

---

### Diagramme 6 — Graphe oscillant (série temporelle)
```
Connexions/sem.
   │
15 ┤  ╱╲  ╱╲    ← Étudiant 1 (motivé, oscillations régulières)
10 ┤╱   ╲╱  ╲╱╲
 5 ┤           ╲╱   ← Étudiant 2 (décrochage semaines 12-16)
   └──────────────────→ Semaines (1 → 16)
      S4  S8  S12 S16
```
**Interprétation :** Permet de détecter les profils de décrochage et d'assiduité cyclique.

---

### Diagramme 8 — Radar (Toile d'araignée)
```
              connexions
                  │
    motivation ───┼─── exercices
         ╲    ████│████    ╱
          ╲  ██   │   ██  ╱
           ╲██    │    ██╱
    devoirs─╳─────┼─────╳─absences
           ╱██    │    ██╲
          ╱  ██   │   ██  ╲
         ╱    ████│████    ╲
         videos   ─── temps_etude
```
**Lecture :** Plus la surface colorée est grande, plus le cluster a des scores élevés sur ces axes.

---

## 4. CODE D'EXÉCUTION COMPLET DE L'EDA

```python
# notebooks/01_exploration.ipynb — Cellule principale
import pandas as pd
from src.analysis import compute_descriptive_stats, detect_outliers_zscore
from src.visualization import (
    plot_histogram_kde, plot_pie_chart, plot_bar_chart_interactive,
    plot_boxplot_multi, plot_correlation_heatmap,
    plot_oscillating_progression, plot_scatter_regression
)

# Charger données
df = pd.read_csv("data/processed/elearning_clean.csv")

# 1. Statistiques descriptives
stats = compute_descriptive_stats(df)
print("=== STATISTIQUES DESCRIPTIVES ===")
print(stats['summary'])
print("\n=== ASYMÉTRIE (Skewness) ===")
print(stats['skewness'])
print("\n=== CORRÉLATIONS AVEC NOTE FINALE ===")
print(stats['correlations']['note_finale'].sort_values(ascending=False))

# 2. Générer tous les diagrammes
num_cols_for_boxplot = [
    'temps_etude_hebdo', 'nb_connexions_semaine', 'exercices_completes_pct',
    'score_motivation', 'nombre_absences', 'note_finale'
]

fig1 = plot_histogram_kde(df, 'note_finale', 'Distribution des notes finales')
fig2 = plot_pie_chart(df, 'genre', 'Répartition par genre')
fig3 = plot_pie_chart(df, 'reussite', 'Taux de réussite global')
fig4 = plot_bar_chart_interactive(df, 'niveau_etudes')
fig5 = plot_bar_chart_interactive(df, 'acces_internet')
fig6 = plot_boxplot_multi(df, num_cols_for_boxplot)
fig7 = plot_correlation_heatmap(df)
fig8 = plot_oscillating_progression(df)
fig9 = plot_scatter_regression(df, 'temps_etude_hebdo', 'note_finale')
fig10 = plot_scatter_regression(df, 'nb_devoirs_rendus', 'note_finale')

# Sauvegarder
for name, fig in [('histo_notes', fig1), ('pie_genre', fig2), ('corr_heatmap', fig7)]:
    fig.savefig(f"rapport/figures/{name}.png", dpi=150, bbox_inches='tight')
```

---

*Document 4/10 — INF 232 EC2*
