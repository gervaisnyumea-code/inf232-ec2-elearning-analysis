# ============================================================
# src/visualization_extra.py
# Extra visualisations issues de DOCS (interactive & multi-plots)
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import Optional, List

sns.set_palette("husl")


def plot_bar_chart_interactive(
    df: pd.DataFrame,
    col: str,
    target: str = 'note_finale',
    title: Optional[str] = None
) -> go.Figure:
    """Barres interactives Plotly : moyenne de la variable cible par catégorie."""
    agg = df.groupby(col)[target].mean().reset_index().sort_values(target, ascending=False)
    fig = px.bar(
        agg, x=col, y=target,
        text=agg[target].round(2),
        color=col,
        color_discrete_sequence=px.colors.qualitative.Set2,
        title=title or f'Moyenne {target} par {col}',
        labels={col: col, target: f'Moyenne {target}'}
    )
    if not agg.empty:
        fig.update_traces(textposition='outside')
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            yaxis=dict(range=[0, max(agg[target]) * 1.15])
        )
    else:
        fig.update_layout(plot_bgcolor='white')
    return fig


def plot_boxplot_multi(
    df: pd.DataFrame,
    cols: List[str],
    title: str = 'Détection des outliers — Boxplots'
) -> plt.Figure:
    """Boîtes à moustaches pour plusieurs variables."""
    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()
    palette = sns.color_palette("husl", n)

    for i, col in enumerate(cols):
        sns.boxplot(y=df[col], ax=axes[i], color=palette[i % len(palette)],
                    flierprops=dict(marker='o', color='red', markersize=4))
        axes[i].set_title(col, fontweight='bold')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


def plot_oscillating_progression(
    df: pd.DataFrame,
    col: str = 'nb_connexions_semaine',
    n_students: int = 5
) -> go.Figure:
    """Évolution simulée des connexions sur 16 semaines (oscillant)."""
    rng = np.random.default_rng(42)
    weeks = list(range(1, 17))
    fig = go.Figure()

    for i in range(n_students):
        if col in df.columns and len(df) > i:
            base = df[col].iloc[i]
        else:
            base = 5
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


def plot_scatter_regression(
    df: pd.DataFrame,
    x_col: str = 'temps_etude_hebdo',
    y_col: str = 'note_finale',
    hue: Optional[str] = 'reussite'
) -> plt.Figure:
    """Nuage de points avec droite de régression et R²."""
    fig, ax = plt.subplots(figsize=(9, 6))

    palette = {0: '#FF5722', 1: '#4CAF50'}
    labels = {0: 'Échec (< 10)', 1: 'Réussite (≥ 10)'}

    if hue and hue in df.columns:
        for val in sorted(df[hue].dropna().unique()):
            mask = df[hue] == val
            ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                       c=palette.get(val, '#2196F3'), label=labels.get(val, str(val)), alpha=0.6, s=40)
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


def plot_radar_clusters(cluster_means: pd.DataFrame, features: List[str], title: str = 'Profil moyen des clusters') -> go.Figure:
    """Toile d'araignée (radar) pour comparer les profils des clusters (Plotly)."""
    fig = go.Figure()
    colors_radar = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']

    for i, (_, row) in enumerate(cluster_means.iterrows()):
        values = row[features].tolist()
        values += values[:1]

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
