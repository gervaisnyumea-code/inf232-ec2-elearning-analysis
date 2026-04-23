# ============================================================
# src/visualization.py
# INF 232 EC2 — Visualisations pour EDA et modélisation
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional


# Configuration globale
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_histogram_kde(df: pd.DataFrame, col: str, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche histogramme + KDE pour une variable numérique."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x=col, kde=True, ax=ax, color='steelblue', alpha=0.7)
    ax.set_title(f'Distribution de {col}', fontsize=12, fontweight='bold')
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel('Fréquence', fontsize=10)
    return ax


def plot_pie_chart(df: pd.DataFrame, col: str, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche un camembert pour une variable catégorielle."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    counts = df[col].value_counts()
    colors = sns.color_palette('husl', len(counts))
    wedges, texts, autotexts = ax.pie(
        counts, labels=counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90, explode=[0.02] * len(counts)
    )
    ax.set_title(f'Distribution de {col}', fontsize=12, fontweight='bold')
    return ax


def plot_bar_chart(df: pd.DataFrame, col: str, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche un diagramme en barres pour une variable catégorielle."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    counts = df[col].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette='viridis')
    ax.set_title(f'Distribution de {col}', fontsize=12, fontweight='bold')
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel('Effectif', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    return ax


def plot_boxplot(df: pd.DataFrame, col: str, hue: Optional[str] = None, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche un boxplot pour détecter les outliers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x=col, y=hue, ax=ax, palette='Set2') if hue else \
        sns.boxplot(data=df, x=col, ax=ax, palette='Set2')
    ax.set_title(f'Boxplot de {col}', fontsize=12, fontweight='bold')
    return ax


def plot_heatmap_correlation(df: pd.DataFrame, cols: Optional[list] = None, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche la heatmap des corrélations."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    corr = df[cols].corr() if cols else df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Matrice de corrélation', fontsize=12, fontweight='bold')
    return ax


def plot_pairplot(df: pd.DataFrame, cols: list, hue: Optional[str] = None, sample: int = 100) -> plt.Figure:
    """Affiche un pairplot pour les relations multivariées."""
    df_sample = df.sample(n=min(sample, len(df)), random_state=42)
    g = sns.pairplot(df_sample[cols], hue=hue, diag_kind='kde', corner=True)
    g.fig.suptitle('Pairplot - Relations multivariées', y=1.02, fontsize=14, fontweight='bold')
    return g.fig


def plot_scatter_simple(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str,
                        title: str, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche un nuage de points pour régression simple."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=0.6, c='steelblue', edgecolors='white', s=60)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    return ax


def plot_regression_line(x: np.ndarray, y: np.ndarray, y_pred: np.ndarray,
                         xlabel: str, ylabel: str, title: str,
                         ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche le nuage de points avec la droite de régression."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=0.6, c='steelblue', label='Données', s=50)
    ax.plot(x, y_pred, color='red', linewidth=2, label='Droite de régression')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    return ax


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche le graphique Résidus vs Fitted pour diagnostic."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.6, c='coral', edgecolors='white', s=60)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Valeurs prédites', fontsize=11)
    ax.set_ylabel('Résidus', fontsize=11)
    ax.set_title('Résidus vs Valeurs prédites', fontsize=12, fontweight='bold')
    return ax


def plot_qqplot(residuals: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche le QQ-plot pour vérifier la normalité des résidus."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('QQ-Plot des résidus', fontsize=12, fontweight='bold')
    return ax


def plot_learning_curve(train_scores: np.ndarray, test_scores: np.ndarray,
                        ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche la courbe d'apprentissage."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_scores, label='Score entraînement', c='steelblue', linewidth=2)
    ax.plot(test_scores, label='Score validation', c='coral', linewidth=2)
    ax.set_xlabel('Nombre d\'itérations', fontsize=11)
    ax.set_ylabel('Score R²', fontsize=11)
    ax.set_title('Courbe d\'apprentissage', fontsize=12, fontweight='bold')
    ax.legend()
    return ax


def plot_confusion_matrix(cm: np.ndarray, labels: list, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche la matrice de confusion."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels,
                yticklabels=labels, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Prédictions', fontsize=11)
    ax.set_ylabel('Réelles', fontsize=11)
    ax.set_title('Matrice de confusion', fontsize=12, fontweight='bold')
    return ax


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: float,
                   ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche la courbe ROC."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'Courbe ROC (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taux de faux positifs', fontsize=11)
    ax.set_ylabel('Taux de vrais positifs', fontsize=11)
    ax.set_title('Courbe ROC', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    return ax


def plot_feature_importance(features: list, importance: np.ndarray,
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche l'importance des variables pour Random Forest."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    indices = np.argsort(importance)[::-1]
    sns.barplot(x=importance[indices], y=[features[i] for i in indices], ax=ax, palette='viridis')
    ax.set_title('Importance des variables (Random Forest)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Importance', fontsize=11)
    return ax


def plot_elbow_method(inertias: list, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche la méthode du coude pour K-Means."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(inertias) + 1), inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Nombre de clusters (k)', fontsize=11)
    ax.set_ylabel('Inertie', fontsize=11)
    ax.set_title('Méthode du coude', fontsize=12, fontweight='bold')
    ax.set_xticks(range(1, len(inertias) + 1))
    return ax


def plot_silhouette(silhouette_scores: list, labels: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche le silhouette plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10
    for i in range(max(labels) + 1):
        cluster_silhouette_values = silhouette_scores[labels == i]
        cluster_silhouette_values.sort()
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax.set_title('Silhouette Plot', fontsize=12, fontweight='bold')
    ax.set_xlabel('Coefficient de silhouette', fontsize=11)
    ax.set_ylabel('Cluster', fontsize=11)
    return ax


def plot_dendrogram(linkage, labels: list, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche le dendrogramme pour CAH."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    from scipy.cluster.hierarchy import dendrogram
    dendrogram(linkage, labels=labels, leaf_rotation=90, ax=ax)
    ax.set_title('Dendrogramme (CAH)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Index ou taille du cluster', fontsize=11)
    ax.set_ylabel('Distance', fontsize=11)
    return ax


def plot_pca_variance(explained_variance: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche la variance expliquée par les composantes ACP."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    cumsum = np.cumsum(explained_variance)
    ax.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Variance individuelle')
    ax.plot(range(1, len(cumsum) + 1), cumsum, 'ro-', label='Variance cumulative')
    ax.set_xlabel('Composante principale', fontsize=11)
    ax.set_ylabel('Variance expliquée', fontsize=11)
    ax.set_title('Variance expliquée (ACP)', fontsize=12, fontweight='bold')
    ax.legend()
    return ax


def plot_pca_biplot(pca, X_scaled: np.ndarray, features: list, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche le biplot ACP."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    # Projection des individus
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.5, c='steelblue', s=30)
    # Vecteurs des variables
    for i, feature in enumerate(features):
        ax.arrow(0, 0, pca.components_[0, i], pca.components_[1, i],
                head_width=0.05, head_length=0.03, fc='red', ec='red')
        ax.text(pca.components_[0, i] * 1.1, pca.components_[1, i] * 1.1, feature, fontsize=9)
    ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%})', fontsize=11)
    ax.set_title('Biplot ACP', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    return ax


def plot_tsne(X_tsne: np.ndarray, labels: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche le scatter plot t-SNE coloré par cluster."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.set_title('Visualisation t-SNE', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    return ax


def plot_lda_projection(X_lda: np.ndarray, y: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche la projection LDA."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    classes = np.unique(y)
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    for cls, color in zip(classes, colors):
        mask = y == cls
        ax.scatter(X_lda[mask, 0], X_lda[mask, 1], c=[color], label=f'Classe {cls}', alpha=0.7, s=50)
    ax.set_xlabel('LDA 1', fontsize=11)
    ax.set_ylabel('LDA 2', fontsize=11)
    ax.set_title('Projection LDA', fontsize=12, fontweight='bold')
    ax.legend()
    return ax


def plot_radar_chart(df: pd.DataFrame, categories: list, cluster_col: str,
                     ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Affiche un radar chart pour les profils des clusters."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    clusters = df[cluster_col].unique()
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for cluster in clusters:
        values = df[df[cluster_col] == cluster][categories].mean().tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title('Profil des clusters (Radar)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    return ax