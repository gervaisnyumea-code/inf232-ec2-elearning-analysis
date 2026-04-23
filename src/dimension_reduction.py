# ============================================================
# src/dimension_reduction.py
# INF 232 EC2 — ACP, t-SNE, LDA
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional


COLORS_CLUSTER = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
COLORS_CLASS   = {0: '#FF5722', 1: '#4CAF50'}
LABELS_CLASS   = {0: 'Échec', 1: 'Réussite'}


# ════════════════════════════════════════════════════════════
# ACP — ANALYSE EN COMPOSANTES PRINCIPALES
# ════════════════════════════════════════════════════════════

class AnalyseACP:
    """
    Pipeline ACP complet avec toutes les visualisations.
    """
    
    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.pca_full = PCA()  # Pour la variance expliquée totale
        self.is_fitted = False
    
    def fit_transform(
        self,
        X_scaled: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """
        Ajuste et transforme les données.
        """
        self.feature_names = feature_names
        
        # ACP complète (pour variance totale)
        self.pca_full.fit(X_scaled)
        
        # ACP réduite
        self.X_pca = self.pca.fit_transform(X_scaled)
        
        self.explained_variance_ratio = self.pca_full.explained_variance_ratio_
        self.components_ = self.pca.components_  # Shape: (n_components, n_features)
        self.loadings_ = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
        
        cumulative = np.cumsum(self.explained_variance_ratio)
        n95 = np.searchsorted(cumulative, 0.95) + 1
        
        print(f"ACP : {self.n_components} composantes retenues")
        print(f"Variance expliquée PC1 : {self.explained_variance_ratio[0]:.1%}")
        print(f"Variance expliquée PC2 : {self.explained_variance_ratio[1]:.1%}")
        print(f"Variance cumulée PC1+PC2 : {cumulative[1]:.1%}")
        print(f"Composantes pour 95% variance : {n95}")
        
        self.is_fitted = True
        return self.X_pca
    
    # ── Diagramme 18 — Biplot ACP ────────────────────────────
    
    def plot_biplot(
        self,
        labels: Optional[np.ndarray] = None,
        label_type: str = 'cluster',
        scale: float = 3.0,
        title: str = 'Biplot ACP — Individus + Variables'
    ) -> plt.Figure:
        """
        DIAGRAMME 18 — Biplot ACP.
        Superpose la projection des individus et des vecteurs de chargement des variables.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # ── Points (individus) ───────────────────────────────
        if labels is not None:
            unique_labels = np.unique(labels)
            if label_type == 'class':
                palette = COLORS_CLASS
                legend_labels = LABELS_CLASS
            else:
                palette = {l: COLORS_CLUSTER[i % len(COLORS_CLUSTER)] 
                          for i, l in enumerate(unique_labels)}
                legend_labels = {l: f'Cluster {l}' for l in unique_labels}
            
            for lbl in unique_labels:
                mask = labels == lbl
                ax.scatter(
                    self.X_pca[mask, 0], self.X_pca[mask, 1],
                    c=palette[lbl], alpha=0.6, s=30,
                    label=legend_labels[lbl], zorder=2
                )
        else:
            ax.scatter(self.X_pca[:, 0], self.X_pca[:, 1],
                      alpha=0.5, color='#2196F3', s=25, zorder=2)
        
        # ── Vecteurs de chargement ───────────────────────────
        for i, feature in enumerate(self.feature_names):
            lx = self.loadings_[i, 0] * scale
            ly = self.loadings_[i, 1] * scale
            
            ax.annotate(
                '', xy=(lx, ly), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
            )
            ax.text(lx * 1.12, ly * 1.12, feature, color='red',
                   fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        # ── Cercle de corrélation ────────────────────────────
        circle = plt.Circle((0, 0), scale, color='gray', fill=False,
                           linestyle='--', alpha=0.4)
        ax.add_patch(circle)
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
        
        ax.set_xlabel(
            f'PC1 ({self.explained_variance_ratio[0]:.1%} variance)',
            fontsize=11
        )
        ax.set_ylabel(
            f'PC2 ({self.explained_variance_ratio[1]:.1%} variance)',
            fontsize=11
        )
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        plt.tight_layout()
        return fig
    
    # ── Diagramme 19 — Variance expliquée (barres + courbe cumulée) ──
    
    def plot_explained_variance(self) -> plt.Figure:
        """
        DIAGRAMME 19 — Variance expliquée par composante.
        Diagramme en barres + courbe cumulée.
        """
        n_show = min(15, len(self.explained_variance_ratio))
        var = self.explained_variance_ratio[:n_show] * 100
        cumvar = np.cumsum(self.explained_variance_ratio[:n_show]) * 100
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        # Barres
        bars = ax1.bar(range(1, n_show + 1), var, color='#2196F3',
                      alpha=0.8, edgecolor='white', label='Variance expliquée (%)')
        ax1.set_xlabel('Composante Principale', fontsize=11)
        ax1.set_ylabel('Variance expliquée (%)', fontsize=11, color='#2196F3')
        ax1.tick_params(axis='y', labelcolor='#2196F3')
        
        # Courbe cumulée
        ax2 = ax1.twinx()
        ax2.plot(range(1, n_show + 1), cumvar, 'ro-', linewidth=2,
                markersize=6, label='Variance cumulée (%)')
        ax2.axhline(95, color='gray', linestyle='--', alpha=0.7, label='Seuil 95%')
        ax2.set_ylabel('Variance cumulée (%)', fontsize=11, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim([0, 105])
        
        # Valeurs sur les barres
        for bar, v in zip(bars, var):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Légende combinée
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='center right')
        
        ax1.set_title('ACP — Variance Expliquée par Composante Principale',
                     fontsize=13, fontweight='bold')
        ax1.set_xticks(range(1, n_show + 1))
        ax1.set_xticklabels([f'PC{i}' for i in range(1, n_show + 1)], rotation=45)
        plt.tight_layout()
        return fig
    
    def plot_loadings_heatmap(self) -> plt.Figure:
        """
        Heatmap des chargements ACP : contribution de chaque variable originale
        à chaque composante principale.
        """
        n_pc = min(5, self.pca.n_components_)
        loadings_df = pd.DataFrame(
            self.pca.components_[:n_pc].T,
            index=self.feature_names,
            columns=[f'PC{i+1}' for i in range(n_pc)]
        )
        
        fig, ax = plt.subplots(figsize=(9, 8))
        sns.heatmap(
            loadings_df, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            cbar_kws={'label': 'Chargement'}, ax=ax
        )
        ax.set_title('Chargements ACP — Contribution des variables aux PC',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig


# ════════════════════════════════════════════════════════════
# t-SNE
# ════════════════════════════════════════════════════════════

class VisualisationTSNE:
    """
    t-SNE pour visualisation 2D non linéaire.
    """
    
    def __init__(
        self,
        perplexity: float = 30.0,
        n_iter: int = 1000,
        random_state: int = 42
    ) -> None:
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.random_state = random_state
        self.tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state,
            learning_rate='auto',
            init='pca'
        )
    
    def fit_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Applique t-SNE (utiliser d'abord PCA pour réduire le bruit).
        """
        print(f"Calcul t-SNE (perplexity={self.perplexity}, n_iter={self.n_iter})...")
        self.X_tsne = self.tsne.fit_transform(X_scaled)
        print(f"✅ t-SNE calculé — KL divergence finale : {self.tsne.kl_divergence_:.4f}")
        return self.X_tsne
    
    # ── Diagramme 20 — t-SNE scatter ────────────────────────
    
    def plot_tsne_scatter(
        self,
        labels: np.ndarray,
        label_type: str = 'cluster',
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        DIAGRAMME 20 — Nuage de points t-SNE coloré par clusters ou classes.
        Révèle les structures non linéaires dans les données.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        unique_labels = np.unique(labels)
        
        if label_type == 'class':
            palette = COLORS_CLASS
            legend_labels = LABELS_CLASS
        else:
            palette = {l: COLORS_CLUSTER[i % len(COLORS_CLUSTER)]
                      for i, l in enumerate(unique_labels)}
            legend_labels = {l: f'Cluster {l}' for l in unique_labels}
        
        for lbl in unique_labels:
            mask = labels == lbl
            ax.scatter(
                self.X_tsne[mask, 0], self.X_tsne[mask, 1],
                c=palette[lbl], label=legend_labels[lbl],
                alpha=0.7, s=30, edgecolors='none'
            )
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
        ax.set_title(
            title or f't-SNE (perplexity={self.perplexity}) — '
                     f'Visualisation des {"clusters" if label_type=="cluster" else "classes"}',
            fontsize=13, fontweight='bold'
        )
        ax.legend(fontsize=10, markerscale=1.5)
        ax.text(0.02, 0.02,
               "⚠️ Les distances absolues ne sont pas interprétables en t-SNE",
               transform=ax.transAxes, fontsize=8, color='gray')
        plt.tight_layout()
        return fig
    
    def plot_tsne_interactive(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        hover_cols: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Version interactive Plotly du scatter t-SNE avec hover données.
        """
        df_plot = df.copy()
        df_plot['tsne_1'] = self.X_tsne[:, 0]
        df_plot['tsne_2'] = self.X_tsne[:, 1]
        df_plot['label'] = labels.astype(str)
        
        fig = px.scatter(
            df_plot, x='tsne_1', y='tsne_2',
            color='label',
            hover_data=hover_cols or ['note_finale', 'reussite', 'temps_etude_hebdo'],
            title=f't-SNE Interactif (perplexity={self.perplexity})',
            color_discrete_sequence=COLORS_CLUSTER,
            labels={'tsne_1': 't-SNE 1', 'tsne_2': 't-SNE 2', 'label': 'Groupe'}
        )
        fig.update_traces(marker=dict(size=6, opacity=0.7))
        fig.update_layout(plot_bgcolor='white')
        return fig
    

# ════════════════════════════════════════════════════════════
# LDA — ANALYSE DISCRIMINANTE LINÉAIRE
# ════════════════════════════════════════════════════════════

class AnalyseLDA:
    """
    LDA supervisée pour réduction de dimension et classification.
    """
    
    def __init__(self) -> None:
        self.lda = LinearDiscriminantAnalysis()
        self.is_fitted = False
    
    def fit_transform(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """
        Ajuste la LDA et projette les données.
        Pour une classification binaire (réussite), produit 1 axe discriminant.
        """
        self.feature_names = feature_names
        self.classes_ = np.unique(y)
        self.X_lda = self.lda.fit_transform(X_scaled, y)
        self.y = y
        
        # Score de classification via LDA
        lda_accuracy = self.lda.score(X_scaled, y)
        print(f"LDA — Exactitude en re-substitution : {lda_accuracy:.3f}")
        print(f"LDA — Coefs de séparation :")
        for fname, coef in zip(feature_names, self.lda.coef_[0]):
            print(f"  {fname:30s}: {coef:+.4f}")
        
        self.is_fitted = True
        return self.X_lda
    
    # ── Diagramme 21 — Projection LDA ───────────────────────
    
    def plot_lda_projection(self) -> plt.Figure:
        """
        DIAGRAMME 21 — Projection LDA et séparation des classes.
        Pour K=2 : histogrammes chevauchants des deux classes sur l'axe discriminant.
        """
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        
        # ── Histogramme de la projection ────────────────────
        colors = [COLORS_CLASS[c] for c in sorted(self.classes_)]
        
        for cls, color, label in zip(sorted(self.classes_), colors,
                                    [LABELS_CLASS[c] for c in sorted(self.classes_)]):
            mask = self.y == cls
            vals = self.X_lda[mask, 0] if self.X_lda.ndim > 1 else self.X_lda[mask]
            axes[0].hist(vals, bins=25, alpha=0.65, color=color,
                        edgecolor='white', label=label, density=True)
        
        axes[0].set_xlabel('Axe discriminant LD1', fontsize=11)
        axes[0].set_ylabel('Densité', fontsize=11)
        axes[0].set_title('Distribution des classes sur l\'axe LDA', fontsize=12, fontweight='bold')
        axes[0].legend()
        
        # ── Coefficients discriminants ───────────────────────
        coefs = pd.Series(
            np.abs(self.lda.coef_[0]),
            index=self.feature_names
        ).sort_values(ascending=True)
        
        colors_bar = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(coefs)))
        coefs.plot(kind='barh', ax=axes[1], color=colors_bar, edgecolor='white')
        axes[1].set_xlabel('|Coefficient LDA| (contribution)', fontsize=11)
        axes[1].set_title('Variables les plus discriminantes (LDA)', fontsize=12, fontweight='bold')
        
        for i, v in enumerate(coefs.values):
            axes[1].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)
        
        plt.suptitle('Analyse Discriminante Linéaire (LDA)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


# ════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════

def run_dimension_reduction_pipeline(
    df_clean: pd.DataFrame,
    cluster_labels: np.ndarray
) -> Dict:
    """
    Pipeline complet de réduction de dimension.
    """
    from src.data_cleaning import get_feature_matrix
    
    X, y_reg, y_clf = get_feature_matrix(df_clean)
    feature_cols = [
        'temps_etude_hebdo', 'nb_devoirs_rendus', 'exercices_completes_pct',
        'nb_connexions_semaine', 'score_motivation', 'nombre_absences',
        'videos_vues_pct', 'note_mi_parcours'
    ]
    X_sub = X[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)
    
    # ── ACP ──────────────────────────────────────────────────
    print("=" * 50)
    print("ACP")
    acp = AnalyseACP(n_components=2)
    X_pca = acp.fit_transform(X_scaled, feature_cols)
    
    # ── t-SNE ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("t-SNE")
    tsne_viz = VisualisationTSNE(perplexity=30, n_iter=1000)
    X_tsne = tsne_viz.fit_transform(X_scaled)
    
    # ── LDA ──────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("LDA")
    lda_viz = AnalyseLDA()
    X_lda = lda_viz.fit_transform(X_scaled, y_clf.values, feature_cols)
    
    return {
        'acp': acp, 'tsne': tsne_viz, 'lda': lda_viz,
        'X_pca': X_pca, 'X_tsne': X_tsne, 'X_lda': X_lda,
        'y_clf': y_clf.values, 'cluster_labels': cluster_labels
    }


if __name__ == "__main__":
    import numpy as np
    df = pd.read_csv("data/processed/elearning_clean.csv")
    dummy_labels = np.random.choice(3, len(df))
    results = run_dimension_reduction_pipeline(df, dummy_labels)