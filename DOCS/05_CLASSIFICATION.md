# INF 232 EC2 — CLASSIFICATION SUPERVISÉE & NON SUPERVISÉE
## K-Means, CAH, Random Forest, KNN + Diagrammes

---

## 1. CLASSIFICATION SUPERVISÉE

### 1.1 Problématique

**Cible :** `reussite` (0 = Échec, 1 = Succès) — binaire  
**Objectif :** Prédire si un étudiant va réussir à partir de ses comportements dès la mi-parcours

### 1.2 Algorithmes Implémentés

| Algorithme | Avantages | Inconvénients | Complexité |
|------------|-----------|---------------|------------|
| **Régression Logistique** | Interprétable, rapide | Hypothèse linéaire | O(nK) |
| **Random Forest** | Robuste, non linéaire | Moins interprétable | O(n·√K·T) |
| **KNN** | Simple, pas d'hypothèse | Lent à prédire | O(n) prédiction |
| **SVM** | Performant sur petits datasets | Lent sur grands | O(n²→n³) |

---

## 2. MODULE `src/classification.py`

```python
# ============================================================
# src/classification.py
# INF 232 EC2 — Classification Supervisée et Non Supervisée
# ============================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve,
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path


# ════════════════════════════════════════════════════════════
# SECTION A — CLASSIFICATION SUPERVISÉE
# ════════════════════════════════════════════════════════════

class ClassificationSupervisée:
    """
    Pipeline de classification supervisée multi-algorithmes.
    Compare Régression Logistique, Random Forest, KNN, SVM.
    """
    
    def __init__(
        self,
        feature_names: List[str],
        target_name: str = 'reussite',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> None:
        self.feature_names = feature_names
        self.target_name = target_name
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models: Dict = {}
        self.results: Dict = {}
    
    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> None:
        """
        Prépare les données : split train/test + standardisation.
        Note : On standardise après le split pour éviter le data leakage.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y  # Stratification pour conserver les proportions
        )
        # Standardisation
        self.X_train_sc = self.scaler.fit_transform(self.X_train)
        self.X_test_sc  = self.scaler.transform(self.X_test)
        
        print(f"<img src=app/static/icons/check.svg alt=check width=18/> Données préparées | Train: {len(self.X_train)} | Test: {len(self.X_test)}")
        print(f"   Taux réussite train: {self.y_train.mean():.1%} | test: {self.y_test.mean():.1%}")
    
    def define_models(self) -> None:
        """Initialise les 4 modèles de classification."""
        self.models = {
            'Régression Logistique': LogisticRegression(
                max_iter=1000, random_state=self.random_state, C=1.0
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=8,
                random_state=self.random_state, n_jobs=-1
            ),
            'KNN (k=7)': KNeighborsClassifier(
                n_neighbors=7, metric='euclidean'
            ),
            'SVM (RBF)': SVC(
                kernel='rbf', C=1.0, probability=True,
                random_state=self.random_state
            ),
        }
    
    def train_all(self) -> None:
        """Entraîne tous les modèles et collecte leurs métriques."""
        self.define_models()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            model.fit(self.X_train_sc, self.y_train)
            y_pred = model.predict(self.X_test_sc)
            y_proba = model.predict_proba(self.X_test_sc)[:, 1]
            
            cv_scores = cross_val_score(
                model, self.X_train_sc, self.y_train,
                cv=cv, scoring='f1', n_jobs=-1
            )
            
            self.results[name] = {
                'model':     model,
                'y_pred':    y_pred,
                'y_proba':   y_proba,
                'accuracy':  round(accuracy_score(self.y_test, y_pred), 4),
                'f1':        round(f1_score(self.y_test, y_pred), 4),
                'precision': round(precision_score(self.y_test, y_pred), 4),
                'recall':    round(recall_score(self.y_test, y_pred), 4),
                'roc_auc':   round(roc_auc_score(self.y_test, y_proba), 4),
                'cv_f1_mean':round(cv_scores.mean(), 4),
                'cv_f1_std': round(cv_scores.std(), 4),
            }
        
        self._print_comparison_table()
    
    def _print_comparison_table(self) -> None:
        """Affiche un tableau comparatif des performances."""
        print("\n╔═══════════════════════════════════════════════════════════════╗")
        print("║         COMPARAISON DES CLASSIFICATEURS                      ║")
        print("╠═══════════════════════╦══════════╦══════╦═══════════╦═══════╣")
        print("║ Modèle                ║ Accuracy ║  F1  ║  ROC-AUC  ║ CV-F1 ║")
        print("╠═══════════════════════╬══════════╬══════╬═══════════╬═══════╣")
        for name, r in self.results.items():
            print(f"║ {name:<21} ║  {r['accuracy']:.3f}   ║{r['f1']:.3f}║  {r['roc_auc']:.3f}    ║ {r['cv_f1_mean']:.3f}║")
        print("╚═══════════════════════╩══════════╩══════╩═══════════╩═══════╝")
    
    def get_best_model_name(self) -> str:
        """Retourne le nom du meilleur modèle selon le F1-score."""
        return max(self.results, key=lambda k: self.results[k]['roc_auc'])
    
    # ── Visualisations ───────────────────────────────────────
    
    def plot_confusion_matrix(self, model_name: Optional[str] = None) -> plt.Figure:
        """
        DIAGRAMME 12 — Matrice de confusion.
        Visualise les vrais positifs, faux positifs, VP, FN.
        """
        if model_name is None:
            model_name = self.get_best_model_name()
        
        y_pred = self.results[model_name]['y_pred']
        cm = confusion_matrix(self.y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Prédit Échec', 'Prédit Réussite'],
            yticklabels=['Réel Échec', 'Réel Réussite'],
            linewidths=2, ax=ax, cbar=False,
            annot_kws={'size': 16, 'fontweight': 'bold'}
        )
        
        # Annotations supplémentaires
        tn, fp, fn, tp = cm.ravel()
        ax.set_title(
            f'Matrice de Confusion — {model_name}\n'
            f'Précision={self.results[model_name]["precision"]:.3f} | '
            f'Rappel={self.results[model_name]["recall"]:.3f} | '
            f'F1={self.results[model_name]["f1"]:.3f}',
            fontsize=11, fontweight='bold'
        )
        ax.set_xlabel('Classe Prédite', fontsize=11)
        ax.set_ylabel('Classe Réelle', fontsize=11)
        plt.tight_layout()
        return fig
    
    def plot_roc_curves(self) -> plt.Figure:
        """
        DIAGRAMME 13 — Courbes ROC pour tous les modèles.
        AUC mesure la capacité discriminante du classifieur.
        """
        fig, ax = plt.subplots(figsize=(8, 7))
        colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
        
        for (name, r), color in zip(self.results.items(), colors):
            fpr, tpr, _ = roc_curve(self.y_test, r['y_proba'])
            ax.plot(fpr, tpr, linewidth=2, color=color,
                    label=f'{name} (AUC = {r["roc_auc"]:.3f})')
        
        # Diagonale
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, label='Classificateur aléatoire')
        
        ax.set_xlabel('Taux de Faux Positifs (FPR)', fontsize=11)
        ax.set_ylabel('Taux de Vrais Positifs (TPR — Rappel)', fontsize=11)
        ax.set_title('Courbes ROC — Comparaison des classificateurs', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        
        # Zone AUC parfaite
        ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self) -> plt.Figure:
        """
        DIAGRAMME 14 — Importance des variables (Random Forest).
        Identifie les variables les plus discriminantes.
        """
        rf = self.results['Random Forest']['model']
        importances = pd.Series(
            rf.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=True)
        
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(importances)))
        bars = importances.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
        
        for i, (v, name) in enumerate(zip(importances.values, importances.index)):
            ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)
        
        ax.set_xlabel("Importance (Gini Mean Decrease)", fontsize=11)
        ax.set_title("Importance des Variables — Random Forest\n(Classification de la réussite)",
                     fontsize=12, fontweight='bold')
        ax.axvline(importances.mean(), color='red', linestyle='--', alpha=0.7,
                   label=f'Importance moyenne ({importances.mean():.3f})')
        ax.legend()
        plt.tight_layout()
        return fig
    
    def plot_metrics_comparison(self) -> go.Figure:
        """
        Diagramme en barres groupées — Comparaison des métriques par modèle.
        """
        metrics = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
        models_list = list(self.results.keys())
        
        fig = go.Figure()
        colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
        
        for metric, color in zip(metrics, colors):
            values = [self.results[m][metric] for m in models_list]
            fig.add_trace(go.Bar(
                name=metric.upper().replace('_', ' '),
                x=models_list, y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='outside',
                marker_color=color,
                opacity=0.85
            ))
        
        fig.update_layout(
            barmode='group',
            title='Comparaison des métriques de classification',
            yaxis=dict(range=[0, 1.15], title='Score'),
            plot_bgcolor='white',
            xaxis_tickangle=-15
        )
        return fig
    
    def save_best_model(self) -> None:
        """Sauvegarde le meilleur modèle."""
        best_name = self.get_best_model_name()
        best_model = self.results[best_name]['model']
        Path("data/models").mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, "data/models/classifier_model.pkl")
        joblib.dump(self.scaler, "data/models/scaler.pkl")
        print(f"<img src=app/static/icons/check.svg alt=check width=18/> Meilleur modèle ({best_name}) sauvegardé.")


# ════════════════════════════════════════════════════════════
# SECTION B — CLASSIFICATION NON SUPERVISÉE
# ════════════════════════════════════════════════════════════

class ClusteringNonSupervisé:
    """
    K-Means et Classification Ascendante Hiérarchique (CAH).
    Segmentation des profils d'étudiants sans labels.
    """
    
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.optimal_k: Optional[int] = None
        self.kmeans_model: Optional[KMeans] = None
        self.labels_: Optional[np.ndarray] = None
    
    def find_optimal_k(
        self,
        X_scaled: np.ndarray,
        k_range: range = range(2, 11)
    ) -> Dict:
        """
        Détermine le nombre optimal de clusters par :
        - La méthode du coude (Elbow)
        - Le score de silhouette
        """
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            km = KMeans(n_clusters=k, init='k-means++',
                       n_init=10, random_state=self.random_state)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, labels))
        
        self.elbow_data = {
            'k': list(k_range),
            'inertia': inertias,
            'silhouette': silhouette_scores
        }
        
        # Optimal k selon silhouette
        self.optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"<img src=app/static/icons/check.svg alt=check width=18/> K optimal (silhouette) : {self.optimal_k}")
        return self.elbow_data
    
    def fit_kmeans(self, X_scaled: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """
        Applique K-Means avec le K optimal.
        
        Returns
        -------
        np.ndarray
            Labels des clusters (0, 1, ..., K-1).
        """
        k = k or self.optimal_k or 3
        self.kmeans_model = KMeans(
            n_clusters=k, init='k-means++',
            n_init=10, random_state=self.random_state
        )
        self.labels_ = self.kmeans_model.fit_predict(X_scaled)
        
        sil = silhouette_score(X_scaled, self.labels_)
        print(f"K-Means (K={k}) | Inertie={self.kmeans_model.inertia_:.1f} | Silhouette={sil:.3f}")
        return self.labels_
    
    # ── Visualisations ───────────────────────────────────────
    
    def plot_elbow_method(self) -> plt.Figure:
        """
        DIAGRAMME 15 — Méthode du coude (Elbow).
        Identifie le K où l'inertie cesse de diminuer fortement.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Elbow
        axes[0].plot(self.elbow_data['k'], self.elbow_data['inertia'],
                    'bo-', linewidth=2, markersize=7)
        axes[0].axvline(self.optimal_k, color='red', linestyle='--', alpha=0.7,
                       label=f'K optimal = {self.optimal_k}')
        axes[0].set_xlabel('Nombre de clusters K', fontsize=11)
        axes[0].set_ylabel('Inertie intra-cluster', fontsize=11)
        axes[0].set_title('Méthode du Coude (Elbow)', fontsize=12, fontweight='bold')
        axes[0].legend()
        
        # Silhouette
        axes[1].plot(self.elbow_data['k'], self.elbow_data['silhouette'],
                    'ro-', linewidth=2, markersize=7)
        axes[1].axvline(self.optimal_k, color='blue', linestyle='--', alpha=0.7,
                       label=f'K optimal = {self.optimal_k}')
        axes[1].set_xlabel('Nombre de clusters K', fontsize=11)
        axes[1].set_ylabel('Score de Silhouette', fontsize=11)
        axes[1].set_title('Score de Silhouette par K', fontsize=12, fontweight='bold')
        axes[1].legend()
        
        plt.suptitle('Détermination du nombre optimal de clusters K-Means',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_silhouette_analysis(self, X_scaled: np.ndarray) -> plt.Figure:
        """
        DIAGRAMME 16 — Silhouette plot détaillé.
        Montre la cohésion intra-cluster et la séparation inter-clusters.
        """
        k = self.optimal_k or 3
        sample_silhouette_values = silhouette_samples(X_scaled, self.labels_)
        avg_sil = silhouette_score(X_scaled, self.labels_)
        
        fig, ax = plt.subplots(figsize=(9, 6))
        y_lower = 10
        colors = plt.cm.Spectral(np.linspace(0, 1, k))
        
        for i in range(k):
            ith_values = np.sort(sample_silhouette_values[self.labels_ == i])
            size = ith_values.shape[0]
            y_upper = y_lower + size
            
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_values, alpha=0.7, color=colors[i],
                            label=f'Cluster {i} (n={size})')
            ax.text(-0.05, y_lower + 0.5 * size, f'C{i}', fontsize=9)
            y_lower = y_upper + 10
        
        ax.axvline(avg_sil, color='red', linestyle='--', linewidth=2,
                  label=f'Silhouette moyenne = {avg_sil:.3f}')
        ax.set_xlabel('Valeur de silhouette', fontsize=11)
        ax.set_ylabel('Cluster', fontsize=11)
        ax.set_title('Analyse de Silhouette — K-Means', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        plt.tight_layout()
        return fig
    
    def plot_dendrogram(
        self,
        X_scaled: np.ndarray,
        n_samples: int = 50,
        method: str = 'ward'
    ) -> plt.Figure:
        """
        DIAGRAMME 17 — Dendrogramme (CAH).
        Visualise la hiérarchie de fusion des groupes.
        
        Parameters
        ----------
        method : str
            Méthode de liaison : 'ward', 'complete', 'average', 'single'
        """
        # Sous-échantillonnage pour lisibilité
        idx = np.random.choice(len(X_scaled), min(n_samples, len(X_scaled)), replace=False)
        X_sample = X_scaled[idx]
        
        linked = linkage(X_sample, method=method)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        dendrogram(
            linked,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True,
            truncate_mode='lastp',
            p=20,
            ax=ax,
            color_threshold=0.7 * max(linked[:, 2])
        )
        ax.set_title(f'Dendrogramme CAH (méthode : {method.capitalize()})',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Échantillons (index)', fontsize=11)
        ax.set_ylabel('Distance de fusion', fontsize=11)
        
        # Ligne de coupure
        cut_height = np.sort(linked[:, 2])[-self.optimal_k + 1]
        ax.axhline(cut_height, color='red', linestyle='--', alpha=0.7,
                  label=f'Coupure → {self.optimal_k} clusters')
        ax.legend()
        plt.tight_layout()
        return fig
    
    def get_cluster_profiles(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Calcule les profils moyens de chaque cluster.
        
        Returns
        -------
        pd.DataFrame
            Moyennes des features par cluster.
        """
        df = df.copy()
        df['cluster'] = self.labels_
        profiles = df.groupby('cluster')[feature_cols + ['note_finale', 'reussite']].mean().round(2)
        
        print("\n=== PROFILS DES CLUSTERS ===")
        print(profiles.to_string())
        return profiles


# ════════════════════════════════════════════════════════════
# SCRIPT D'EXÉCUTION PRINCIPAL
# ════════════════════════════════════════════════════════════

def run_classification_pipeline(df_clean: pd.DataFrame) -> Tuple:
    """Pipeline complet de classification."""
    from src.data_cleaning import get_feature_matrix
    
    X, _, y_clf = get_feature_matrix(df_clean)
    
    feature_cols = [
        'temps_etude_hebdo', 'nb_devoirs_rendus', 'exercices_completes_pct',
        'nb_connexions_semaine', 'score_motivation', 'nombre_absences',
        'videos_vues_pct', 'note_mi_parcours'
    ]
    
    # ── Classification supervisée ────────────────────────────
    print("=" * 55)
    print("CLASSIFICATION SUPERVISÉE")
    clf = ClassificationSupervisée(feature_names=feature_cols)
    clf.prepare_data(X[feature_cols], y_clf)
    clf.train_all()
    clf.save_best_model()
    
    # ── Clustering ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("CLASSIFICATION NON SUPERVISÉE (K-MEANS + CAH)")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[feature_cols])
    
    clustering = ClusteringNonSupervisé()
    clustering.find_optimal_k(X_scaled)
    clustering.fit_kmeans(X_scaled)
    profiles = clustering.get_cluster_profiles(df_clean, feature_cols)
    
    return clf, clustering, profiles


if __name__ == "__main__":
    df = pd.read_csv("data/processed/elearning_clean.csv")
    clf, clustering, profiles = run_classification_pipeline(df)
```

---

## 3. INTERPRÉTATION DES CLUSTERS ATTENDUS

| Cluster | Profil | Caractéristiques | Note Finale Moy. | Réussite |
|---------|--------|------------------|------------------|----------|
| **C0** | <img src=app/static/icons/circle_green.svg alt=circle_green width=18/> Étudiant engagé | Temps > 15h, devoirs rendus ≥ 9, exercices > 80% | ~15.2 | ~95% |
| **C1** | 🟡 Étudiant moyen | Temps ~10h, devoirs ~7, exercices ~65% | ~11.0 | ~58% |
| **C2** | <img src=app/static/icons/circle_red.svg alt=circle_red width=18/> Étudiant à risque | Temps < 7h, absences > 6, exercices < 40% | ~6.5 | ~12% |

---

## 4. RAPPORT DE CLASSIFICATION (Exemple attendu)

```
=== RAPPORT DE CLASSIFICATION — Random Forest ===

              precision    recall  f1-score   support

     Échec       0.89      0.85      0.87        42
   Réussite      0.91      0.94      0.92        58

    accuracy                         0.90       100
   macro avg     0.90      0.89      0.90       100
weighted avg     0.90      0.90      0.90       100

Matrice de confusion :
             Prédit Échec   Prédit Réussite
Réel Échec       36              6
Réel Réussite     4             54

AUC-ROC : 0.956
```

---

*Document 6/10 — INF 232 EC2*
