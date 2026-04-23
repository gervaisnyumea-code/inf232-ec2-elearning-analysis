# ============================================================
# app/main.py
# INF 232 EC2 — Application Streamlit principale
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_cleaning import load_raw_data, full_pipeline, get_feature_matrix
from src.models import RegressionModel, ClassificationModel
from src.visualization import plot_histogram_kde, plot_pie_chart, plot_heatmap_correlation
from src.visualization_extra import plot_bar_chart_interactive, plot_boxplot_multi, plot_scatter_regression, plot_oscillating_progression

# SVG icon helper
import urllib.parse as _urllib_parse

def _svg_data_uri(name):
    p = Path(__file__).parent / 'static' / 'icons' / f'{name}.svg'
    if p.exists():
        svg = p.read_text()
        return f"data:image/svg+xml;utf8,{_urllib_parse.quote(svg)}"
    return ''


def icon_html(name, width=18):
    uri = _svg_data_uri(name)
    if uri:
        return f"<img src='{uri}' width='{width}' style='vertical-align:middle'/> "
    return ''

# Configuration de la page
st.set_page_config(
    page_title="INF232 - E-Learning Analysis",
    page_icon=None,
    layout="wide"
)


@st.cache_data
def load_data():
    """Charge les données."""
    try:
        df = pd.read_csv("data/processed/elearning_clean.csv")
    except:
        df = full_pipeline()
        X, _, _ = get_feature_matrix(df)
        # Recharger après processing
        df = pd.read_csv("data/processed/elearning_clean.csv")
    return df


@st.cache_resource
def load_models():
    """Charge les modèles entraînés. Supporte plusieurs formats de sauvegarde.

    Retourne
    -------
    (reg_model, clf_model)
    """
    import joblib
    import numpy as np

    reg_model = None
    clf_model = None

    # Charger modèle de régression (si présent)
    try:
        # Préférence : méthode de classe fournie par RegressionModel
        reg_model = RegressionModel.load("data/models/regression_model.pkl")
    except Exception:
        try:
            data = joblib.load("data/models/regression_model.pkl")
            # Si c'est un dict structuré, réutiliser la méthode de chargement
            if isinstance(data, dict) and 'model' in data:
                reg_model = RegressionModel.load("data/models/regression_model.pkl")
            else:
                reg_model = None
        except Exception:
            reg_model = None

    # Charger classifieur — plusieurs formats possibles
    try:
        # Si ClassificationModel définit une méthode load(), l'utiliser
        if hasattr(ClassificationModel, 'load'):
            clf_model = ClassificationModel.load("data/models/classifier_model.pkl")
        else:
            clf_raw = joblib.load("data/models/classifier_model.pkl")
            # Charger scaler séparé si disponible
            try:
                scaler = joblib.load("data/models/scaler.pkl")
            except Exception:
                scaler = None

            class _WrappedClassifier:
                def __init__(self, model, scaler=None):
                    self.model = model
                    self.scaler = scaler

                def _apply_scaler(self, X):
                    if self.scaler is None:
                        return X
                    try:
                        import pandas as _pd
                        # If scaler stored feature names, subset DataFrame accordingly
                        if hasattr(self.scaler, 'feature_names_in_') and isinstance(X, _pd.DataFrame):
                            cols = list(self.scaler.feature_names_in_)
                            # Keep only columns present
                            cols_present = [c for c in cols if c in X.columns]
                            X_sub = X[cols_present]
                            try:
                                return self.scaler.transform(X_sub)
                            except Exception:
                                return self.scaler.transform(X_sub.values)
                        # General fallback
                        try:
                            return self.scaler.transform(X)
                        except Exception:
                            try:
                                return self.scaler.transform(getattr(X, 'values', X))
                            except Exception:
                                return getattr(X, 'values', X)
                    except Exception:
                        try:
                            return self.scaler.transform(getattr(X, 'values', X))
                        except Exception:
                            return getattr(X, 'values', X)

                def predict(self, X):
                    Xs = self._apply_scaler(X)
                    return self.model.predict(Xs)

                def predict_proba(self, X):
                    Xs = self._apply_scaler(X)
                    if hasattr(self.model, 'predict_proba'):
                        proba = self.model.predict_proba(Xs)
                        try:
                            return proba[:, 1]
                        except Exception:
                            return proba
                    elif hasattr(self.model, 'decision_function'):
                        df = self.model.decision_function(Xs)
                        return 1 / (1 + np.exp(-df))
                    else:
                        return self.model.predict(Xs)

                def feature_importance(self, X=None):
                    if hasattr(self.model, 'feature_importances_'):
                        return self.model.feature_importances_
                    if hasattr(self.model, 'coef_'):
                        arr = np.abs(self.model.coef_)
                        if arr.ndim > 1:
                            arr = arr[0]
                        return arr
                    return None

            clf_model = _WrappedClassifier(clf_raw, scaler)
    except Exception:
        clf_model = None

    return reg_model, clf_model


# ============================================================
# SIDEBAR - Navigation
# ============================================================
st.sidebar.markdown(icon_html("chart",16) + " INF232 EC2", unsafe_allow_html=True)
st.sidebar.markdown("## Navigation")

page = st.sidebar.radio(
    "Aller à",
    ["Accueil", "Collecte", "EDA", "Modélisation", "Visualisation"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Projet:** Analyse de la performance académique")
st.sidebar.markdown("**Problématique:** Quels comportements influencent la réussite?")


# ============================================================
# PAGE D'ACCUEIL
# ============================================================
if page == "Accueil":
    st.markdown(icon_html("chart",24) + " Analyse de la Performance Académique en E-Learning", unsafe_allow_html=True)

    st.markdown("""
    ## Bienvenue dans l'application INF232 EC2

    Cette application permet d'analyser les comportements d'apprentissage
    des étudiants sur une plateforme e-learning et de prédire leur réussite académique.

    ### Fonctionnalités disponibles :

    - **Collecte** : Formulaire de saisie de données étudiants
    - **EDA** : Analyse exploratoire et statistiques descriptives
    - **Modélisation** : Régression linéaire et Classification
    - **Visualisation** : Réduction de dimension (ACP, t-SNE, LDA)

    ### Dataset :
    - 500 étudiants
    - 17 variables (comportementales, socio-démographiques, académiques)
    - Taux de réussite : ~62%
    """)

    st.markdown("---")
    st.info("Utilisez le menu latéral pour naviguer entre les pages")

    # Quick stats
    df = load_data()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Étudiants", len(df))
    col2.metric("Note Moyenne", f"{df['note_finale'].mean():.1f}/20")
    col3.metric("Taux de Réussite", f"{df['reussite'].mean()*100:.1f}%")
    col4.metric("Temps d'étude", f"{df['temps_etude_hebdo'].mean():.1f}h/sem")


# ============================================================
# PAGE COLLECTE
# ============================================================
elif page == "Collecte":
    st.markdown(icon_html("pin",20) + " Collecte de Données", unsafe_allow_html=True)

    st.markdown("### Saisissez les données d'un nouvel étudiant")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Âge", 17, 45, 20)
        genre = st.selectbox("Genre", ["M", "F"])
        niveau = st.selectbox("Niveau d'études", ["L1", "L2", "L3", "M1", "M2"])
        connections = st.number_input("Connexions/semaine", 1.0, 21.0, 8.0)
        study_time = st.number_input("Temps d'étude (h/sem)", 0.0, 35.0, 12.0)
        exercises = st.slider("% Exercices complétés", 0, 100, 75)

    with col2:
        videos = st.slider("% Vidéos vues", 0, 100, 70)
        forums = st.number_input("Messages forums", 0, 50, 5)
        homework = st.number_input("Devoirs rendus", 0, 10, 8)
        motivation = st.slider("Score motivation", 1.0, 10.0, 6.5)
        absences = st.number_input("Absences", 0, 20, 3)
        internet = st.selectbox("Accès internet", ["Stable", "Instable", "Limité"])
        revenus = st.selectbox("Revenu famille", ["Bas", "Moyen", "Élevé"])

    if st.button("Prédire la réussite", type="primary"):
        # Préparer les données pour la prédiction
        df = load_data()
        X, _, _ = get_feature_matrix(df)

        # Créer un dictionnaire avec les valeurs saisies
        encoding_map = {
            'niveau_etudes_num': {'L1': 1, 'L2': 2, 'L3': 3, 'M1': 4, 'M2': 5},
            'genre_num': {'M': 1, 'F': 0},
            'revenu_famille_num': {'Bas': 0, 'Moyen': 1, 'Élevé': 2},
            'acces_internet_num': {'Stable': 2, 'Instable': 1, 'Limité': 0}
        }

        # Construire le vecteur de features
        new_student = pd.DataFrame([[
            age,
            encoding_map['genre_num'][genre],
            encoding_map['niveau_etudes_num'][niveau],
            connections,
            study_time,
            exercises,
            videos,
            forums,
            homework,
            encoding_map['revenu_famille_num'][revenus],
            motivation,
            absences,
            encoding_map['acces_internet_num'][internet],
            10.0  # note_mi_parcours estimée
        ]], columns=X.columns)

        # Charger les modèles
        reg_model, clf_model = load_models()

        if reg_model and clf_model:
            # Prédiction
            note_predite = reg_model.predict(new_student)[0]
            reussite_proba = clf_model.predict_proba(new_student)[0]
            reussite_pred = clf_model.predict(new_student)[0]

            # Afficher les résultats
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Note finale prédite", f"{note_predite:.1f}/20")

            with col2:
                st.metric("Probabilité de réussite", f"{reussite_proba*100:.1f}%")

            if reussite_pred == 1:
                st.markdown(icon_html("check",20) + " **Réussite prédite!**", unsafe_allow_html=True)
            else:
                st.markdown(icon_html("warning",20) + " **Risque d'échec - Consulter les recommandations**", unsafe_allow_html=True)

            # Recommandations
            st.markdown(icon_html("search",16) + " ### Recommandations", unsafe_allow_html=True)
            if study_time < 10:
                st.write("- 🔹 Augmenter le temps d'étude hebdomadaire")
            if exercises < 70:
                st.write("- 🔹 Compléter plus d'exercices")
            if homework < 8:
                st.write("- 🔹 Rendre plus de devoirs")
            if absences > 5:
                st.write("- 🔹 Réduire les absences")


# ============================================================
# PAGE EDA
# ============================================================
elif page == "EDA":
    st.markdown(icon_html("chart",24) + " Analyse Exploratoire", unsafe_allow_html=True)

    df = load_data()

    # Statistiques globales
    st.markdown(icon_html("chart",16) + " ### Statistiques Descriptives", unsafe_allow_html=True)
    st.dataframe(df.describe().T, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(icon_html("pin",14) + " ### Distribution par Genre", unsafe_allow_html=True)
        genre_counts = df['genre'].value_counts()
        st.bar_chart(genre_counts)

    with col2:
        st.markdown(icon_html("chart",14) + " ### Distribution par Niveau", unsafe_allow_html=True)
        niveau_counts = df['niveau_etudes'].value_counts()
        st.bar_chart(niveau_counts)

    st.markdown(icon_html("target",14) + " ### Distribution des Notes", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("Note Mi-parcours")
        st.bar_chart(df['note_mi_parcours'].value_counts().sort_index())

    with col2:
        st.markdown("Note Finale")
        st.bar_chart(df['note_finale'].value_counts().sort_index())

    st.markdown("### 🔗 Corrélations avec la Réussite")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_with_success = df[numeric_cols].corr()['reussite'].sort_values(ascending=False)
    st.bar_chart(corr_with_success)

    # Live streaming quick demo (simulé)
    st.markdown(icon_html("loop",16) + " ### Flux de données en temps réel (simulation)", unsafe_allow_html=True)
    st.markdown("Utilisez le simulateur pour injecter des données continues et rafraîchissez le graphique.")
    col_a, col_b = st.columns([1,3])
    with col_a:
        if st.button("Démarrer simulateur de données"):
            import subprocess, sys, os
            os.makedirs('logs', exist_ok=True)
            with open('logs/streamer.log','a') as out, open('logs/streamer.err','a') as err:
                subprocess.Popen([sys.executable, 'scripts/data_stream_simulator.py'], stdout=out, stderr=err)
            st.success('Simulateur démarré (voir logs/streamer.log)')
        if st.button("Arrêter simulateur"):
            st.info('Arrêt manuel non implémenté; utilisez kill sur le processus (démo).')
        auto_refresh = st.checkbox("Auto-refresh (rafraîchir automatiquement)", value=False)
        refresh_interval = st.slider("Intervalle (s)", min_value=1, max_value=10, value=2)
        csv_limit = st.number_input("Points affichés", min_value=50, max_value=10000, value=500, step=50)
        if st.button("Exporter CSV live"):
            from src.data_streaming import read_live_data
            df_live = read_live_data(100000)
            csv_bytes = df_live.to_csv(index=False).encode('utf-8')
            st.download_button("Télécharger CSV live", csv_bytes, file_name='live_data.csv', mime='text/csv')
        if st.button("Générer rapport périodique"):
            from src.reporting import generate_report
            from src.data_streaming import read_live_data
            df_live = read_live_data(100000)
            rpt_path = generate_report(df_live, out_dir='reports')
            try:
                with open(rpt_path, 'rb') as f:
                    st.download_button("Télécharger rapport (ZIP)", f.read(), file_name=Path(rpt_path).name)
            except Exception:
                st.write('Erreur lors de la génération du rapport.')
    with col_b:
        from src.data_streaming import read_live_data
        df_live = read_live_data(csv_limit)
        if df_live.empty:
            st.write('Aucune donnée de streaming détectée. Cliquez sur "Démarrer simulateur de données".')
        else:
            import plotly.express as px
            fig = px.line(df_live, x='t', y=['value1','value2'], title='Courbes oscillantes (value1 & value2)')
            st.plotly_chart(fig, use_container_width=True)
            # annotate periodic points
            from src.reporting import find_periodic_points
            peaks = find_periodic_points(df_live, 't', 'value1')
            if not peaks.empty:
                st.markdown(icon_html('pin',12) + f" **Points périodiques détectés (value1):** {len(peaks)}", unsafe_allow_html=True)
                st.dataframe(peaks)
            # Auto refresh
            if auto_refresh:
                import time
                time.sleep(refresh_interval)
                try:
                    rerun = getattr(st, 'experimental_rerun', None)
                    if callable(rerun):
                        # Preferred method when available
                        rerun()
                    else:
                        # Fallback: tweak query params to force a rerun (works without experimental_rerun)
                        try:
                            params = st.experimental_get_query_params()
                            params['_autorefresh'] = str(time.time())
                            st.experimental_set_query_params(**params)
                        except Exception:
                            # If even query params are unavailable, skip silently
                            pass
                except Exception:
                    # Any unexpected error should not crash the app
                    pass



# ============================================================
# PAGE MODÉLISATION
# ============================================================
elif page == "Modélisation":
    st.title("🤖 Modélisation Prédictive")

    reg_model, clf_model = load_models()

    if reg_model and clf_model:
        st.markdown(icon_html("check",20) + " Modèles chargés avec succès!", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Régression", "Classification"])

        with tab1:
            st.markdown("### Régression Linéaire")
            st.markdown("Prédiction de la **note finale** en fonction des comportements d'apprentissage.")

            st.markdown("""
            **Métriques d'évaluation:**
            - R² : Coefficient de détermination
            - RMSE : Root Mean Square Error
            - MAE : Mean Absolute Error
            """)

            # Afficher les coefficients
            if hasattr(reg_model.model, 'coef_'):
                st.markdown("### Coefficients du modèle")
                coef_df = pd.DataFrame({
                    'Variable': reg_model.feature_names,
                    'Coefficient': reg_model.model.coef_
                }).sort_values('Coefficient', key=abs, ascending=False)
                st.dataframe(coef_df, use_container_width=True)

        with tab2:
            st.markdown("### Classification")
            st.markdown("Prédiction binaire de la **réussite** (0 = Échec, 1 = Succès).")

            st.markdown("""
            **Métriques d'évaluation:**
            - Accuracy : Précision globale
            - AUC : Area Under ROC Curve
            """)

            # Importance des features (si disponible)
            try:
                X_feat, _, _ = get_feature_matrix(load_data())
                importances = None
                if clf_model is not None:
                    if callable(getattr(clf_model, 'feature_importance', None)):
                        importances = clf_model.feature_importance(X_feat)
                    else:
                        importances = getattr(clf_model, 'feature_importance', None)

                if importances is not None:
                    try:
                        importance_df = pd.DataFrame({
                            'Variable': X_feat.columns,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)
                        st.markdown("### Importance des Variables")
                        st.dataframe(importance_df, use_container_width=True)
                    except Exception:
                        st.markdown(icon_html("warning",12) + " Impossible d'afficher l'importance des variables.", unsafe_allow_html=True)
            except Exception:
                st.markdown(icon_html("warning",12) + " Erreur lors du calcul de l'importance des variables.", unsafe_allow_html=True)
    else:
        st.markdown(icon_html("cross",20) + " <span style='color:#e74c3c;font-weight:bold'>Modèles non disponibles. Veuillez les entraîner d'abord.</span>", unsafe_allow_html=True)


# ============================================================
# PAGE VISUALISATION
# ============================================================
elif page == "Visualisation":
    st.markdown(icon_html("chart",24) + " Réduction de Dimension", unsafe_allow_html=True)

    st.markdown("""
    Cette page présente les techniques de réduction de dimension
    pour visualiser les données en 2D ou 3D.
    """)

    df = load_data()
    X, _, _ = get_feature_matrix(df)

    # Normaliser les données
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    technique = st.selectbox("Technique de réduction", ["ACP", "t-SNE", "LDA"])

    if technique == "ACP":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_scaled)

        st.markdown(f"### Analyse en Composantes Principales (ACP)")
        st.markdown(f"Variance expliquée par PC1: {pca.explained_variance_ratio_[0]:.2%}")
        st.markdown(f"Variance expliquée par PC2: {pca.explained_variance_ratio_[1]:.2%}")

        # Scatter plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=df['reussite'],
                            cmap='viridis', alpha=0.7, s=50)
        plt.colorbar(scatter, ax=ax, label='Réussite')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('ACP - Projection 2D')
        st.pyplot(fig)

    elif technique == "t-SNE":
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_reduced = tsne.fit_transform(X_scaled)

        st.markdown("### t-SNE (t-Distributed Stochastic Neighbor Embedding)")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=df['reussite'],
                            cmap='viridis', alpha=0.7, s=50)
        plt.colorbar(scatter, ax=ax, label='Réussite')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('t-SNE - Projection 2D')
        st.pyplot(fig)

    elif technique == "LDA":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(n_components=1)
        X_lda = lda.fit_transform(X_scaled, df['reussite'])

        st.markdown("### LDA (Linear Discriminant Analysis)")
        st.info("LDA est une méthode supervisée qui maximise la séparabilité des classes")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Streamlit apps are launched via the CLI: `streamlit run app/main.py`.
    # No action is required here; avoid calling non-existent st.run().
    pass