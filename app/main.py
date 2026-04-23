# ============================================================
# app/main.py
# INF 232 EC2 — Application Streamlit principale
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time
import json
from pathlib import Path

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent))
# Charger variables depuis .env (persistées) pour que les scripts et Streamlit héritent des mêmes valeurs
try:
    from src.env_loader import load_dotenv, persist_env
    load_dotenv()
except Exception:
    pass

from src.data_cleaning import load_raw_data, full_pipeline, get_feature_matrix
from src.models import RegressionModel, ClassificationModel
from src.orchestration import BrainNet
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
    ["Accueil", "Collecte", "EDA", "Modélisation", "Visualisation", "LLM BrainNet"]
)

st.sidebar.markdown("---")

# Contrôle d'auto-refresh global (injecte un petit JS pour recharger la page)
autoref = st.sidebar.checkbox("Auto-refresh global (rafraîchir toutes les pages)", value=(os.getenv('GLOBAL_AUTORELOAD','false').lower() in ('1','true','yes')), key='global_autorefresh')
if autoref:
    interval = st.sidebar.number_input("Intervalle auto-refresh (s)", min_value=1, max_value=3600, value=int(os.getenv('GLOBAL_AUTORELOAD_INTERVAL','5')))
    st.html(f"<script>setTimeout(()=>location.reload(), {int(interval)*1000});</script>")

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
            note_predite = float(reg_model.predict(new_student)[0])
            # robust handling of predict_proba output
            try:
                proba_arr = clf_model.predict_proba(new_student)
                # clf_model.predict_proba returns array-like (n,) or (n,2)
                if hasattr(proba_arr[0], '__len__') and len(proba_arr.shape) > 1:
                    positive_proba = float(proba_arr[0][1])
                else:
                    positive_proba = float(proba_arr[0])
            except Exception:
                positive_proba = 0.0
            reussite_pred = int(clf_model.predict(new_student)[0])

            # Afficher les résultats
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Note finale prédite", f"{note_predite:.1f}/20")

            with col2:
                st.metric("Probabilité de réussite", f"{positive_proba*100:.1f}%")

            if reussite_pred == 1:
                st.markdown(icon_html("check",20) + " **Réussite prédite!**", unsafe_allow_html=True)
            else:
                st.markdown(icon_html("warning",20) + " **Risque d'échec - Consulter les recommandations**", unsafe_allow_html=True)

            # Recommandations statiques
            st.markdown("---")
            st.markdown(icon_html("search",16) + " <h4 style='display:inline;vertical-align:middle;margin:0'>Recommandations</h4>", unsafe_allow_html=True)
            if study_time < 10:
                st.markdown(icon_html('target',12) + " Augmenter le temps d'étude hebdomadaire", unsafe_allow_html=True)
            if exercises < 70:
                st.markdown(icon_html('target',12) + " Compléter plus d'exercices", unsafe_allow_html=True)
            if homework < 8:
                st.markdown(icon_html('target',12) + " Rendre plus de devoirs", unsafe_allow_html=True)
            if absences > 5:
                st.markdown(icon_html('target',12) + " Réduire les absences", unsafe_allow_html=True)

            # Option IA: demander recommandations personnalisées via BrainNet
            use_llm_rec = st.checkbox('Demander recommandation IA personnalisée pour cet étudiant', value=False, key='collect_llm')
            if use_llm_rec:
                col_r1, col_r2 = st.columns([1, 2])
                with col_r1:
                    rounds_llm = st.number_input('Rounds LLM', min_value=1, max_value=3, value=2, key='collect_llm_rounds')
                with col_r2:
                    generate_rec = st.button('Générer les recommandations IA', key='generate_llm_btn')
                
                if generate_rec:
                    st.info('🤖 Consultation IA en cours...')
                    try:
                        from src.llm_orchestrator import LLMOrchestrator, get_conversation_memory
                        from src.env_loader import load_dotenv
                        load_dotenv()
                        
                        orch = LLMOrchestrator()
                        mem = get_conversation_memory()
                    except Exception as e:
                        orch = None
                        st.error(f'Orchestrateur LLM non disponible: {e}')

                    if orch:
                        profile = { 'age': age, 'genre': genre, 'niveau': niveau, 
                                   'connections': connections, 'study_time': study_time,
                                   'exercises': exercises, 'videos': videos, 'forums': forums, 
                                   'homework': homework, 'revenus': revenus, 'motivation': motivation, 
                                   'absences': absences, 'internet': internet }
                        try:
                            df_summary = load_data()
                            dsum_lines = [f"Données: {len(df_summary)} étudiants"]
                            numc = df_summary.select_dtypes(include=['number']).columns.tolist()
                            for c in numc[:6]:
                                s = df_summary[c].describe()
                                dsum_lines.append(f"{c}: mean={s['mean']:.2f}, std={s['std']:.2f}")
                            data_summary_text = "\n".join(dsum_lines)
                        except Exception:
                            data_summary_text = f"Profil étudiant: {profile}"

                        question = f"Profil étudiant: {profile}\nContexte: {data_summary_text}\nDonne 5 recommandations pédagogiques priorisées, actionnables et détaillées en français."

                        try:
                            res = orch.concert_and_merge(
                                question, 
                                rounds=rounds_llm,
                                include_data=True,
                                data_window_sec=40000,
                                max_tokens=1024,
                                force_real=True,
                                use_memory=True,
                                cache_data=True
                            )
                            
                            # Afficher les contributions repliables avec un EXPANDER
                            st.markdown("---")
                            st.markdown("### 🤖 Réflexions des IA (cliquez pour développer)")
                            
                            # Créer un expandeur pour chaque tour
                            for tour in range(rounds_llm + 1):
                                contributions_tour = [c for c in res.get('rounds', []) if c['round'] == tour]
                                if contributions_tour:
                                    exp_label = f"📊 Tour {tour} - Contributions des {len(contributions_tour)} IA"
                                    with st.expander(exp_label, expanded=False):
                                        for c in contributions_tour:
                                            provider = c.get('provider', 'unknown')
                                            text = c.get('text', '')
                                            st.markdown(f"**{provider.upper()}:**")
                                            st.markdown(text)
                                            st.markdown("---")
                            
                            # La réponse FUSIONNÉE toujours visible (pas dans un expander)
                            st.markdown("---")
                            st.markdown("### ✅ Recommandations Fusionnées")
                            st.markdown(res.get('merged', 'Aucune recommandation'))
                            
                            # Sauvegarder
                            out_dir = Path(os.getenv('EXPORT_DIR', 'reports'))
                            out_dir.mkdir(parents=True, exist_ok=True)
                            ts = int(time.time())
                            out_path = out_dir / f'llm_reco_collect_{ts}.json'
                            out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2))
                            st.success(f'Recommandations sauvegardées -> {out_path}')
                            
                        except Exception as e:
                            st.error(f'Erreur lors de la concertation IA: {e}')


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

    st.markdown(icon_html("chart",16) + " Corrélations avec la Réussite", unsafe_allow_html=True)
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
        stream_auto_refresh = st.checkbox("Auto-refresh (rafraîchir automatiquement)", value=False, key='stream_auto_refresh')
        refresh_interval = st.slider("Intervalle (s)", min_value=1, max_value=10, value=2, key='stream_refresh_interval')
        csv_limit = st.number_input("Points affichés", min_value=50, max_value=10000, value=500, step=50, key='stream_csv_limit')
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
        if st.button("Exécuter ensemble sur live + rapport"):
            from src.orchestration import BrainNet
            from src.data_streaming import read_live_data
            from src.reporting import generate_report
            import time
            bn = BrainNet(auto_load=True)
            df_live = read_live_data(100000)
            if df_live.empty:
                st.write("Aucune donnée de streaming pour l'ensemble.")
            else:
                ensembled_csv = bn.save_ensemble_predictions(df_live, out_csv=f'reports/ensemble_live_{int(time.time())}.csv')
                rpt = generate_report(df_live, out_dir='reports')
                st.success("Ensemble et rapport générés.")
                try:
                    with open(ensembled_csv, 'rb') as fe:
                        st.download_button("Télécharger preds (CSV)", fe.read(), file_name=Path(ensembled_csv).name)
                except Exception:
                    pass
                try:
                    with open(rpt, 'rb') as fr:
                        st.download_button("Télécharger rapport (ZIP)", fr.read(), file_name=Path(rpt).name)
                except Exception:
                    pass
        if st.button("Créer un résumé LLM du snapshot"):
            from src.llm_integration import LLMClient
            from src.data_streaming import read_live_data
            from src.reporting import find_periodic_points
            df_live = read_live_data(100000)
            if df_live.empty:
                st.write("Aucune donnée live disponible pour synthèse.")
            else:
                peaks = find_periodic_points(df_live, 't', 'value1')
                summary_text = f"Snapshot rows={len(df_live)}, mean(value1)={df_live['value1'].mean():.4f}, peaks={len(peaks)}"
                client = LLMClient()
                llm_summary = client.summarize(summary_text)
                st.markdown("### Résumé LLM")
                st.write(llm_summary)
    with col_b:
        from src.data_streaming import read_live_data
        df_live = read_live_data(csv_limit)
        if df_live.empty:
            st.write('Aucune donnée de streaming détectée. Cliquez sur "Démarrer simulateur de données".')
        else:
            import plotly.express as px
            # Assurer que 't' est numérique et tri chronologique
            if 't' in df_live.columns:
                df_live['t'] = pd.to_numeric(df_live['t'], errors='coerce')
                df_live = df_live.sort_values('t').reset_index(drop=True)

            fig = px.line(df_live, x='t', y=['value1','value2'], title='Courbes oscillantes (value1 & value2)')
            st.plotly_chart(fig, width='stretch')
            # annotate periodic points
            from src.reporting import find_periodic_points
            peaks = find_periodic_points(df_live, 't', 'value1')
            if not peaks.empty:
                st.markdown(icon_html('pin',12) + f" **Points périodiques détectés (value1):** {len(peaks)}", unsafe_allow_html=True)
                st.dataframe(peaks)
            # Auto refresh
            if stream_auto_refresh:
                import time
                time.sleep(refresh_interval)
                try:
                    rerun = getattr(st, 'experimental_rerun', None)
                    if callable(rerun):
                        rerun()
                    else:
                        try:
                            params = st.experimental_get_query_params()
                            params['_autorefresh'] = str(time.time())
                            st.experimental_set_query_params(**params)
                        except Exception:
                            pass
                except Exception:
                    pass



# ============================================================
# PAGE MODÉLISATION
# ============================================================
elif page == "Modélisation":
    st.markdown(icon_html("brain",24) + " Modélisation Prédictive", unsafe_allow_html=True)

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

                # Diagnostics supplémentaires pour la régression
                try:
                    df_all = load_data()
                    X_all, y_reg_all, _ = get_feature_matrix(df_all)
                    y_pred_all = reg_model.predict(X_all)
                    import plotly.express as px
                    fig_pred = px.scatter(x=y_reg_all, y=y_pred_all, labels={'x':'Vrai','y':'Prévu'}, title='Prédits vs Réels')
                    fig_pred.add_shape(type='line', x0=min(y_reg_all), y0=min(y_reg_all), x1=max(y_reg_all), y1=max(y_reg_all), line=dict(color='red', dash='dash'))
                    st.plotly_chart(fig_pred, use_container_width=True)
                    residuals = y_reg_all - y_pred_all
                    fig_res = px.histogram(residuals, nbins=50, title='Distribution des résidus')
                    st.plotly_chart(fig_res, use_container_width=True)
                    fig_res_vs_pred = px.scatter(x=y_pred_all, y=residuals, labels={'x':'Prévu','y':'Résidu'}, title='Résidus vs Prévu')
                    fig_res_vs_pred.add_shape(type='line', x0=min(y_pred_all), x1=max(y_pred_all), y0=0, y1=0, line=dict(color='red', dash='dash'))
                    st.plotly_chart(fig_res_vs_pred, use_container_width=True)
                except Exception as e:
                    st.info('Diagnostics de régression non disponibles: ' + str(e))

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

            # Diagnostics classification (ROC, Confusion, PR, calibration)
            try:
                from src.visualization_extra import plot_classification_diagnostics
                try:
                    df_all = load_data()
                    X_all, _, y_all = get_feature_matrix(df_all)
                    figs = plot_classification_diagnostics(clf_model, X_all, y_all)
                    if figs is None:
                        st.info('Diagnostics classification non disponibles.')
                    else:
                        if isinstance(figs, list):
                            for f in figs:
                                if f is not None:
                                    st.plotly_chart(f, use_container_width=True)
                        else:
                            st.plotly_chart(figs, use_container_width=True)
                except Exception as e:
                    st.info('Diagnostics classification non disponibles: ' + str(e))
            except Exception:
                pass
            # Orchestrateur BrainNet — configuration des poids
            try:
                bn = BrainNet(auto_load=True)
                model_names = bn.get_model_names()
                st.markdown("### Orchestrateur BrainNet")
                st.write("Modèles détectés:", model_names)
                current_weights = getattr(bn, 'weights', None) or {}

                with st.expander("Configurer les poids (weights) des modèles"):
                    with st.form("weights_form"):
                        weight_inputs = {}
                        for name in model_names:
                            default_w = float(current_weights.get(name, 1.0))
                            # use a stable key per model name
                            weight_inputs[name] = st.number_input(f"Poids - {name}", min_value=0.0, max_value=10.0, value=default_w, step=0.1, key=f"w_{name}")
                        submitted = st.form_submit_button("Enregistrer les poids")
                        if submitted:
                            # persist weights
                            try:
                                weights_dict = {k: float(v) for k, v in weight_inputs.items()}
                                bn.set_weights(weights_dict, persist=True)
                                st.success("Poids enregistrés")
                            except Exception:
                                st.error("Erreur lors de l'enregistrement des poids")

                # Aperçu des prédictions avec les poids appliqués
                try:
                    sample = X_feat.sample(min(5, len(X_feat)))
                    preds, probs = bn.ensemble_predict_classification(sample)
                    st.markdown("### Exemple de prédictions de l'ensemble (5 observations)")
                    df_preds = sample.reset_index(drop=True).copy()
                    if preds is not None:
                        df_preds['ensemble_pred'] = list(preds)
                    if probs is not None:
                        df_preds['ensemble_proba'] = list(probs)
                    st.dataframe(df_preds, use_container_width=True)
                except Exception:
                    st.write("Impossible d'exécuter l'ensemble d'exemple.")

            except Exception as e:
                st.write("BrainNet non initialisé:", e)
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
    # GRAPHIQUES SUPPLÉMENTAIRES
    # ============================================================
    st.markdown(icon_html('chart',24) + ' Tous les Graphiques', unsafe_allow_html=True)
    st.markdown('Visualisation multiple : camembert, barres, scatter, boxplot, heatmap, histogram, density')
    import plotly.express as px

    try:
        df_viz = load_data()
    except Exception:
        df_viz = df

    cat_cols = df_viz.select_dtypes(include=['object','category']).columns.tolist()
    num_cols = df_viz.select_dtypes(include=[np.number]).columns.tolist()

    chart_type = st.selectbox('Choisir un type de graphique', ['Camembert (Pie)','Barres','Scatter','Boxplot','Correlation Heatmap','Histogram','Density','Oscillating (Simulation)'])

    if chart_type == 'Camembert (Pie)':
        sel_col = st.selectbox('Variable catégorielle', cat_cols if cat_cols else ['niveau_etudes'])
        counts = df_viz[sel_col].value_counts().reset_index()
        counts.columns = [sel_col,'count']
        fig = px.pie(counts, names=sel_col, values='count', title=f'Camembert - {sel_col}')
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == 'Barres':
        sel_cat = st.selectbox('Catégorie (axe X)', cat_cols if cat_cols else ['niveau_etudes'])
        sel_val = st.selectbox('Valeur (axe Y)', num_cols if num_cols else [num_cols[0]])
        agg_df = df_viz.groupby(sel_cat)[sel_val].mean().reset_index()
        fig = px.bar(agg_df, x=sel_cat, y=sel_val, title=f'Barres: moyenne {sel_val} par {sel_cat}')
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == 'Scatter':
        if len(num_cols) >= 2:
            xcol = st.selectbox('X', num_cols, index=0)
            ycol = st.selectbox('Y', num_cols, index=1)
            color = st.selectbox('Color by (optionnel)', [None] + cat_cols)
            fig = px.scatter(df_viz, x=xcol, y=ycol, color=color, title=f'Scatter: {xcol} vs {ycol}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Pas assez de colonnes numériques pour un scatter.')

    elif chart_type == 'Boxplot':
        if num_cols and cat_cols:
            val = st.selectbox('Valeur numérique', num_cols)
            by = st.selectbox('Group by', cat_cols)
            fig = px.box(df_viz, x=by, y=val, title=f'Boxplot: {val} groupé par {by}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Boxplot nécessite au moins une colonne numérique et une catégorielle.')

    elif chart_type == 'Correlation Heatmap':
        if len(num_cols) >= 2:
            corr = df_viz[num_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect='auto', title='Matrice de corrélation')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Pas assez de colonnes numériques pour une matrice de corrélation.')

    elif chart_type == 'Histogram':
        if num_cols:
            colh = st.selectbox('Variable numérique', num_cols)
            bins = st.slider('Bins', 5, 200, 30)
            fig = px.histogram(df_viz, x=colh, nbins=bins, title=f'Histogramme {colh}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Aucune colonne numérique trouvée.')

    elif chart_type == 'Density':
        if num_cols:
            cold = st.selectbox('Variable numérique pour densité', num_cols)
            fig = px.histogram(df_viz, x=cold, nbins=100, histnorm='density', title=f'Densité {cold}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Aucune colonne numérique trouvée.')

    elif chart_type == 'Oscillating (Simulation)':
        n_students = st.slider('Nombre de courbes', 1, 20, 5)
        col_choice = st.selectbox('Colonne de base (optionnel)', [None] + num_cols)
        col_to_use = col_choice if col_choice else 'nb_connexions_semaine'
        fig = plot_oscillating_progression(df_viz, col=col_to_use, n_students=n_students)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')

# ============================================================
# PAGE LLM BRAINNET
# ============================================================
elif page == "LLM BrainNet":
    st.markdown(icon_html("brain",24) + " LLM BrainNet", unsafe_allow_html=True)
    st.markdown("Console centralisée pour orchestrer et questionner les LLM. Configurez la consommation ci-dessous.")

    col_a, col_b = st.columns([2,3])
    with col_a:
        enable = st.checkbox("Activer appels LLM (globaux)", value=(os.getenv('LLM_CALLS_ENABLED','false').lower() in ('1','true','yes')))
        max_calls = st.number_input("Max appels / heure", min_value=1, max_value=10000, value=int(os.getenv('LLM_MAX_CALLS_PER_HOUR','60')))
        mistral_cost = st.number_input("Coût Mistral (USD / 1k tokens)", min_value=0.0, max_value=100.0, value=float(os.getenv('MISTRAL_COST_PER_1K','0.002')), format="%.6f")
        gemini_cost = st.number_input("Coût Gemini (USD / 1k tokens)", min_value=0.0, max_value=100.0, value=float(os.getenv('GEMINI_COST_PER_1K','0.03')), format="%.6f")
        groq_cost = st.number_input("Coût Groq (USD / 1k tokens)", min_value=0.0, max_value=100.0, value=float(os.getenv('GROQ_COST_PER_1K','0.02')), format="%.6f")
        if st.button("Appliquer paramètres LLM"):
            os.environ['LLM_CALLS_ENABLED'] = 'true' if enable else 'false'
            os.environ['LLM_MAX_CALLS_PER_HOUR'] = str(max_calls)
            os.environ['MISTRAL_COST_PER_1K'] = str(mistral_cost)
            os.environ['GEMINI_COST_PER_1K'] = str(gemini_cost)
            os.environ['GROQ_COST_PER_1K'] = str(groq_cost)
            # Persist to .env using helper
            try:
                from src.env_loader import persist_env
                success = persist_env({
                    'LLM_CALLS_ENABLED': 'true' if enable else 'false',
                    'LLM_MAX_CALLS_PER_HOUR': str(max_calls),
                    'MISTRAL_COST_PER_1K': str(mistral_cost),
                    'GEMINI_COST_PER_1K': str(gemini_cost),
                    'GROQ_COST_PER_1K': str(groq_cost)
                }, path=Path(__file__).parent.parent / '.env')
            except Exception as e:
                st.error(f"Impossible d'écrire .env: {e}")
                success = False

            if success:
                st.success("Paramètres LLM appliqués et sauvegardés dans .env.")
            else:
                st.warning("Paramètres appliqués en mémoire mais échec de la sauvegarde sur disque.")

    # Charger et exécuter la console (module importable)
    try:
        import importlib
        llm_mod = None
        try:
            llm_mod = importlib.import_module('app.llm_console')
        except Exception:
            from importlib.util import spec_from_file_location, module_from_spec
            spec = spec_from_file_location('llm_console', str(Path(__file__).parent / 'llm_console.py'))
            llm_mod = module_from_spec(spec)
            spec.loader.exec_module(llm_mod)
    except Exception as e:
        st.error(f"Impossible de charger le module LLM console: {e}")
        llm_mod = None

    if llm_mod and hasattr(llm_mod, 'render_llm_console'):
        llm_mod.render_llm_console()
    else:
        st.info("LLM console non disponible. Exécutez app/llm_console.py séparément ou vérifiez app/llm_console.py")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Streamlit apps are launched via the CLI: `streamlit run app/main.py`.
    # No action is required here; avoid calling non-existent st.run().
    pass