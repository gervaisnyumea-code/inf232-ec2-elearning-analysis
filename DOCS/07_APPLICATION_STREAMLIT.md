# INF 232 EC2 — APPLICATION STREAMLIT COMPLÈTE
## Code Source Commenté — Multi-pages

---

## 1. POINT D'ENTRÉE `app/main.py`

```python
# ============================================================
# app/main.py
# INF 232 EC2 — Application Streamlit
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import sys

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_cleaning import full_pipeline, get_feature_matrix
from src.visualization import (
    plot_histogram_kde, plot_pie_chart, plot_bar_chart_interactive,
    plot_boxplot_multi, plot_correlation_heatmap,
    plot_scatter_regression, plot_oscillating_progression
)

# ── Configuration globale ────────────────────────────────────
st.set_page_config(
    page_title="INF 232 — Analyse Performance E-Learning",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS personnalisé ─────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: bold;
        color: #1565C0; text-align: center; padding: 1rem 0;
    }
    .metric-card {
        background: #f0f4ff; border-radius: 10px;
        padding: 1rem; text-align: center;
        border-left: 4px solid #1565C0;
    }
    .success-badge {
        background: #e8f5e9; color: #2e7d32;
        padding: 4px 12px; border-radius: 20px; font-weight: bold;
    }
    .risk-badge {
        background: #fce4ec; color: #c62828;
        padding: 4px 12px; border-radius: 20px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ── Chargement des données (mis en cache) ────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    """Charge et met en cache le dataset nettoyé."""
    try:
        df = pd.read_csv("data/processed/elearning_clean.csv")
        return df
    except FileNotFoundError:
        st.warning("⚠️ Dataset non trouvé. Génération automatique en cours...")
        from src.data_generation import generate_student_dataset, save_dataset
        from src.data_cleaning import full_pipeline
        df_raw = generate_student_dataset(n_samples=500)
        save_dataset(df_raw)
        return full_pipeline()


@st.cache_resource
def load_models():
    """Charge les modèles pré-entraînés."""
    models = {}
    try:
        models['regression'] = joblib.load("data/models/regression_model.pkl")
        models['classifier'] = joblib.load("data/models/classifier_model.pkl")
        models['scaler']     = joblib.load("data/models/scaler.pkl")
    except FileNotFoundError:
        pass
    return models


# ── Interface principale ─────────────────────────────────────
def main():
    st.markdown('<div class="main-header">🎓 Analyse de Performance Académique<br>en E-Learning — INF 232 EC2</div>',
               unsafe_allow_html=True)
    
    # Navigation
    menu = st.sidebar.selectbox(
        "📌 Navigation",
        ["🏠 Accueil", "📥 Collecte de données",
         "📊 Analyse exploratoire", "🤖 Modélisation",
         "🔬 Réduction de dimension", "🔮 Prédiction"],
        index=0
    )
    
    df = load_data()
    models = load_models()
    
    # ── PAGES ────────────────────────────────────────────────
    if menu == "🏠 Accueil":
        page_accueil(df)
    elif menu == "📥 Collecte de données":
        page_collecte(df)
    elif menu == "📊 Analyse exploratoire":
        page_eda(df)
    elif menu == "🤖 Modélisation":
        page_modelisation(df, models)
    elif menu == "🔬 Réduction de dimension":
        page_reduction(df)
    elif menu == "🔮 Prédiction":
        page_prediction(models)


# ════════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL
# ════════════════════════════════════════════════════════════

def page_accueil(df: pd.DataFrame) -> None:
    """Page d'accueil : vue d'ensemble du dataset."""
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Étudiants", f"{len(df):,}")
    with col2:
        st.metric("✅ Taux de réussite", f"{df['reussite'].mean():.1%}")
    with col3:
        st.metric("📈 Note moyenne", f"{df['note_finale'].mean():.2f}/20")
    with col4:
        st.metric("⏱️ Temps étude moyen", f"{df['temps_etude_hebdo'].mean():.1f}h/sem")
    
    st.divider()
    
    # Aperçu des données
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📋 Aperçu du dataset")
        st.dataframe(
            df.head(10).style.highlight_max(subset=['note_finale'], color='#c8e6c9')
                              .highlight_min(subset=['note_finale'], color='#ffcdd2'),
            use_container_width=True
        )
    
    with col_right:
        st.subheader("📊 Statistiques clés")
        stats_display = df[['temps_etude_hebdo', 'note_finale', 
                            'exercices_completes_pct', 'nombre_absences']].describe()
        st.dataframe(stats_display.round(2), use_container_width=True)
    
    # Distribution des notes
    st.subheader("📈 Distribution des notes finales")
    fig = plot_histogram_kde(df, 'note_finale', 'Distribution des notes finales')
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════
# PAGE 2 — COLLECTE DE DONNÉES
# ════════════════════════════════════════════════════════════

def page_collecte(df: pd.DataFrame) -> None:
    """Formulaire de saisie d'une nouvelle observation."""
    
    st.header("📥 Collecte d'une nouvelle observation")
    st.info("Remplissez le formulaire ci-dessous pour enregistrer un étudiant.")
    
    col1, col2 = st.columns(2)
    
    with st.form("form_nouvel_etudiant", clear_on_submit=True):
        st.subheader("👤 Informations personnelles")
        
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Âge", min_value=17, max_value=45, value=22)
        genre = c2.selectbox("Genre", ['M', 'F'])
        niveau = c3.selectbox("Niveau", ['L1', 'L2', 'L3', 'M1', 'M2'])
        
        st.subheader("💻 Comportements d'apprentissage")
        
        c4, c5 = st.columns(2)
        connexions = c4.slider("Connexions/semaine", 1.0, 21.0, 8.0, 0.5)
        temps_etude = c5.slider("Temps d'étude hebdomadaire (h)", 0.0, 35.0, 12.0, 0.5)
        
        c6, c7 = st.columns(2)
        exercices = c6.slider("Exercices complétés (%)", 0.0, 100.0, 70.0, 1.0)
        videos = c7.slider("Vidéos vues (%)", 0.0, 100.0, 65.0, 1.0)
        
        c8, c9 = st.columns(2)
        forums = c8.number_input("Messages forums", 0, 50, 5)
        devoirs = c9.number_input("Devoirs rendus (sur 10)", 0, 10, 7)
        
        st.subheader("📊 Contexte")
        c10, c11, c12 = st.columns(3)
        revenu = c10.selectbox("Revenu famille", ['Bas', 'Moyen', 'Élevé'])
        motivation = c11.slider("Motivation (1-10)", 1.0, 10.0, 6.5, 0.1)
        absences = c12.number_input("Absences (séances)", 0, 20, 3)
        
        internet = st.selectbox("Accès internet", ['Stable', 'Instable', 'Limité'])
        note_mi = st.number_input("Note mi-parcours (/20)", 0.0, 20.0, 10.0, 0.5)
        
        submitted = st.form_submit_button("💾 Enregistrer l'observation", type="primary")
        
        if submitted:
            new_obs = pd.DataFrame([{
                'etudiant_id': len(df) + 1,
                'age': age, 'genre': genre, 'niveau_etudes': niveau,
                'nb_connexions_semaine': connexions,
                'temps_etude_hebdo': temps_etude,
                'exercices_completes_pct': exercices,
                'videos_vues_pct': videos,
                'participation_forums': forums,
                'nb_devoirs_rendus': devoirs,
                'revenu_famille': revenu,
                'score_motivation': motivation,
                'nombre_absences': absences,
                'acces_internet': internet,
                'note_mi_parcours': note_mi,
                'note_finale': None,
                'reussite': None
            }])
            
            # Sauvegarder
            output_path = "data/collected_data.csv"
            header = not Path(output_path).exists()
            new_obs.to_csv(output_path, mode='a', header=header, index=False)
            
            st.success(f"✅ Observation enregistrée ! (ID: {len(df) + 1})")
            st.dataframe(new_obs)


# ════════════════════════════════════════════════════════════
# PAGE 3 — ANALYSE EXPLORATOIRE
# ════════════════════════════════════════════════════════════

def page_eda(df: pd.DataFrame) -> None:
    """Dashboard d'analyse exploratoire."""
    st.header("📊 Analyse Exploratoire des Données (EDA)")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Distributions", "📊 Catégorielles",
        "🌡️ Corrélations", "📉 Évolution"
    ])
    
    with tab1:
        st.subheader("Distributions des variables numériques")
        num_var = st.selectbox(
            "Variable à analyser",
            ['note_finale', 'temps_etude_hebdo', 'nb_connexions_semaine',
             'exercices_completes_pct', 'score_motivation', 'nombre_absences']
        )
        col_a, col_b = st.columns(2)
        with col_a:
            fig = plot_histogram_kde(df, num_var)
            st.pyplot(fig)
            plt.close()
        with col_b:
            fig2, ax = plt.subplots(figsize=(6, 5))
            sns_import = __import__('seaborn')
            sns_import.boxplot(y=df[num_var], ax=ax, color='#2196F3')
            ax.set_title(f'Boxplot — {num_var}', fontweight='bold')
            st.pyplot(fig2)
            plt.close()
        
        # Statistiques
        st.write("**Statistiques descriptives :**")
        st.dataframe(df[[num_var]].describe().round(3).T)
    
    with tab2:
        st.subheader("Répartition des variables catégorielles")
        cat_col = st.selectbox(
            "Variable catégorielle",
            ['genre', 'niveau_etudes', 'revenu_famille', 'acces_internet', 'reussite']
        )
        col_c, col_d = st.columns(2)
        with col_c:
            fig3 = plot_pie_chart(df, cat_col)
            st.pyplot(fig3)
            plt.close()
        with col_d:
            fig4 = plot_bar_chart_interactive(df, cat_col)
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        st.subheader("Matrice de corrélation")
        fig5 = plot_correlation_heatmap(df)
        st.pyplot(fig5)
        plt.close()
        
        st.subheader("Nuage de points — Relation avec la note finale")
        x_var = st.selectbox(
            "Variable X",
            ['temps_etude_hebdo', 'nb_devoirs_rendus', 'exercices_completes_pct',
             'score_motivation', 'nombre_absences', 'nb_connexions_semaine']
        )
        fig6 = plot_scatter_regression(df, x_var, 'note_finale')
        st.pyplot(fig6)
        plt.close()
    
    with tab4:
        st.subheader("Évolution temporelle des connexions (simulation)")
        n_students = st.slider("Nombre d'étudiants à afficher", 2, 8, 5)
        fig7 = plot_oscillating_progression(df, n_students=n_students)
        st.plotly_chart(fig7, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 4 — MODÉLISATION
# ════════════════════════════════════════════════════════════

def page_modelisation(df: pd.DataFrame, models: dict) -> None:
    """Résultats des modèles entraînés."""
    st.header("🤖 Modélisation — Régression & Classification")
    
    tab1, tab2 = st.tabs(["📈 Régression Linéaire", "🎯 Classification"])
    
    with tab1:
        st.subheader("Régression Linéaire Multiple — Prédiction de la note finale")
        
        if 'regression' in models:
            reg = models['regression']
            st.info(f"**R² = {reg.metrics['R²']}** | RMSE = {reg.metrics['RMSE']} | MAE = {reg.metrics['MAE']}")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = reg.plot_regression_line()
                st.pyplot(fig); plt.close()
            with col2:
                fig2 = reg.plot_coefficients_bar()
                st.pyplot(fig2); plt.close()
            
            st.subheader("Diagnostics des hypothèses")
            col3, col4 = st.columns(2)
            with col3:
                fig3 = reg.plot_residuals_vs_fitted()
                st.pyplot(fig3); plt.close()
            with col4:
                fig4 = reg.plot_qq_plot()
                st.pyplot(fig4); plt.close()
            
            fig5 = reg.plot_learning_curve()
            st.pyplot(fig5); plt.close()
        else:
            st.warning("⚠️ Modèle non trouvé. Exécutez `python src/models.py` d'abord.")
    
    with tab2:
        st.subheader("Classification — Prédiction de la réussite")
        st.info("Random Forest — le meilleur classifieur selon ROC-AUC")
        
        if 'classifier' in models:
            clf = models['classifier']
            # Affichage des résultats stockés en cache
            st.success("Modèle chargé avec succès ✅")
        else:
            st.warning("⚠️ Modèle non trouvé. Exécutez `python src/classification.py` d'abord.")


# ════════════════════════════════════════════════════════════
# PAGE 5 — RÉDUCTION DE DIMENSION
# ════════════════════════════════════════════════════════════

def page_reduction(df: pd.DataFrame) -> None:
    """Visualisations ACP, t-SNE, LDA."""
    st.header("🔬 Réduction de Dimension — ACP · t-SNE · LDA")
    st.info("Chargement et calcul des réductions... (peut prendre quelques secondes)")
    
    from src.data_cleaning import get_feature_matrix
    from sklearn.preprocessing import StandardScaler
    from src.dimension_reduction import AnalyseACP, VisualisationTSNE, AnalyseLDA
    
    feature_cols = [
        'temps_etude_hebdo', 'nb_devoirs_rendus', 'exercices_completes_pct',
        'nb_connexions_semaine', 'score_motivation', 'nombre_absences'
    ]
    
    X, _, y_clf = get_feature_matrix(df)
    X_sub = X[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)
    
    tab1, tab2, tab3 = st.tabs(["📐 ACP", "🌌 t-SNE", "🎯 LDA"])
    
    with tab1:
        with st.spinner("Calcul ACP..."):
            acp = AnalyseACP(n_components=2)
            acp.fit_transform(X_scaled, feature_cols)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = acp.plot_explained_variance()
            st.pyplot(fig); plt.close()
        with col2:
            fig2 = acp.plot_biplot(
                labels=y_clf.values, label_type='class',
                title='Biplot ACP — Réussite vs Échec'
            )
            st.pyplot(fig2); plt.close()
    
    with tab2:
        with st.spinner("Calcul t-SNE (peut prendre ~30 secondes)..."):
            tsne = VisualisationTSNE(perplexity=30)
            tsne.fit_transform(X_scaled)
        
        fig3 = tsne.plot_tsne_scatter(y_clf.values, label_type='class')
        st.pyplot(fig3); plt.close()
        
        fig4 = tsne.plot_tsne_interactive(df, y_clf.values,
                                          ['note_finale', 'temps_etude_hebdo'])
        st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        with st.spinner("Calcul LDA..."):
            lda = AnalyseLDA()
            lda.fit_transform(X_scaled, y_clf.values, feature_cols)
        
        fig5 = lda.plot_lda_projection()
        st.pyplot(fig5); plt.close()


# ════════════════════════════════════════════════════════════
# PAGE 6 — PRÉDICTION INTERACTIVE
# ════════════════════════════════════════════════════════════

def page_prediction(models: dict) -> None:
    """Interface de prédiction pour un nouvel étudiant."""
    st.header("🔮 Prédiction — Profil d'un Étudiant")
    
    if 'regression' not in models or 'classifier' not in models:
        st.error("❌ Modèles non disponibles. Exécutez le pipeline d'entraînement.")
        return
    
    st.subheader("Entrez le profil de l'étudiant :")
    
    c1, c2, c3, c4 = st.columns(4)
    temps = c1.slider("Temps d'étude (h/sem)", 0.0, 35.0, 12.0, 0.5)
    devoirs = c2.slider("Devoirs rendus (/10)", 0, 10, 7)
    exercices = c3.slider("Exercices complétés (%)", 0.0, 100.0, 65.0)
    connexions = c4.slider("Connexions/sem", 1.0, 21.0, 8.0, 0.5)
    
    c5, c6, c7, c8 = st.columns(4)
    motivation = c5.slider("Motivation (/10)", 1.0, 10.0, 6.5, 0.1)
    absences = c6.number_input("Absences", 0, 20, 2)
    videos = c7.slider("Vidéos vues (%)", 0.0, 100.0, 60.0)
    note_mi = c8.number_input("Note mi-parcours (/20)", 0.0, 20.0, 10.5, 0.5)
    
    if st.button("🔮 Prédire", type="primary"):
        input_data = np.array([[
            temps, devoirs, exercices, connexions,
            motivation, absences, videos, note_mi
        ]])
        
        reg = models['regression']
        clf = models['classifier']
        scaler = models['scaler']
        
        input_scaled = scaler.transform(input_data)
        
        note_pred = reg.model.predict(input_data)[0]
        reussite_pred = clf.predict(input_scaled)[0]
        reussite_proba = clf.predict_proba(input_scaled)[0][1]
        
        note_pred = np.clip(note_pred, 0, 20)
        
        st.divider()
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.metric("📊 Note Finale Prédite", f"{note_pred:.2f} / 20")
        with col_res2:
            badge = "✅ RÉUSSITE" if reussite_pred == 1 else "❌ ÉCHEC"
            st.metric("🎯 Prédiction Réussite", badge)
        with col_res3:
            st.metric("📈 Probabilité de réussite", f"{reussite_proba:.1%}")
        
        # Jauge
        st.progress(min(int(note_pred / 20 * 100), 100))
        
        # Recommandation
        if reussite_pred == 0:
            st.warning(
                "⚠️ **Étudiant à risque d'échec.** Recommandations :\n"
                "- Augmenter le temps d'étude hebdomadaire\n"
                "- Réduire les absences\n"
                "- Compléter plus d'exercices pratiques"
            )
        else:
            st.success("🎉 **Profil favorable à la réussite !**")


if __name__ == "__main__":
    main()
```

---

## 2. COMMANDES DE LANCEMENT

```bash
# Générer les données + entraîner les modèles
python src/data_generation.py
python src/data_cleaning.py
python src/models.py
python src/classification.py

# Lancer l'application
streamlit run app/main.py

# Accéder à l'application
# → http://localhost:8501
```

---

## 3. STRUCTURE DES PAGES DE L'APPLICATION

```
┌─────────────────────────────────────────────────────────┐
│              SIDEBAR NAVIGATION                         │
│  🏠 Accueil        ← KPIs + aperçu dataset              │
│  📥 Collecte       ← Formulaire saisie + SQLite/CSV     │
│  📊 EDA            ← Histos, Pie, Barres, Heatmap, Osc. │
│  🤖 Modélisation   ← Régression + Classification        │
│  🔬 Réduction      ← ACP + t-SNE + LDA                  │
│  🔮 Prédiction     ← Interface de prédiction temps réel  │
└─────────────────────────────────────────────────────────┘
```

---

*Document 8/10 — INF 232 EC2*
