import os
import time
import json
from pathlib import Path

import streamlit as st
import pandas as pd
from urllib.request import Request, urlopen

# Charger .env tôt pour que les clés et quotas soient disponibles
try:
    from src.env_loader import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.llm_integration import LLMClient
from src.llm_orchestrator import LLMOrchestrator


def send_webhook(url: str, payload: dict, timeout: int = 10) -> dict:
    try:
        req = Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode('utf-8')
            try:
                return {'ok': True, 'resp': json.loads(raw)}
            except Exception:
                return {'ok': True, 'resp': raw}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def check_and_send_alert(threshold: float, webhook_url: str) -> dict:
    usage_path = Path(os.getenv('LLM_USAGE_FILE', 'logs/llm_usage.json'))
    try:
        entries = json.loads(usage_path.read_text()) if usage_path.exists() else []
    except Exception:
        entries = []
    now = int(time.time())
    last_hour = [e for e in entries if e.get('ts') and now - int(e.get('ts')) <= 3600]
    cost_last_hour = sum(float(e.get('cost', 0.0)) for e in last_hour)
    if cost_last_hour >= float(threshold):
        payload = {'alert': 'LLM hourly cost exceeded', 'threshold': float(threshold), 'cost_last_hour': cost_last_hour, 'count_last_hour': len(last_hour)}
        res = send_webhook(webhook_url, payload)
        alert_log = Path('logs/llm_alerts.log')
        try:
            alert_log.parent.mkdir(parents=True, exist_ok=True)
            prev = alert_log.read_text() if alert_log.exists() else ''
            alert_log.write_text(prev + f"\n{int(time.time())} ALERT: {payload} -> {res}\n")
        except Exception:
            pass
        return {'alert_sent': True, 'cost_last_hour': cost_last_hour, 'webhook_resp': res}
    return {'alert_sent': False, 'cost_last_hour': cost_last_hour}


def render_llm_console():
    """Render the LLM BrainNet console inside a larger Streamlit app.

    This function is import-safe (does not call set_page_config). Use the
    standalone runner (python app/llm_console.py) to run as a separate app.
    """
    st.title('LLM BrainNet — Console pour concertation des LLM')

    orch = LLMOrchestrator()
    providers = orch.available_providers()

    st.sidebar.header('Configuration')
    st.sidebar.write('Providers détectés: ' + (', '.join(providers) if providers else 'aucun'))
    rounds = st.sidebar.slider('Nombre de rounds de concertation', 0, 3, 1)
    use_real = st.sidebar.checkbox('Activer appels réels (respecter quotas)', value=(os.getenv('LLM_CALLS_ENABLED', 'false').lower() in ('1','true','yes')))

    # Usage summary (tokens / coût estimé)
    usage_path = Path(os.getenv('LLM_USAGE_FILE', 'logs/llm_usage.json'))
    try:
        if usage_path.exists():
            usage = json.loads(usage_path.read_text())
        else:
            usage = []
    except Exception:
        usage = []

    total_calls = len(usage)
    total_cost = sum(float(item.get('cost', 0.0)) for item in usage)
    calls_by_provider = {}
    for item in usage:
        p = item.get('provider', 'unknown')
        calls_by_provider[p] = calls_by_provider.get(p, 0) + 1

    st.sidebar.markdown('---')
    st.sidebar.markdown('### Usage LLM')
    st.sidebar.metric('Appels totaux', total_calls)
    st.sidebar.metric('Coût estimé', f"{total_cost:.4f} USD")
    for p, cnt in calls_by_provider.items():
        st.sidebar.write(f"- {p}: {cnt} appels")

    if st.sidebar.button('Effacer logs usage'):
        try:
            usage_path.write_text('[]')
            st.sidebar.success('Logs usage effacés')
        except Exception:
            st.sidebar.error('Impossible d\'effacer les logs')

    question = st.text_area('Question à adresser aux LLM', height=140)

    if 'convo' not in st.session_state:
        st.session_state['convo'] = []

    col1, col2 = st.columns([3,1])
    with col2:
        if st.button('Démarrer la concertation'):
            if not question.strip():
                st.warning('Entrez une question.')
            elif not providers:
                st.warning('Aucun provider détecté. Vérifiez .env ou exécutez en mode simulation.')
            else:
                st.session_state['convo'] = []
                # initial answers
                for p in providers:
                    client = LLMClient(p)
                    # respect manual toggle: if user disabled real calls override client.enabled
                    client.enabled = use_real
                    with st.spinner(f"{p} réfléchit…"):
                        try:
                            ans = client.summarize(question, max_tokens=256)
                        except Exception as e:
                            ans = f'[Erreur appel LLM {p}: {e}]'
                    st.session_state['convo'].append({'round': 0, 'provider': p, 'text': ans, 'ts': int(time.time())})
                    st.write(f'**{p}**: {ans}')

                # debate rounds: each provider responds to others
                for r in range(1, rounds + 1):
                    st.write(f'--- Round {r} de concertation ---')
                    last_texts = {c['provider']: c['text'] for c in st.session_state['convo'] if c['round'] == r-1}
                    for p in providers:
                        client = LLMClient(p)
                        client.enabled = use_real
                        # compile context: question + other providers' last messages
                        others = '\n'.join([f'{other}: {txt}' for other, txt in last_texts.items() if other != p])
                        prompt = f"Question: {question}\nContributions: {others}\nRépond en tenant compte des autres points de vue."
                        with st.spinner(f"{p} réfléchit au round {r}…"):
                            try:
                                ans = client.summarize(prompt, max_tokens=256)
                            except Exception as e:
                                ans = f'[Erreur appel LLM {p}: {e}]'
                        st.session_state['convo'].append({'round': r, 'provider': p, 'text': ans, 'ts': int(time.time())})
                        st.write(f'**{p} (round {r})**: {ans}')

                # reconciliation / final answer using GROQ or fallback
                st.write('--- Réconciliation finale ---')
                ensemble_meta = {'conversation': st.session_state['convo'], 'question': question}
                recon = orch.reconcile_ensemble(ensemble_meta)
                st.write('Réconciliation (par provider):')
                st.json(recon)

                # save conversation
                out_dir = Path(os.getenv('EXPORT_DIR', 'reports'))
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = int(time.time())
                out_path = out_dir / f'llm_conversation_{ts}.json'
                out_path.write_text(json.dumps({'question': question, 'conversation': st.session_state['convo'], 'reconciliation': recon}, indent=2, ensure_ascii=False))
                st.success(f'Conversation sauvegardée -> {out_path}')

                # Vérification automatique des alertes (si configuré)
                try:
                    threshold_env = float(os.getenv('LLM_ALERT_HOURLY_THRESHOLD', '0'))
                except Exception:
                    threshold_env = 0.0
                webhook_env = os.getenv('LLM_ALERT_WEBHOOK_URL', '').strip()
                if threshold_env and webhook_env:
                    al_res = check_and_send_alert(threshold_env, webhook_env)
                    if al_res.get('alert_sent'):
                        st.warning(f"Alerte envoyée (coût dernière heure = {al_res.get('cost_last_hour'):.4f} USD)")
                    else:
                        st.info(f"Aucune alerte nécessaire (coût dernière heure = {al_res.get('cost_last_hour'):.4f} USD)")

    # show previous convo if present
    if st.session_state.get('convo'):
        st.sidebar.markdown('### Conversation en mémoire')
        for msg in st.session_state['convo']:
            st.sidebar.write(f"({msg['round']}) {msg['provider']}: {msg['text'][:120]}...")

    # Detailed usage history and exports
    st.markdown('---')
    st.subheader('Historique détaillé des appels LLM')
    usage_file = Path(os.getenv('LLM_USAGE_FILE', 'logs/llm_usage.json'))
    try:
        entries = json.loads(usage_file.read_text()) if usage_file.exists() else []
    except Exception:
        entries = []

    if not entries:
        st.info('Aucun enregistrement d\'usage pour le moment.')
    else:
        df_usage = pd.DataFrame(entries)
        if 'ts' in df_usage.columns:
            df_usage['ts'] = pd.to_datetime(df_usage['ts'], unit='s')
        if 'extra' in df_usage.columns:
            df_usage['error'] = df_usage['extra'].apply(lambda e: e.get('error') if isinstance(e, dict) else None)
            df_usage['api_prompt_tokens'] = df_usage['extra'].apply(lambda e: (e.get('api_usage') or {}).get('prompt_tokens') if isinstance(e, dict) else None)
            df_usage['api_completion_tokens'] = df_usage['extra'].apply(lambda e: (e.get('api_usage') or {}).get('completion_tokens') if isinstance(e, dict) else None)
        provider_options = ['Tous'] + sorted(df_usage['provider'].dropna().unique().tolist())
        provider_filter = st.selectbox('Filtrer par provider', provider_options)
        n_rows = st.number_input('Lignes affichées', min_value=10, max_value=1000, value=200)
        if provider_filter != 'Tous':
            df_show = df_usage[df_usage['provider'] == provider_filter]
        else:
            df_show = df_usage
        st.dataframe(df_show.sort_values('ts', ascending=False).head(n_rows), use_container_width=True)
        csv_bytes = df_show.to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger historique CSV", csv_bytes, file_name='llm_usage_history.csv', mime='text/csv')

    if st.button('Effacer historique usage (main)'):
        try:
            usage_file.write_text('[]')
            st.success('Logs usage effacés')
        except Exception:
            st.error('Impossible d\'effacer les logs')


if __name__ == '__main__':
    # Standalone runner
    st.set_page_config(page_title='LLM BrainNet Console', layout='wide')
    render_llm_console()
