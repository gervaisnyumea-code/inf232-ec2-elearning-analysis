import os
import sys
import time
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
from urllib.request import Request, urlopen

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Charger .env tôt pour que les clés et quotas soient disponibles
try:
    from src.env_loader import load_dotenv
    load_dotenv(str(project_root / '.env'))
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

    def _get_orchestrator():
        try:
            import importlib
            # Reload llm_integration FIRST so LLMClient picks up force_real param
            import src.llm_integration as llm_int_mod
            importlib.reload(llm_int_mod)
            import src.llm_orchestrator as llm_orch_mod
            importlib.reload(llm_orch_mod)
            return llm_orch_mod.LLMOrchestrator()
        except Exception:
            try:
                return LLMOrchestrator()
            except Exception:
                return None

    orch = _get_orchestrator()
    providers = orch.available_providers() if orch else []

    st.sidebar.header('Configuration')
    st.sidebar.write('Providers détectés: ' + (', '.join(providers) if providers else 'aucun'))
    rounds = st.sidebar.slider('Nombre de rounds de concertation', 0, 5, 1)
    use_real = st.sidebar.checkbox('Activer appels réels (respecter quotas)', value=(os.getenv('LLM_CALLS_ENABLED', 'false').lower() in ('1','true','yes')))
    
    # Show quota status
    if use_real and orch and providers:
        try:
            from src.llm_usage import summary as get_usage_summary
            usage = get_usage_summary()
            quota_max = int(os.getenv('LLM_MAX_CALLS_PER_HOUR', '200'))
            calls_remaining = quota_max - usage['total_calls']
            if calls_remaining > 0:
                st.sidebar.success(f'Quota: {usage["total_calls"]}/{quota_max} appels (reste: {calls_remaining})')
            else:
                st.sidebar.error(f'Quota: {usage["total_calls"]}/{quota_max} appels (ÉPUISÉ!)')
        except Exception:
            st.sidebar.info('Quota: Indisponible')
    elif not use_real:
        st.sidebar.info('Mode simulation (pas d\'appels API)')
    else:
        st.sidebar.warning('Aucun provider disponible')

    st.sidebar.markdown('---')
    st.sidebar.header('Concertation continue')
    runner_pid_file = Path('logs/llm_runner.pid')
    runner_status = None
    if runner_pid_file.exists():
        try:
            runner_pid = int(runner_pid_file.read_text().strip())
            os.kill(runner_pid, 0)
            runner_status = f'Running (PID {runner_pid})'
        except Exception:
            runner_status = 'Not running'
    else:
        runner_status = 'Not running'
    st.sidebar.write(f'Runner status: {runner_status}')
    runner_interval = st.sidebar.number_input('Intervalle concertation (s)', min_value=10, max_value=86400, value=int(os.getenv('LLM_PERIODIC_INTERVAL_SEC','3600')))
    if st.sidebar.button('Démarrer runner now'):
        import subprocess
        # Use absolute paths for runner script and logs
        runner_script = str(project_root / 'scripts' / 'llm_periodic_runner.py')
        log_out_path = str(project_root / 'logs' / 'llm_periodic.out')
        log_err_path = str(project_root / 'logs' / 'llm_periodic.err')
        
        runner_pid_file.parent.mkdir(parents=True, exist_ok=True)
        log_dir = Path(log_out_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        py = str(project_root / 'venv' / 'bin' / 'python3')
        cmd = [py, runner_script]
        
        # Set environment variables for the subprocess
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root)
        
        log_out = open(log_out_path, 'a')
        log_err = open(log_err_path, 'a')
        p = subprocess.Popen(cmd, stdout=log_out, stderr=log_err, stdin=subprocess.DEVNULL, 
                             close_fds=True, env=env, cwd=str(project_root))
        runner_pid_file.write_text(str(p.pid))
        st.sidebar.success(f'Runner démarré (PID {p.pid})')
    if st.sidebar.button('Arrêter runner'):
        full_pid_path = project_root / 'logs' / 'llm_runner.pid'
        if full_pid_path.exists():
            try:
                pid = int(full_pid_path.read_text().strip())
                os.kill(pid, 15)
                full_pid_path.unlink()
                if runner_pid_file.exists():
                    runner_pid_file.unlink()
                st.sidebar.success('Runner stoppé')
            except Exception as e:
                st.sidebar.error(f'Erreur arrêt runner: {e}')
        elif runner_pid_file.exists():
            try:
                pid = int(runner_pid_file.read_text().strip())
                os.kill(pid, 15)
                runner_pid_file.unlink()
                st.sidebar.success('Runner stoppé')
            except Exception as e:
                st.sidebar.error(f'Erreur arrêt runner: {e}')
        else:
            st.sidebar.info('Runner non trouvé.')


    # Usage summary (tokens / coût estimé)
    usage_path = project_root / 'logs' / 'llm_usage.json'
    if not usage_path.exists():
        usage_path = Path(os.getenv('LLM_USAGE_FILE', str(project_root / 'logs' / 'llm_usage.json')))
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

    if st.button('Démarrer la concertation'):
        if not question.strip():
            st.warning('Entrez une question.')
        elif not providers:
            st.warning('Aucun provider détecté. Vérifiez .env ou exécutez en mode simulation.')
        else:
            st.session_state['convo'] = []
            include_data = st.sidebar.checkbox('Inclure contexte données live', value=True)
            data_window_sec = st.sidebar.number_input('Fenêtre données (s)', min_value=60, max_value=86400, value=3600)
            force_real = use_real
            with st.spinner('Concertation en cours...'):
                try:
                    res = orch.concert_and_merge(question, rounds=rounds, include_data=include_data, data_window_sec=int(data_window_sec), max_tokens=256, force_real=force_real)
                except Exception as e:
                    st.error(f"Erreur pendant la concertation: {e}")
                    res = None
            if res:
                st.markdown('### Contexte de données')
                st.code(res.get('data_summary', ''))

                st.markdown('### Contributions par tour')
                for msg in res.get('rounds', []):
                    text = msg['text']
                    # Style placeholder messages to make them clearly visible
                    if text.startswith('['):
                        st.warning(f"({msg['round']}) **{msg['provider']}**: {text}")
                    else:
                        st.success(f"({msg['round']}) **{msg['provider']}**: {text}")

                st.markdown('### Réponse fusionnée')
                merged = res.get('merged', '')
                if merged.startswith('['):
                    st.warning(merged)
                else:
                    st.info(merged)

                out_dir = project_root / 'reports'
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = int(time.time())
                out_path = out_dir / f'llm_conversation_{ts}.json'
                out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2))
                st.success(f'Conversation sauvegardée -> {out_path}')

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
    usage_file = project_root / 'logs' / 'llm_usage.json'
    if not usage_file.exists():
        usage_file = Path(os.getenv('LLM_USAGE_FILE', str(project_root / 'logs' / 'llm_usage.json')))
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
        df_disp = df_show.sort_values('ts', ascending=False).head(n_rows)
        st.dataframe(df_disp, width='stretch')
        csv_bytes = df_show.to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger historique CSV", csv_bytes, file_name='llm_usage_history.csv', mime='text/csv')

        # Cumulative cost chart per provider
        try:
            if not df_show.empty and 'cost' in df_show.columns:
                parts = []
                for p in df_show['provider'].dropna().unique():
                    sub = df_show[df_show['provider']==p].copy()
                    sub['ts_dt'] = pd.to_datetime(sub['ts'], unit='s')
                    sub = sub.sort_values('ts_dt')
                    sub['cum_cost'] = sub['cost'].cumsum()
                    parts.append(sub[['ts_dt','cum_cost']].assign(provider=p))
                if parts:
                    df_plot = pd.concat(parts)
                    df_plot = df_plot.sort_values('ts_dt')
                    fig_cost = px.line(df_plot, x='ts_dt', y='cum_cost', color='provider', title='Coût cumulé par provider')
                    st.plotly_chart(fig_cost, use_container_width=True)
        except Exception:
            pass

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
