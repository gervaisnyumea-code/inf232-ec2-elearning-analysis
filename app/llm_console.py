import os
import time
import json
from pathlib import Path

import streamlit as st

from src.llm_integration import LLMClient
from src.llm_orchestrator import LLMOrchestrator


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

    # show previous convo if present
    if st.session_state.get('convo'):
        st.sidebar.markdown('### Conversation en mémoire')
        for msg in st.session_state['convo']:
            st.sidebar.write(f"({msg['round']}) {msg['provider']}: {msg['text'][:120]}...")


if __name__ == '__main__':
    # Standalone runner
    st.set_page_config(page_title='LLM BrainNet Console', layout='wide')
    render_llm_console()
