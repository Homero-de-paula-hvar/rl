#!/usr/bin/env python3
"""
Auto RL Framework - Aplicação Principal
=======================================

Este é o arquivo principal para executar o framework de Auto Reinforcement Learning
com interface visual Streamlit.

Para executar:
    streamlit run app.py

Ou:
    python -m streamlit run app.py
"""

import sys
import os
import streamlit as st
import pandas as pd
import tempfile
from src.autorl_framework.simulation.auto_rl_engine import AutoRLEngine

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

st.set_page_config(page_title="AutoRL Dashboard", layout="wide")
st.title("🤖 Auto Reinforcement Learning Dashboard")
st.markdown("""
Este painel permite rodar experimentos de AutoRL, otimizar hiperparâmetros com algoritmos evolutivos e visualizar resultados de forma interativa.
""")

# --- 1. Upload/Seleção de Dados ---
st.header("1. Dados de Entrada")
data_option = st.radio("Escolha a fonte dos dados:", ["Usar dados de exemplo", "Fazer upload de CSV"])

if data_option == "Usar dados de exemplo":
    control_path = "data/control_group.csv"
    test_path = "data/test_group.csv"
    df_control = pd.read_csv(control_path, sep=';')
    df_test = pd.read_csv(test_path, sep=';')
else:
    uploaded_control = st.file_uploader("Upload do grupo de controle (CSV)", type=["csv"], key="control")
    uploaded_test = st.file_uploader("Upload do grupo de teste (CSV)", type=["csv"], key="test")
    df_control, df_test = None, None
    if uploaded_control is not None and uploaded_test is not None:
        df_control = pd.read_csv(uploaded_control, sep=';')
        df_test = pd.read_csv(uploaded_test, sep=';')

if df_control is not None and df_test is not None:
    st.success(f"Dados carregados! {len(df_control)} linhas (controle), {len(df_test)} linhas (teste)")
    # Preview
    with st.expander("Pré-visualizar dados de controle"):
        st.dataframe(df_control.head(10))
    with st.expander("Pré-visualizar dados de teste"):
        st.dataframe(df_test.head(10))
else:
    st.warning("Por favor, carregue ambos os arquivos CSV.")
    st.stop()

# --- 2. Configuração do Experimento ---
st.header("2. Configuração do Experimento")
algorithms = {
    "Epsilon-Greedy": "epsilon_greedy",
    "UCB1": "ucb",
    "Thompson Sampling": "thompson"
}
optimizers = {
    "Algoritmo Genético": "genetic",
    "PSO": "pso",
    "Evolução Diferencial": "differential_evolution"
}

col1, col2 = st.columns(2)
with col1:
    alg_display = st.selectbox("Algoritmo de RL", list(algorithms.keys()))
    alg = algorithms[alg_display]
with col2:
    opt_display = st.selectbox("Otimizador Evolutivo", list(optimizers.keys()))
    opt = optimizers[opt_display]

n_generations = st.slider("Gerações de Otimização", 5, 100, 20)
n_iterations = st.slider("Execuções por combinação", 1, 10, 3)

# --- 3. Execução do Experimento ---
st.header("3. Execução e Monitoramento")
run_exp = st.button("Rodar Experimento de AutoRL 🚀")

# Função para preparar dados
def prepare_data(df_control, df_test):
    new_cols = {
        'Campaign Name': 'campaign_name', 'Date': 'date', 'Spend [USD]': 'spend',
        '# of Impressions': 'impressions', 'Reach': 'reach', '# of Website Clicks': 'website_clicks',
        '# of Searches': 'searches', '# of View Content': 'view_content',
        '# of Add to Cart': 'add_to_cart', '# of Purchase': 'purchases'
    }
    df_control = df_control.rename(columns=new_cols)
    df_test = df_test.rename(columns=new_cols)
    for col in df_control.columns:
        if df_control[col].isnull().any():
            mean_val = df_control[col].mean()
            df_control[col] = df_control[col].fillna(mean_val)
    df_full = pd.concat([df_control, df_test], ignore_index=True)
    df_full['date'] = pd.to_datetime(df_full['date'], format='%d.%m.%Y', errors='coerce')
    df_full = df_full.sort_values('date')
    df_full['binary_conversion'] = (df_full['purchases'] > 0).astype(int)
    return df_full

if run_exp:
    with st.spinner("Executando experimento e otimizando hiperparâmetros..."):
        df_full = prepare_data(df_control, df_test)
        engine = AutoRLEngine(data=df_full, arms_col='campaign_name', reward_col='binary_conversion')
        results_df = engine.run_comparison(
            algorithms=[alg],
            optimizers=[opt],
            n_iterations=n_iterations
        )
        st.session_state['last_results'] = results_df
    st.success("Experimento concluído!")

# --- 4. Resultados e Visualização ---
st.header("4. Resultados e Visualização")
results_df = st.session_state.get('last_results', None)
if results_df is not None and not results_df.empty:
    st.subheader("Tabela de Resultados")
    st.dataframe(results_df)
    import matplotlib.pyplot as plt
    import seaborn as sns
    import io
    # Gráfico de recompensa
    fig1, ax1 = plt.subplots(figsize=(8,4))
    sns.barplot(data=results_df, x='iteration', y='total_reward', ax=ax1)
    ax1.set_title('Recompensa Total por Execução')
    st.pyplot(fig1)
    # Gráfico de arrependimento
    fig2, ax2 = plt.subplots(figsize=(8,4))
    sns.barplot(data=results_df, x='iteration', y='total_regret', ax=ax2)
    ax2.set_title('Arrependimento Total por Execução')
    st.pyplot(fig2)
    # Download dos resultados
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar Resultados CSV", csv, "resultados_autorl.csv", "text/csv")
else:
    st.info("Rode um experimento para visualizar os resultados.")

# --- 5. Histórico de Execuções (opcional) ---
st.header("5. Histórico de Execuções (Sessão)")
if 'history' not in st.session_state:
    st.session_state['history'] = []
if run_exp and results_df is not None:
    st.session_state['history'].append(results_df)
if st.session_state['history']:
    for i, hist_df in enumerate(st.session_state['history']):
        with st.expander(f"Execução #{i+1}"):
            st.dataframe(hist_df)
else:
    st.info("Nenhuma execução anterior nesta sessão.") 