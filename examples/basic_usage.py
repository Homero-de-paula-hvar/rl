#!/usr/bin/env python3
"""
Exemplo básico de uso do Auto RL Framework
==========================================

Este exemplo demonstra como usar o framework para:
1. Carregar dados
2. Executar algoritmos de RL
3. Otimizar hiperparâmetros
4. Comparar resultados
"""

import sys
import os
import pandas as pd

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from autorl_framework.simulation.auto_rl_engine import AutoRLEngine
from autorl_framework.rl_algs import EpsilonGreedy, UCB1, ThompsonSampling

def load_sample_data():
    """Carrega dados de exemplo"""
    # Carregar datasets
    df_control = pd.read_csv('../data/control_group.csv', sep=';')
    df_test = pd.read_csv('../data/test_group.csv', sep=';')

    # Renomear colunas
    new_cols = {
        'Campaign Name': 'campaign_name', 'Date': 'date', 'Spend [USD]': 'spend',
        '# of Impressions': 'impressions', 'Reach': 'reach', '# of Website Clicks': 'website_clicks',
        '# of Searches': 'searches', '# of View Content': 'view_content',
        '# of Add to Cart': 'add_to_cart', '# of Purchase': 'purchases'
    }
    df_control = df_control.rename(columns=new_cols)
    df_test = df_test.rename(columns=new_cols)

    # Imputação de dados ausentes
    for col in df_control.columns:
        if df_control[col].isnull().any():
            mean_val = df_control[col].mean()
            df_control[col] = df_control[col].fillna(mean_val)

    # Combinar dataframes
    df_full = pd.concat([df_control, df_test], ignore_index=True)
    df_full['date'] = pd.to_datetime(df_full['date'])
    df_full = df_full.sort_values('date')

    # Definir recompensa binária
    df_full['binary_conversion'] = (df_full['purchases'] > 0).astype(int)

    return df_full

def example_1_basic_algorithms():
    """Exemplo 1: Executar algoritmos básicos"""
    print("=" * 50)
    print("Exemplo 1: Algoritmos Básicos")
    print("=" * 50)
    
    # Carregar dados
    data = load_sample_data()
    print(f"Dados carregados: {len(data)} registros")
    
    # Criar engine
    engine = AutoRLEngine(data=data, arms_col='campaign_name', reward_col='binary_conversion')
    print(f"Número de braços: {engine.n_arms}")
    
    # Executar algoritmos básicos
    algorithms = {
        "Epsilon-Greedy": EpsilonGreedy(n_arms=engine.n_arms, epsilon=0.1),
        "UCB1": UCB1(n_arms=engine.n_arms),
        "Thompson Sampling": ThompsonSampling(n_arms=engine.n_arms)
    }
    
    results = []
    for name, alg in algorithms.items():
        print(f"Executando {name}...")
        result = engine.simulator.run(algorithm=alg, run_name=name)
        results.append(result)
    
    # Combinar resultados
    all_results = pd.concat(results, ignore_index=True)
    
    # Mostrar resumo
    summary = all_results.groupby('run_name').agg({
        'reward': ['sum', 'mean'],
        'regret': 'sum'
    }).round(3)
    summary.columns = ['Recompensa Total', 'Recompensa Média', 'Arrependimento Total']
    print("\nResumo dos resultados:")
    print(summary)
    
    return engine, all_results

def example_2_hyperparameter_optimization():
    """Exemplo 2: Otimização de hiperparâmetros"""
    print("\n" + "=" * 50)
    print("Exemplo 2: Otimização de Hiperparâmetros")
    print("=" * 50)
    
    # Usar engine do exemplo anterior
    engine, _ = example_1_basic_algorithms()
    
    # Otimizar epsilon-greedy com algoritmo genético
    print("Otimizando Epsilon-Greedy com Algoritmo Genético...")
    best_params, best_score = engine.optimize_hyperparameters(
        algorithm_name='epsilon_greedy',
        optimizer_name='genetic',
        n_iterations=20  # Reduzido para demonstração
    )
    
    print(f"Melhor score: {best_score:.3f}")
    print("Melhores parâmetros:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    return engine

def example_3_auto_comparison():
    """Exemplo 3: Comparação automática"""
    print("\n" + "=" * 50)
    print("Exemplo 3: Comparação Automática")
    print("=" * 50)
    
    # Carregar dados
    data = load_sample_data()
    engine = AutoRLEngine(data=data, arms_col='campaign_name', reward_col='binary_conversion')
    
    # Executar comparação automática
    results_df = engine.run_comparison(
        algorithms=['epsilon_greedy', 'ppo'],
        optimizers=['genetic', 'pso'],
        n_iterations=15  # Reduzido para demonstração
    )
    
    print("Resultados da comparação automática:")
    print(results_df)
    
    # Mostrar melhor resultado
    best_idx = results_df['total_reward'].idxmax()
    best_result = results_df.loc[best_idx]
    print(f"\n🎯 Melhor resultado: {best_result['algorithm']} com {best_result['optimizer']}")
    print(f"   Score: {best_result['total_reward']:.3f}")
    
    return results_df

def main():
    """Função principal"""
    print("🤖 Auto RL Framework - Exemplos de Uso")
    print("=" * 60)
    
    try:
        # Exemplo 1: Algoritmos básicos
        example_1_basic_algorithms()
        
        # Exemplo 2: Otimização de hiperparâmetros
        example_2_hyperparameter_optimization()
        
        # Exemplo 3: Comparação automática
        example_3_auto_comparison()
        
        print("\n" + "=" * 60)
        print("✅ Todos os exemplos executados com sucesso!")
        print("💡 Para interface visual, execute: streamlit run ../app.py")
        
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        print("Certifique-se de que os dados estão na pasta 'data/'")

if __name__ == "__main__":
    main() 