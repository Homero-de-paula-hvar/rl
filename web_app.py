#!/usr/bin/env python3
"""
AutoRL Web Application
Interface web para o framework de Auto Reinforcement Learning
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
import json
import tempfile
from datetime import datetime
import sys

# Adicionar o diretório src ao path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from autorl_framework.simulation.auto_rl_engine import AutoRLEngine

# Função para redirecionar prints para arquivo e console
class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    def close(self):
        self.file.close()

# Redirecionar prints para arquivo por execução
log_filename = f"log_experimento_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
sys.stdout = Tee(log_filename)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'autorl-secret-key'

# Configurações globais
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Upload de arquivos CSV"""
    try:
        if 'control_file' not in request.files or 'test_file' not in request.files:
            return jsonify({'error': 'Ambos os arquivos são necessários'}), 400
        
        control_file = request.files['control_file']
        test_file = request.files['test_file']
        
        if control_file.filename == '' or test_file.filename == '':
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
        
        # Salvar arquivos
        control_path = os.path.join(UPLOAD_FOLDER, 'control_group.csv')
        test_path = os.path.join(UPLOAD_FOLDER, 'test_group.csv')
        
        control_file.save(control_path)
        test_file.save(test_path)
        
        return jsonify({'success': True, 'message': 'Arquivos carregados com sucesso'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    """Executar experimento de AutoRL"""
    try:
        data = request.get_json()
        
        # Parâmetros do experimento
        algorithm = data.get('algorithm', 'epsilon_greedy')
        optimizer = data.get('optimizer', 'genetic')
        n_generations = int(data.get('n_generations', 20))
        n_iterations = int(data.get('n_iterations', 3))
        use_sample_data = data.get('use_sample_data', True)
        
        # Mapear nomes amigáveis
        algorithm_names = {
            'epsilon_greedy': 'Epsilon-Greedy',
            'ucb': 'UCB1 (Upper Confidence Bound)',
            'thompson': 'Thompson Sampling',
            'ab_test': 'A/B Test Agent'
        }
        
        optimizer_names = {
            'genetic': 'Algoritmo Genético',
            'pso': 'PSO (Particle Swarm Optimization)',
            'differential_evolution': 'Evolução Diferencial',
            'grid_search': 'Grid Search',
            'random_search': 'Random Search',
            'bayesian_optimization': 'Bayesian Optimization'
        }
        
        # Carregar dados
        if use_sample_data:
            control_path = "data/control_group.csv"
            test_path = "data/test_group.csv"
        else:
            control_path = os.path.join(UPLOAD_FOLDER, 'control_group.csv')
            test_path = os.path.join(UPLOAD_FOLDER, 'test_group.csv')
        
        # Preparar dados
        df_control = pd.read_csv(control_path, sep=';')
        df_test = pd.read_csv(test_path, sep=';')
        
        # Processar dados
        df_full = prepare_data(df_control, df_test)
        
        # Estatísticas dos dados
        data_stats = {
            'total_campaigns': len(df_full['campaign_name'].unique()),
            'total_days': len(df_full),
            'avg_purchases': float(df_full['purchases'].mean()),
            'max_purchases': float(df_full['purchases'].max()),
            'total_spend': float(df_full['spend'].sum()),
            'avg_roi': float(df_full['roi'].mean()),
            'conversion_rate': float(df_full['conversion_rate'].mean())
        }
        
        # Executar experimento
        engine = AutoRLEngine(data=df_full, arms_col='campaign_name', reward_col='purchases_reward')
        results = engine.run_comparison(
            algorithms=[algorithm],
            optimizers=[optimizer],
            n_iterations=n_iterations
        )
        
        # Converter resultados para JSON
        results_json = results.to_dict('records')
        
        # Calcular estatísticas detalhadas
        best_run = results.loc[results['total_reward'].idxmax()]
        worst_run = results.loc[results['total_reward'].idxmin()]
        
        # Histórico de otimização (se disponível)
        optimization_history = []
        if hasattr(engine, 'optimization_history'):
            optimization_history = engine.optimization_history
        
        # Métricas de treinamento
        training_metrics = {
            'total_iterations': len(results),
            'successful_runs': len(results[results['total_reward'] > 0]),
            'failed_runs': len(results[results['total_reward'] <= 0]),
            'convergence_rate': len(results[results['total_reward'] > results['total_reward'].mean()]) / len(results),
            'std_reward': float(results['total_reward'].std()),
            'std_regret': float(results['total_regret'].std()),
            'reward_improvement': float(best_run['total_reward'] - worst_run['total_reward']),
            'regret_reduction': float(worst_run['total_regret'] - best_run['total_regret'])
        }
        
        # Hiperparâmetros otimizados
        optimized_params = {}
        if 'best_params' in best_run and best_run['best_params']:
            optimized_params = best_run['best_params']
        
        # Configuração do experimento
        experiment_config = {
            'algorithm': algorithm_names.get(algorithm, algorithm),
            'optimizer': optimizer_names.get(optimizer, optimizer),
            'n_generations': n_generations,
            'n_iterations': n_iterations,
            'data_source': 'Dados de Exemplo' if use_sample_data else 'Upload Customizado',
            'reward_metric': 'Número de Compras',
            'optimization_goal': 'Maximizar Recompensa Total - Penalizar Regret'
        }
        
        return jsonify({
            'success': True,
            'results': results_json,
            'experiment_config': experiment_config,
            'data_statistics': data_stats,
            'training_metrics': training_metrics,
            'optimized_parameters': optimized_params,
            'optimization_history': optimization_history,
            'summary': {
                'total_runs': len(results),
                'avg_reward': float(results['total_reward'].mean()),
                'avg_regret': float(results['total_regret'].mean()),
                'best_reward': float(results['total_reward'].max()),
                'worst_reward': float(results['total_reward'].min()),
                'best_regret': float(results['total_regret'].min()),
                'worst_regret': float(results['total_regret'].max()),
                'best_params': best_run['best_params'] if 'best_params' in best_run else {},
                'best_run_id': int(best_run.name),
                'worst_run_id': int(worst_run.name)
            },
            'detailed_results': {
                'best_run': {
                    'iteration': int(best_run.name),
                    'reward': float(best_run['total_reward']),
                    'regret': float(best_run['total_regret']),
                    'params': best_run['best_params'] if 'best_params' in best_run else {},
                    'fitness': float(best_run['best_fitness']) if 'best_fitness' in best_run else 0
                },
                'worst_run': {
                    'iteration': int(worst_run.name),
                    'reward': float(worst_run['total_reward']),
                    'regret': float(worst_run['total_regret']),
                    'params': worst_run['best_params'] if 'best_params' in worst_run else {},
                    'fitness': float(worst_run['best_fitness']) if 'best_fitness' in worst_run else 0
                }
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def prepare_data(df_control, df_test):
    """Preparar dados para o experimento"""
    new_cols = {
        'Campaign Name': 'campaign_name', 'Date': 'date', 'Spend [USD]': 'spend',
        '# of Impressions': 'impressions', 'Reach': 'reach', '# of Website Clicks': 'website_clicks',
        '# of Searches': 'searches', '# of View Content': 'view_content',
        '# of Add to Cart': 'add_to_cart', '# of Purchase': 'purchases'
    }
    df_control = df_control.rename(columns=new_cols)
    df_test = df_test.rename(columns=new_cols)
    
    # Preencher valores NaN com a média da coluna
    for col in df_control.columns:
        if df_control[col].dtype in ['float64', 'int64'] and df_control[col].isnull().any():
            mean_val = df_control[col].mean()
            df_control[col] = df_control[col].fillna(mean_val)
    
    for col in df_test.columns:
        if df_test[col].dtype in ['float64', 'int64'] and df_test[col].isnull().any():
            mean_val = df_test[col].mean()
            df_test[col] = df_test[col].fillna(mean_val)
    
    df_full = pd.concat([df_control, df_test], ignore_index=True)
    df_full['date'] = pd.to_datetime(df_full['date'], format='%d.%m.%Y', errors='coerce')
    df_full = df_full.sort_values('date')
    
    # Criar múltiplas métricas de recompensa
    # 1. Conversão binária (se houve compra)
    df_full['binary_conversion'] = (df_full['purchases'] > 0).astype(int)
    
    # 2. Número de compras (recompensa principal)
    df_full['purchases_reward'] = df_full['purchases']
    
    # 3. ROI (Return on Investment) - compras / gasto
    df_full['roi'] = df_full['purchases'] / df_full['spend']
    df_full['roi'] = df_full['roi'].fillna(0)  # Se não houve gasto, ROI = 0
    
    # 4. Taxa de conversão (compras / impressões)
    df_full['conversion_rate'] = df_full['purchases'] / df_full['impressions']
    df_full['conversion_rate'] = df_full['conversion_rate'].fillna(0)
    
    # 5. Recompensa normalizada (0-1) baseada no histórico
    max_purchases = df_full['purchases'].max()
    df_full['normalized_reward'] = df_full['purchases'] / max_purchases if max_purchases > 0 else 0
    
    return df_full

@app.route('/download_results', methods=['POST'])
def download_results():
    """Download dos resultados em CSV"""
    try:
        data = request.get_json()
        results_data = data.get('results', [])
        
        if not results_data:
            return jsonify({'error': 'Nenhum resultado para download'}), 400
        
        # Criar DataFrame
        df = pd.DataFrame(results_data)
        
        # Salvar temporariamente
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        df.to_csv(temp_file.name, index=False)
        
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=f'autorl_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mimetype='text/csv'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🤖 AutoRL Web Application")
    print("=" * 40)
    print("Acesse: http://localhost:8080")
    print("=" * 40)
    app.run(debug=True, host='0.0.0.0', port=8080) 