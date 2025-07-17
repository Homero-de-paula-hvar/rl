# 🤖 Auto Reinforcement Learning Framework

Um framework completo e avançado para otimização automática de algoritmos de Reinforcement Learning, com interface web moderna, algoritmos evolutivos para tuning de hiperparâmetros e análise detalhada de resultados.

## 🚀 Características Principais

- **🎯 Algoritmos de RL Avançados**: Epsilon-Greedy, UCB1, Thompson Sampling
- **🧬 Otimizadores Evolutivos**: Algoritmo Genético, PSO, Differential Evolution
- **🌐 Interface Web Moderna**: Aplicação Flask com dashboard avançado
- **📊 Métricas Detalhadas**: Análise completa de treinamento e otimização
- **🔧 Auto RL**: Otimização automática de hiperparâmetros
- **📈 Visualizações Interativas**: Gráficos múltiplos e relatórios
- **📄 Relatórios Completos**: Download de resultados e análises
- **🎨 UI/UX Profissional**: Interface responsiva e intuitiva

## 📁 Estrutura do Projeto

```
autorl_framework/
├── src/autorl_framework/
│   ├── rl_algs/           # Algoritmos de Reinforcement Learning
│   │   ├── base.py
│   │   ├── epsilon_greedy.py
│   │   ├── ucb.py
│   │   └── thompson_sampling.py
│   ├── optimizers/        # Algoritmos de otimização evolutiva
│   │   ├── base.py
│   │   ├── genetic_algorithm.py
│   │   ├── particle_swarm.py
│   │   └── differential_evolution.py
│   └── simulation/        # Simulação e engine principal
│       ├── simulator.py
│       └── auto_rl_engine.py
├── templates/             # Templates HTML da interface web
│   └── index.html
├── static/               # Arquivos estáticos (CSS, JS)
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── data/                 # Datasets de exemplo
│   ├── control_group.csv
│   └── test_group.csv
├── web_app.py           # Aplicação Flask principal
├── main.py              # Script de linha de comando
├── test_evolutionary_optimization.py  # Testes do framework
├── requirements.txt     # Dependências Python
└── README.md
```

## 🛠️ Instalação

### Pré-requisitos

- Python 3.12+
- pip ou poetry

### Instalação Rápida

```bash
# Clonar o repositório
git clone <repository-url>
cd autorl_framework

# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

### Verificação da Instalação

```bash
# Verificar se tudo está funcionando
python check_install.py
```

## 🚀 Como Usar

### Interface Web (Recomendado)

```bash
# Ativar ambiente virtual
source venv/bin/activate

# Executar aplicação web
python web_app.py
```

Acesse: **http://localhost:8080**

### Funcionalidades da Interface Web

#### 📊 **Gerenciamento de Dados de Entrada**
- Upload de arquivos CSV customizados
- Dados de exemplo (campanhas de marketing)
- Prévia dos dados antes da execução

#### ⚙️ **Configuração do Experimento**
- Seleção de algoritmos RL com descrições detalhadas
- Otimizadores evolutivos configuráveis
- Ajuste de gerações e iterações

#### 📋 **Configuração do Experimento Executado**
- Resumo completo dos parâmetros utilizados
- Fonte de dados e métricas de recompensa
- Objetivo de otimização

#### 📊 **Estatísticas dos Dados**
- Total de campanhas e dias de dados
- Gasto total e compras médias
- ROI médio e taxa de conversão

#### 📈 **Métricas de Treinamento**
- Total de execuções vs bem-sucedidas vs falharam
- Taxa de convergência
- Desvio padrão de recompensas e arrependimentos
- Melhorias obtidas na otimização

#### 🔧 **Hiperparâmetros Otimizados**
- Melhores parâmetros encontrados
- Detalhes da melhor e pior execução
- Valores de fitness e performance

#### 📊 **Resultados e Visualizações**
- **4 Gráficos Interativos**:
  - Evolução das Recompensas
  - Evolução do Arrependimento
  - Distribuição de Recompensas
  - Correlação: Recompensa vs Arrependimento
- Tabela detalhada de resultados
- Download de CSV e relatórios completos

### Script de Linha de Comando

```bash
# Executar framework via script
python main.py
```

## 📊 Algoritmos Implementados

### 🎯 Algoritmos de Multi-Armed Bandit

- **Epsilon-Greedy**: Exploração vs explotação balanceada
  - Hiperparâmetro: `epsilon` (taxa de exploração)
- **UCB1**: Upper Confidence Bound
  - Hiperparâmetro: `alpha` (fator de exploração)
- **Thompson Sampling**: Amostragem bayesiana
  - Hiperparâmetro: `prior_alpha`, `prior_beta` (priors)

### 🧬 Otimizadores Evolutivos

- **Algoritmo Genético**: Seleção natural
  - População, taxa de mutação, taxa de crossover
- **PSO**: Particle Swarm Optimization
  - Inércia, cognição, social
- **Differential Evolution**: Evolução diferencial
  - Fator de diferenciação, taxa de crossover

## 🔧 Configuração Avançada

### Hiperparâmetros dos Algoritmos

```python
# Exemplo: Epsilon-Greedy
epsilon_greedy_params = {
    'epsilon': (0.01, 0.5)  # Taxa de exploração
}

# Exemplo: UCB1
ucb_params = {
    'alpha': (0.1, 2.0)  # Fator de exploração
}

# Exemplo: Thompson Sampling
thompson_params = {
    'prior_alpha': (0.1, 2.0),
    'prior_beta': (0.1, 2.0)
}
```

### Configuração de Otimizadores

```python
# Algoritmo Genético
ga_config = {
    'population_size': 50,
    'mutation_rate': 0.1,
    'crossover_rate': 0.8,
    'n_generations': 20
}

# PSO
pso_config = {
    'n_particles': 30,
    'inertia': 0.7,
    'cognition': 1.5,
    'social': 1.5
}
```

## 📈 Exemplos de Uso

### Otimização Individual

```python
from autorl_framework.simulation.auto_rl_engine import AutoRLEngine
import pandas as pd

# Carregar dados
df_control = pd.read_csv('data/control_group.csv', sep=';')
df_test = pd.read_csv('data/test_group.csv', sep=';')

# Preparar dados
df_full = pd.concat([df_control, df_test], ignore_index=True)

# Criar engine
engine = AutoRLEngine(
    data=df_full, 
    arms_col='campaign_name', 
    reward_col='purchases_reward'
)

# Otimizar Epsilon-Greedy com algoritmo genético
results = engine.run_comparison(
    algorithms=['epsilon_greedy'],
    optimizers=['genetic'],
    n_iterations=3
)

print(f"Melhor recompensa: {results['total_reward'].max()}")
print(f"Melhores parâmetros: {results.loc[results['total_reward'].idxmax(), 'best_params']}")
```

### Comparação Automática

```python
# Comparar múltiplos algoritmos e otimizadores
results = engine.run_comparison(
    algorithms=['epsilon_greedy', 'ucb', 'thompson'],
    optimizers=['genetic', 'pso', 'differential_evolution'],
    n_iterations=5
)

# Análise dos resultados
print("Resultados por algoritmo:")
for alg in results['algorithm'].unique():
    alg_results = results[results['algorithm'] == alg]
    print(f"{alg}: {alg_results['total_reward'].mean():.2f} ± {alg_results['total_reward'].std():.2f}")
```

## 📊 Métricas e Análises

### Métricas de Performance

- **Recompensa Total**: Soma das recompensas obtidas
- **Recompensa Média**: Média por iteração
- **Arrependimento Total**: Diferença para o ótimo
- **Taxa de Convergência**: % de execuções acima da média
- **Desvio Padrão**: Variabilidade dos resultados

### Métricas de Dados

- **Total de Campanhas**: Número de arms disponíveis
- **Total de Dias**: Período de dados
- **Gasto Total**: Investimento total
- **ROI Médio**: Return on Investment
- **Taxa de Conversão**: Efetividade das campanhas

### Visualizações Geradas

1. **Evolução das Recompensas**: Linha temporal com preenchimento
2. **Evolução do Arrependimento**: Linha temporal com preenchimento
3. **Distribuição de Recompensas**: Gráfico de barras por iteração
4. **Correlação Recompensa vs Arrependimento**: Scatter plot

## 📄 Relatórios e Exportação

### Download de Resultados

- **CSV dos Resultados**: Tabela completa com todas as métricas
- **Relatório Completo**: Arquivo .txt com análise detalhada
- **Gráficos Interativos**: Visualizações no navegador

### Estrutura do Relatório

```
============================================================
RELATÓRIO DE EXPERIMENTO AUTORL
============================================================

CONFIGURAÇÃO DO EXPERIMENTO:
- Algoritmo: Epsilon-Greedy (Exploração vs Exploração)
- Otimizador: Algoritmo Genético (Seleção Natural)
- Gerações: 20
- Iterações: 3

ESTATÍSTICAS DOS DADOS:
- Total de Campanhas: 2
- Total de Dias: 30
- Gasto Total: $123,456
- Compras Médias: 45.67

RESULTADOS:
- Total de Execuções: 3
- Recompensa Média: 17,306.10
- Melhor Recompensa: 17,360.90
- Arrependimento Médio: 1,234.56

DETALHES DAS EXECUÇÕES:
Iteração 1:
  Recompensa: 17,306.10
  Arrependimento: 1,234.56
  Parâmetros: {"epsilon": 0.2163}
...
```

## 🧪 Testes e Validação

### Executar Testes

```bash
# Teste completo do framework
python test_evolutionary_optimization.py

# Teste de melhorias
python test_improvements.py
```

### Verificação de Funcionamento

```bash
# Verificar instalação
python check_install.py

# Executar testes unitários
python -m pytest tests/
```

## 🚀 Desenvolvimento

### Estrutura Modular

O framework foi projetado para ser facilmente extensível:

- **Novos Algoritmos**: Implementar `BaseRLAlgorithm`
- **Novos Otimizadores**: Implementar `BaseOptimizer`
- **Novas Métricas**: Adicionar ao `AutoRLEngine`

### Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature
3. Implemente suas mudanças
4. Adicione testes
5. Submeta um pull request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para detalhes.

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor, leia as diretrizes de contribuição antes de submeter pull requests.

## 📞 Suporte

Para dúvidas, problemas ou sugestões:

1. Abra uma issue no GitHub
2. Consulte a documentação
3. Execute os testes para verificar a instalação

---

**🎉 Framework AutoRL - Otimização Automática de Reinforcement Learning com Interface Web Avançada!**
