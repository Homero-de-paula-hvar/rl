git # 🤖 Auto Reinforcement Learning Framework

Framework completo para experimentação, comparação e otimização automática de algoritmos de Multi-Armed Bandit (MAB) com interface web moderna, suporte a diversos otimizadores e logging detalhado.

---

## 🚀 Principais Recursos

- **Algoritmos MAB:** Epsilon-Greedy, UCB1, Thompson Sampling, A/B Test Agent
- **Otimizadores:** Algoritmo Genético, PSO, Differential Evolution, Grid Search, Random Search, Bayesian Optimization
- **Interface Web Flask:** Dashboard interativo para configuração, execução e visualização dos experimentos
- **Testes automáticos:** Script para rodar todos os algoritmos/otimizadores e salvar logs detalhados
- **Logging:** Cada execução gera um arquivo TXT com todo o histórico do experimento

---

## 📦 Estrutura do Projeto

```
framework-main/
├── autorl_framework/
│   ├── rl_algs/           # Algoritmos MAB
│   ├── optimizers/        # Otimizadores
│   └── simulation/        # Engine e simulador
├── data/                  # Datasets de exemplo
├── static/                # CSS/JS da interface web
├── templates/             # HTML da interface web
├── web_app.py             # Servidor Flask (interface web)
├── run_full_test.py       # Script de teste automático
├── requirements.txt       # Dependências
└── README.md
```

---

## ⚡ Instalação

### Pré-requisitos

- Python 3.10+ (recomendado)
- pip

### Passos

```bash
git clone https://github.com/Homero-de-paula-hvar/rl.git
cd rl/framework-main

python -m venv venv
venv\Scripts\activate      # Windows
# ou
source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

---

## 💻 Como Usar

### Interface Web

```bash
python web_app.py
```
Acesse: [http://localhost:8080](http://localhost:8080)

### Teste Automático de Todos os Algoritmos/Otimizadores

```bash
python run_full_test.py
```
- Gera um arquivo TXT com o log completo do teste na pasta do projeto.

---

## 🧠 Algoritmos e Otimizadores Disponíveis

### Algoritmos MAB
- **Epsilon-Greedy**
- **UCB1**
- **Thompson Sampling**
- **A/B Test Agent**

### Otimizadores
- **Genetic Algorithm**
- **Particle Swarm Optimization (PSO)**
- **Differential Evolution**
- **Grid Search**
- **Random Search**
- **Bayesian Optimization**

---

## 📊 Logging

- Cada execução do servidor web gera um arquivo `log_experimento_YYYYMMDD_HHMMSS.txt` com todo o histórico do experimento.
- O teste automático gera um arquivo `teste_automatico_YYYYMMDD_HHMMSS.txt` com os resultados de todos os algoritmos/otimizadores.

---

## 📚 Exemplos de Uso

### Usando a Engine em Python

```python
from autorl_framework.simulation.auto_rl_engine import AutoRLEngine
import pandas as pd

df_control = pd.read_csv('data/control_group.csv', sep=';')
df_test = pd.read_csv('data/test_group.csv', sep=';')
# Prepare os dados conforme o pipeline do projeto

df_full = pd.concat([df_control, df_test], ignore_index=True)
engine = AutoRLEngine(data=df_full, arms_col='campaign_name', reward_col='purchases_reward')
results, log_file = engine.run_full_test_and_save(n_iterations=3)
print(f"Resultados salvos em: {log_file}")
```

---

## 📝 Licença

UFOP