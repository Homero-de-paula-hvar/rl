# Conteúdo para autorl_framework/simulation/simulator.py

import pandas as pd
from autorl_framework.rl_algs.base import BaseRLAlgorithm
import numpy as np

class MABSimulator:
    """
    Executa uma simulação de Multi-Armed Bandit a partir de um dataset histórico.
    Replica o processo iterativo dia a dia, como descrito no guia. 
    """

    def __init__(self, data: pd.DataFrame, arms_col: str, reward_col: str):
        self.data = data
        self.arms_col = arms_col
        self.reward_col = reward_col
        
        self.arm_names = self.data[self.arms_col].unique()
        self.n_arms = len(self.arm_names)
        
        # Mapeia nomes de braços para índices
        self.arm_map = {name: i for i, name in enumerate(self.arm_names)}
        
        # Prepara os dados para consulta rápida
        # Pivota a tabela para que as colunas sejam os braços e as linhas os dias
        self.reward_history = self.data.pivot(index='date', columns=self.arms_col, values=self.reward_col)
        self.horizon = len(self.reward_history) # 

    def run(self, algorithm: BaseRLAlgorithm, run_name: str):
        """
        Executa uma simulação completa para um dado algoritmo.

        Args:
            algorithm: Uma instância de um algoritmo que herda de BaseRLAlgorithm.
            run_name: Um nome para esta execução (ex: "EpsilonGreedy").

        Returns:
            Um DataFrame do pandas com os resultados da simulação.
        """
        if algorithm.n_arms != self.n_arms:
            raise ValueError("O número de braços do algoritmo não corresponde aos dados.")

        results = []
        
        # Loop de Simulação: itera sobre cada dia do experimento 
        for t in range(self.horizon):
            # Definir contexto: bias + iteração
            context = np.array([1, t+1])
            # 1. O agente decide qual braço puxar
            if hasattr(algorithm, 'select_arm') and 'context' in algorithm.select_arm.__code__.co_varnames:
                chosen_arm_idx = algorithm.select_arm(context)
            else:
                chosen_arm_idx = algorithm.select_arm()
            
            # 2. Observamos a recompensa do braço escolhido no dataset histórico 
            reward = self.reward_history.iloc[t, chosen_arm_idx]
            
            # 3. O algoritmo atualiza seu conhecimento 
            if hasattr(algorithm, 'update') and 'context' in algorithm.update.__code__.co_varnames:
                algorithm.update(chosen_arm_idx, reward, context)
            else:
                algorithm.update(chosen_arm_idx, reward)
            
            # 4. Cálculo do arrependimento (regret)
            # Como temos feedback completo, sabemos a recompensa do melhor braço naquele dia 
            optimal_reward = self.reward_history.iloc[t].max()
            regret = optimal_reward - reward
            
            # 5. Registro de métricas 
            results.append({
                'run_name': run_name,
                'timestep': t + 1,
                'chosen_arm': self.arm_names[chosen_arm_idx],
                'reward': reward,
                'regret': regret
            })
            
        return pd.DataFrame(results)