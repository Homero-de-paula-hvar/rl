# Conteúdo para autorl_framework/rl_algs/base.py

from abc import ABC, abstractmethod

class BaseRLAlgorithm(ABC):
    """
    Classe base abstrata para um algoritmo de Aprendizagem por Reforço
    no contexto de Multi-Armed Bandits. 
    """

    def __init__(self, n_arms: int):
        """
        Inicializa o algoritmo.

        Args:
            n_arms: O número de braços (ações) disponíveis. 
        """
        self.n_arms = n_arms

    @abstractmethod
    def select_arm(self) -> int:
        """
        Seleciona um braço para puxar com base na estratégia do algoritmo. 

        Returns:
            O índice do braço selecionado.
        """
        pass

    @abstractmethod
    def update(self, chosen_arm: int, reward: float):
        """
        Atualiza o conhecimento interno do algoritmo com base na recompensa observada. 

        Args:
            chosen_arm: O índice do braço que foi escolhido.
            reward: A recompensa recebida.
        """
        pass