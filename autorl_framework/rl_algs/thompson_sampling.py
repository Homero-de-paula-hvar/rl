import numpy as np
from autorl_framework.rl_algs.base import BaseRLAlgorithm

class ThompsonSamplingNormalGamma(BaseRLAlgorithm):
    """
    Implementação robusta do Thompson Sampling para recompensas contínuas.

    Este algoritmo modela a distribuição de recompensa de cada braço usando uma
    distribuição Normal com média e variância desconhecidas. Ele utiliza um
    prior conjugado Normal-Gamma para realizar a inferência bayesiana.

    O processo de amostragem é feito a partir da distribuição marginal posterior
    para a média, que é uma distribuição t de Student. Esta abordagem é mais
    robusta do que assumir uma variância conhecida, pois permite que o algoritmo
    aprenda e se adapte tanto à recompensa esperada quanto ao risco (variância)
    de cada braço.

    A implementação é vetorizada com NumPy para alta eficiência computacional.
    """

    def __init__(self, n_arms: int, mu_0: float = 0.0, kappa_0: float = 1e-6,
                 alpha_0: float = 1e-6, beta_0: float = 1e-6):
        """
        Inicializa o algoritmo Thompson Sampling com um prior Normal-Gamma.

        Args:
            n_arms: O número de braços (ações) no problema de bandit.
            mu_0: O prior para a média das recompensas.
            kappa_0: A confiança no prior da média (número de observações virtuais).
            alpha_0: O parâmetro de forma do prior Gamma para a precisão.
            beta_0: O parâmetro de taxa (escala) do prior Gamma para a precisão.
        """
        super().__init__(n_arms)
        # Hiperparâmetros do prior, vetorizados para cada braço
        self.mu = np.full(n_arms, mu_0, dtype=np.float64)
        self.kappa = np.full(n_arms, kappa_0, dtype=np.float64)
        self.alpha = np.full(n_arms, alpha_0, dtype=np.float64)
        self.beta = np.full(n_arms, beta_0, dtype=np.float64)

        # Estatísticas suficientes para as atualizações
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.sum_rewards = np.zeros(n_arms, dtype=np.float64)
        self.sum_squared_rewards = np.zeros(n_arms, dtype=np.float64)

    def select_arm(self) -> int:
        """
        Seleciona um braço para jogar usando amostragem de Thompson.

        Amostra um valor da distribuição marginal posterior da média (uma t de Student)
        para cada braço e seleciona o braço com a maior amostra.

        Returns:
            O índice do braço selecionado.
        """
        # Evitar avisos de divisão por zero para braços não jogados onde alpha ou kappa são zero
        # Usamos valores padrão que resultarão em uma variância muito grande, promovendo a exploração
        safe_alpha = np.where(self.alpha > 0, self.alpha, 1e-6)
        safe_kappa = np.where(self.kappa > 0, self.kappa, 1e-6)

        # Graus de liberdade para a distribuição t de Student
        degrees_of_freedom = 2 * safe_alpha

        # Parâmetros de localização (loc) e escala (scale) da t de Student
        loc = self.mu
        scale = np.sqrt(self.beta / (safe_alpha * safe_kappa))

        # Amostrar da distribuição t de Student padrão e depois escalar
        # np.random.standard_t espera os graus de liberdade como argumento
        sampled_theta = loc + np.random.standard_t(degrees_of_freedom, size=self.n_arms) * scale

        # Escolher o braço com o maior valor amostrado
        return int(np.argmax(sampled_theta))

    def update(self, chosen_arm: int, reward: float):
        """
        Atualiza os parâmetros do braço escolhido com base na recompensa observada.

        Args:
            chosen_arm: O índice do braço que foi jogado.
            reward: A recompensa recebida.
        """
        # Obter os parâmetros a priori para o braço escolhido
        mu_0 = self.mu[chosen_arm]
        kappa_0 = self.kappa[chosen_arm]
        alpha_0 = self.alpha[chosen_arm]
        beta_0 = self.beta[chosen_arm]

        # Atualizar estatísticas suficientes
        self.counts[chosen_arm] += 1
        self.sum_rewards[chosen_arm] += reward
        self.sum_squared_rewards[chosen_arm] += reward**2

        n = self.counts[chosen_arm]
        sum_r = self.sum_rewards[chosen_arm]
        sum_r_sq = self.sum_squared_rewards[chosen_arm]

        # Atualizar os hiperparâmetros posteriores para o braço escolhido
        # Estas são as atualizações para um prior Normal-Gamma com n=1 observação
        # e dados (recompensa). Para evitar recálculos, usamos as estatísticas
        # suficientes acumuladas.

        # Média e soma dos desvios quadrados dos dados observados até agora
        if n > 0:
            mean_reward = sum_r / n
            # Sum of squared deviations from the sample mean
            sum_sq_dev = sum_r_sq - n * (mean_reward**2)
        else: # n=0, caso que não deveria ocorrer aqui, mas por segurança
            mean_reward = 0.0
            sum_sq_dev = 0.0

        # Atualizações posteriores
        kappa_n = kappa_0 + n
        mu_n = (kappa_0 * mu_0 + sum_r) / kappa_n
        alpha_n = alpha_0 + n / 2
        beta_n = beta_0 + 0.5 * sum_sq_dev + (kappa_0 * n * (mean_reward - mu_0)**2) / (2 * kappa_n)

        # Armazenar os novos parâmetros posteriores
        self.mu[chosen_arm] = mu_n
        self.kappa[chosen_arm] = kappa_n
        self.alpha[chosen_arm] = alpha_n
        self.beta[chosen_arm] = beta_n

ThompsonSampling = ThompsonSamplingNormalGamma