'''
Gere uma rede aleatória (ER) com 10000 vértices e grau médio < 𝑘 >= 20. Comece
com 5 vértices infectados escolhidos aleatoriamente. Execute múltiplas simulações da
propagação da infecção pelo modelo SIS com os parâmetros abaixo e compare com os
resultados esperados. (sugestão: faça em torno de 100 simulações e descreva o
comportamento da epidemia “na média”)

a. beta = 0.02 e mi = 0.1
b. beta = 0.02 e mi = 0.4
c. beta = 0.02 e mi = 0.5

'''

import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd

class EpidemiologyGraph:
    def __init__(self, n, k):
        self.n = n
        self.k = k

    def generate_random_graph(self):
        p = self.k / (self.n - 1)
        return nx.erdos_renyi_graph(self.n, p)

    def simulate_sis(self, beta, mu, num_initial_infected, max_steps):
        # Gera um novo grafo
        G = self.generate_random_graph()

        # Inicializa os estados dos nós
        states = np.zeros(self.n, dtype=np.int8)

        # Escolhe nós iniciais infectados
        initial_infected = random.sample(list(G.nodes()), num_initial_infected)
        for node in initial_infected:
            states[node] = 1

        infection_counts = []

        neighbors = {node: list(G.neighbors(node)) for node in G.nodes()}

        # Faz a simulação no grafo
        for step in range(max_steps):
            new_states = states.copy()
            for node in G.nodes():
                if states[node] == 1:
                    # Tenta infectar vizinhos
                    for neighbor in neighbors[node]:
                        if states[neighbor] == 0 and random.random() < beta:
                            new_states[neighbor] = 1

                    # Tenta recuperar
                    if random.random() < mu:
                        new_states[node] = 0

            states = new_states
            infection_counts.append(sum(1 for state in states if state == 1))

        return infection_counts

    def run_simulations(self, beta, mu, num_initial_infected, max_steps, num_simulations):
        all_infection_counts = []
        for _ in tqdm(range(num_simulations)):
            infection_counts = self.simulate_sis(beta, mu, num_initial_infected, max_steps)
            all_infection_counts.append(infection_counts)

        # Calcula a média das infecções ao longo do tempo
        avg_infection_counts = np.mean(all_infection_counts, axis=0)

        plot_all_curves(all_infection_counts, avg_infection_counts, beta, mu)

        return avg_infection_counts

def plot_infection_curve(infection_counts, beta, mu):
    plt.figure(figsize=(10, 6))
    plt.plot(infection_counts, label=f'β={beta}, μ={mu}')
    plt.xlabel('Tempo')
    plt.ylabel('Número médio de infectados (100 simulações)')
    plt.title(f'Curva de Infecção Média ao Longo do Tempo com β={beta}, μ={mu}')
    plt.legend()
    plt.grid()
    plt.savefig(f'infection_curve_beta_{beta}_mu_{mu}.png')
    plt.show()

def plot_all_curves(curves, avg_curve, beta, mu):
    plt.figure(figsize=(10, 6))
    for i, infection_counts in enumerate(curves):
        plt.plot(infection_counts, alpha=0.3, linewidth=0.5, color='gray')
    plt.plot(avg_curve, label='Média', color='red', linewidth=2)
    plt.xlabel('Tempo')
    plt.ylabel('Número de infectados')
    plt.title(f'Todas as simulações com β={beta}, μ={mu}')
    plt.grid()
    plt.savefig(f'all_infection_curves_{beta}_{mu}.png')
    plt.show()

def plot_all_avg_curves(avg_curves, params):
    plt.figure(figsize=(10, 6))
    for (beta, mu), avg_curve in zip(params, avg_curves):
        plt.plot(avg_curve, label=f'β={beta}, μ={mu}')
    plt.xlabel('Tempo')
    plt.ylabel('Número médio de infectados')
    plt.title('Curvas de Infecção Média ao Longo do Tempo')
    plt.legend()
    plt.grid()
    plt.savefig('all_avg_infection_curves.png')
    plt.show()

# Criar pasta para resultados
os.makedirs('resultados', exist_ok=True)

# Inicializar o grafo epidemiológico
epi_graph = EpidemiologyGraph(n=10000, k=20)

# Simulação 1
print("Executando simulação 1 (β=0.02, μ=0.1)...")
all_infection_counts_1 = epi_graph.run_simulations(beta=0.02, mu=0.1, num_initial_infected=5, max_steps=60, num_simulations=100)
df1 = pd.DataFrame({'tempo': range(len(all_infection_counts_1)), 'infectados_media': all_infection_counts_1})
df1.to_csv('resultados/simulacao_1_beta_0.02_mu_0.1.csv', index=False)
plot_infection_curve(all_infection_counts_1, beta=0.02, mu=0.1)

# Simulação 2
print("Executando simulação 2 (β=0.02, μ=0.4)...")
all_infection_counts_2 = epi_graph.run_simulations(beta=0.02, mu=0.4, num_initial_infected=5, max_steps=60, num_simulations=100)
df2 = pd.DataFrame({'tempo': range(len(all_infection_counts_2)), 'infectados': all_infection_counts_2})
df2.to_csv('resultados/simulacao_2_beta_0.02_mu_0.4.csv', index=False)
plot_infection_curve(all_infection_counts_2, beta=0.02, mu=0.4)

# Simulação 3
print("Executando simulação 3 (β=0.02, μ=0.5)...")
all_infection_counts_3 = epi_graph.run_simulations(beta=0.02, mu=0.5, num_initial_infected=5, max_steps=60, num_simulations=100)
df3 = pd.DataFrame({'tempo': range(len(all_infection_counts_3)), 'infectados': all_infection_counts_3})
df3.to_csv('resultados/simulacao_3_beta_0.02_mu_0.5.csv', index=False)
plot_infection_curve(all_infection_counts_3, beta=0.02, mu=0.5)

# Gráfico comparativo
plot_all_avg_curves(
    [all_infection_counts_1, all_infection_counts_2, all_infection_counts_3],
    params=[(0.02, 0.1), (0.02, 0.4), (0.02, 0.5)]
)

print("Simulações concluídas! Resultados salvos na pasta 'resultados/'")
