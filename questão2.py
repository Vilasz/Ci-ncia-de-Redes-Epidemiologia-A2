'''
Gere uma rede “livre de escala” com 10000 vértices, grau médio < 𝑘 >= 20 e
expoente 𝛾 = 2.5. Comece com 5 vértices infectados escolhidos aleatoriamente. Execute
múltiplas simulações da propagação da infecção pelo modelo SIS com os parâmetros abaixo e
compare com os resultados esperados. (sugestão: faça em torno de 100 simulações e descreva
o comportamento da epidemia “na média”)
a. 𝛽 = 0.01 e 𝜇 = 0.1
b. 𝛽 = 0.01 e 𝜇 = 0.2
c. 𝛽 = 0.01 e 𝜇 = 0.3
'''

import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
from joblib import Parallel, delayed

class EpidemiologyGraph:
    def __init__(self, n, k, gamma):
        self.n = n
        self.k = k
        self.gamma = gamma

    def generate_random_graph(self):
        # Gera os graus seguindo uma distribuição de potência
        degrees = nx.utils.powerlaw_sequence(self.n, self.gamma)

        # ajusta para ter grau médio desejado
        current_avg = np.mean(degrees)
        factor = self.k / current_avg
        degrees = [max(1, int(d * factor)) for d in degrees]

        if sum(degrees) % 2 == 1:
            degrees[-1] += 1

        # Aplica o processo descrito em 4.8 do Barabási
        G = nx.configuration_model([int(d) for d in degrees])
        G = nx.Graph(G)

        # Removendo os loops na mão. Não fara muita diferença para n grande
        G.remove_edges_from(nx.selfloop_edges(G))
        return G

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
        # --- MUDANÇA AQUI ---
        # Utilizamos o Parallel para rodar as simulações simultaneamente.
        # n_jobs=-1 instrui o computador a usar todos os núcleos disponíveis.
        # delayed(...) prepara a chamada da função para ser distribuída.
        
        all_infection_counts = Parallel(n_jobs=-1)(
            delayed(self.simulate_sis)(beta, mu, num_initial_infected, max_steps)
            for _ in tqdm(range(num_simulations), desc=f"Simulando β={beta}, μ={mu}")
        )
        # --------------------

        # Calcula a média das infecções ao longo do tempo (convertendo para array numpy primeiro)
        avg_infection_counts = np.mean(all_infection_counts, axis=0)

        # Plot das curvas (o resto do código segue igual)
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
    plt.savefig(f'infection_curve_scale_free_beta_{beta}_mu_{mu}.png')
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
    plt.savefig(f'all_infection_curves_scale_free_{beta}_{mu}.png')
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
    plt.savefig('all_avg_infection_curves_scale_free.png')
    plt.show()

if __name__ == "__main__":
    # Criar pasta para resultados
    os.makedirs('resultados_questao2', exist_ok=True)

    # Inicializar o grafo epidemiológico
    epi_graph = EpidemiologyGraph(n=10000, k=20, gamma=2.5)

    # Simulação 1
    print("Executando simulação 1 (β=0.01, μ=0.1)...")
    all_infection_counts_1 = epi_graph.run_simulations(
        beta=0.01, mu=0.1, num_initial_infected=5, max_steps=60, num_simulations=100
    )
    df1 = pd.DataFrame({'tempo': range(len(all_infection_counts_1)), 'infectados_media': all_infection_counts_1})
    df1.to_csv('resultados_questao2/simulacao_1_beta_0.01_mu_0.1.csv', index=False)
    plot_infection_curve(all_infection_counts_1, beta=0.01, mu=0.1)

    # Simulação 2
    print("Executando simulação 2 (β=0.01, μ=0.2)...")
    all_infection_counts_2 = epi_graph.run_simulations(
        beta=0.01, mu=0.2, num_initial_infected=5, max_steps=60, num_simulations=100
    )
    df2 = pd.DataFrame({'tempo': range(len(all_infection_counts_2)), 'infectados_media': all_infection_counts_2})
    df2.to_csv('resultados_questao2/simulacao_2_beta_0.01_mu_0.2.csv', index=False)
    plot_infection_curve(all_infection_counts_2, beta=0.01, mu=0.2)

    # Simulação 3
    print("Executando simulação 3 (β=0.01, μ=0.3)...")
    all_infection_counts_3 = epi_graph.run_simulations(
        beta=0.01, mu=0.3, num_initial_infected=5, max_steps=60, num_simulations=100
    )
    df3 = pd.DataFrame({'tempo': range(len(all_infection_counts_3)), 'infectados_media': all_infection_counts_3})
    df3.to_csv('resultados_questao2/simulacao_3_beta_0.01_mu_0.3.csv', index=False)
    plot_infection_curve(all_infection_counts_3, beta=0.01, mu=0.3)

    # Gráfico comparativo
    plot_all_avg_curves(
        [all_infection_counts_1, all_infection_counts_2, all_infection_counts_3],
        params=[(0.01, 0.1), (0.01, 0.2), (0.01, 0.3)]
    )

    print("Simulações concluídas! Resultados salvos na pasta 'resultados_questao2/'")