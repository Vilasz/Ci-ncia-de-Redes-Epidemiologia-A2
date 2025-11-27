import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
from joblib import Parallel, delayed

# Parâmetros globais do problema (mesmos da questão 2, letra a)
N = 10000
K_MEDIO = 20
GAMMA = 2.5
BETA = 0.01
MU = 0.1

def generate_scale_free_graph(n, k_medio, gamma):
    """
    Gera uma rede livre de escala com n vértices, grau médio aproximado k_medio
    e expoente gamma, usando o configuration_model como na questão 2.
    """
    degrees = nx.utils.powerlaw_sequence(n, gamma)
    current_avg = np.mean(degrees)
    fator = k_medio / current_avg
    degrees = [max(1, int(d * fator)) for d in degrees]

    if sum(degrees) % 2 == 1:
        degrees[-1] += 1

    G = nx.configuration_model(degrees)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def choose_immunized_nodes(G, frac_imunizados, strategy):
    """
    Retorna o conjunto de vértices imunizados de acordo com a estratégia.
    strategy: 'random', 'hubs' ou 'neighbors'
    """
    n = G.number_of_nodes()
    num_imunizados = int(frac_imunizados * n)
    nodes = list(G.nodes())

    if num_imunizados == 0:
        return set()

    if strategy == 'random':
        # a) vértices imunizados escolhidos aleatoriamente
        return set(random.sample(nodes, num_imunizados))

    elif strategy == 'hubs':
        # b) imuniza vértices de maior grau
        graus = dict(G.degree())
        ordenados = sorted(graus.items(), key=lambda x: x[1], reverse=True)
        imunizados = [node for node, _ in ordenados[:num_imunizados]]
        return set(imunizados)

    elif strategy == 'neighbors':
        # c) imuniza vizinhos de vértices escolhidos aleatoriamente
        imunizados = set()
        while len(imunizados) < num_imunizados:
            v = random.choice(nodes)
            vizinhos = list(G.neighbors(v))
            if not vizinhos:
                continue
            u = random.choice(vizinhos)
            imunizados.add(u)
        return imunizados
    else:
        raise ValueError("Estratégia desconhecida: use 'random', 'hubs' ou 'neighbors'.")

def simulate_sis_with_immunization(G, beta, mu, immune_nodes,
                                   num_initial_infected=5,
                                   max_steps=200):
    """
    Simulação SIS em uma rede G com um conjunto de nós previamente imunizados.
    Estados:
      -1: imunizado (não pode ser infectado)
       0: suscetível
       1: infectado
    """
    n = G.number_of_nodes()
    states = np.zeros(n, dtype=np.int8)

    # marca imunizados
    immune_mask = np.zeros(n, dtype=bool)
    for node in immune_nodes:
        immune_mask[node] = True
        states[node] = -1  # apenas para marcar

    # escolhe infectados iniciais apenas entre suscetíveis
    suscetiveis_iniciais = [node for node in G.nodes() if not immune_mask[node]]
    if len(suscetiveis_iniciais) < num_initial_infected:
        raise ValueError("Não há suscetíveis suficientes para iniciar a simulação.")
    initial_infected = random.sample(suscetiveis_iniciais, num_initial_infected)
    for node in initial_infected:
        states[node] = 1

    neighbors = {node: list(G.neighbors(node)) for node in G.nodes()}
    infected_counts = []

    for step in range(max_steps):
        new_states = states.copy()
        for node in G.nodes():
            if states[node] == 1:
                # tenta infectar vizinhos suscetíveis (não imunizados)
                for neighbor in neighbors[node]:
                    if (not immune_mask[neighbor]) and states[neighbor] == 0:
                        if random.random() < beta:
                            new_states[neighbor] = 1
                # tentativa de recuperação
                if random.random() < mu:
                    new_states[node] = 0
        states = new_states
        infected_counts.append(np.sum(states == 1))

    return infected_counts

def run_single_simulation(G, beta, mu, immune_nodes, max_steps):
    counts = simulate_sis_with_immunization(
        G, beta, mu, immune_nodes, 
        num_initial_infected=5, 
        max_steps=max_steps
    )
    tail = counts[max_steps // 2:]
    # Retorna o valor final calculado
    return np.mean(tail) / (G.number_of_nodes() - len(immune_nodes))

def run_experiments_for_strategy(strategy, frac_list, num_simulations=50, max_steps=200):
    results = []
    G_base = generate_scale_free_graph(N, K_MEDIO, GAMMA)
    
    
    for frac in tqdm(frac_list, desc=f"Estratégia {strategy}"):
        immune_nodes = choose_immunized_nodes(G_base, frac, strategy)
        finais = Parallel(n_jobs=-1)(
            delayed(run_single_simulation)(G_base, BETA, MU, immune_nodes, max_steps)
            for _ in range(num_simulations)
        )
        
        prevalencia_media = np.mean(finais)
        results.append((frac, prevalencia_media))
    return results

def estimate_threshold(results, epsilon=0.01):
    """
    Dada a lista results = [(frac, prevalencia_media), ...],
    retorna a menor fração de imunizados para a qual a
    prevalência média endêmica é menor que epsilon.
    """
    for frac, prev in results:
        if prev < epsilon:
            return frac
    return None

def main():
    os.makedirs("resultados_questao3", exist_ok=True)

    frac_list = [i / 20 for i in range(0, 13)]

    estrategias = {
        "random": "Imunização aleatória",
        "hubs": "Imunização de hubs",
        "neighbors": "Imunização de vizinhos de vértices aleatórios"
    }

    todos_resultados = {}

    for key in estrategias:
        resultados = run_experiments_for_strategy(
            strategy=key,
            frac_list=frac_list,
            num_simulations=50,
            max_steps=200
        )
        todos_resultados[key] = resultados

        df = pd.DataFrame(resultados, columns=["frac_imunizados", "prevalencia_media"])
        df.to_csv(f"resultados_questao3/resultados_{key}.csv", index=False)

        # estima limiar
        limiar = estimate_threshold(resultados, epsilon=0.01)
        if limiar is not None:
            num_vertices = int(limiar * N)
            print(f"Estratégia: {estrategias[key]}")
            print(f"  Fração mínima aproximada de imunizados: {limiar:.2f}")
            print(f"  Número aproximado de vértices imunizados: {num_vertices}")
        else:
            print(f"Estratégia: {estrategias[key]}")
            print("  Não foi encontrado limiar dentro do intervalo de frações testado.")

    plt.figure(figsize=(10, 6))
    for key, label in estrategias.items():
        fracs = [x[0] for x in todos_resultados[key]]
        prevs = [x[1] for x in todos_resultados[key]]
        plt.plot(fracs, prevs, marker="o", label=label)
    plt.xlabel("Fração de vértices imunizados")
    plt.ylabel("Prevalência média endêmica (fração de infectados)")
    plt.title("Questão 3 - Efeito da imunização na fixação do estado endêmico (β = 0.01, μ = 0.1)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("resultados_questao3/prevalencia_vs_imunizados.png")
    plt.show()

if __name__ == "__main__":
    main()
