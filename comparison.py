import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Parâmetros globais (iguais usados nas questões 1, 2 e 3)
# -------------------------------------------------------------------
N = 10000          # número de vértices
K_MEDIO = 20       # grau médio

# -------------------------------------------------------------------
# Funções utilitárias
# -------------------------------------------------------------------

def load_infection_series(csv_path):
    """
    Lê um CSV gerado nas questões 1 ou 2 e devolve:
        t: array de tempos
        i: array com número de infectados (média sobre simulações)
    Tenta automaticamente usar as colunas 'infectados_media' ou 'infectados'.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)

    if 'infectados_media' in df.columns:
        y = df['infectados_media'].values
    elif 'infectados' in df.columns:
        y = df['infectados'].values
    else:
        raise ValueError(
            f"Colunas de infectados não encontradas em {csv_path}. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    if 'tempo' in df.columns:
        t = df['tempo'].values
    else:
        t = np.arange(len(y))

    return t, y


def sis_R0(beta, mu, k=K_MEDIO):
    """
    Calcula R0 no modelo de campo médio homogêneo:
        R0 = beta * k / mu
    """
    return beta * k / mu


def sis_prevalencia_teorica(beta, mu, k=K_MEDIO):
    """
    Prevalência endêmica aproximada no modelo SIS homogêneo:
        i* = 1 - mu / (beta * k), se R0 > 1
        i* = 0, caso contrário
    """
    R0 = sis_R0(beta, mu, k)
    if R0 <= 1:
        return 0.0
    return 1.0 - (mu / (beta * k))


def resumo_sis(nome_questao, tipo_rede, cenario, beta, mu, csv_path):
    """
    Carrega resultados de um cenário, calcula R0, prevalência simulada e teórica
    e devolve um dicionário com o resumo.
    """
    t, infectados = load_infection_series(csv_path)
    fracao_infectados = infectados / N

    # média na segunda metade da simulação como proxy da prevalência endêmica
    metade = len(fracao_infectados) // 2
    preval_simulada = fracao_infectados[metade:].mean()

    R0 = sis_R0(beta, mu, K_MEDIO)
    preval_teorica = sis_prevalencia_teorica(beta, mu, K_MEDIO)

    return {
        "questao": nome_questao,
        "rede": tipo_rede,
        "cenario": cenario,
        "beta": beta,
        "mu": mu,
        "R0": R0,
        "R0_maior_1": R0 > 1.0,
        "prevalencia_teorica_frac": preval_teorica,
        "prevalencia_simulada_frac": preval_simulada,
        "csv": csv_path,
    }, (t, fracao_infectados)


# -------------------------------------------------------------------
# Caminhos esperados para os resultados das questões 1, 2 e 3
# (ajuste se você tiver usado nomes de arquivos diferentes)
# -------------------------------------------------------------------

# Questão 1 – rede ER
Q1_CENARIOS = [
    # cenario, beta, mu, caminho_csv
    ("a", 0.02, 0.1, "resultados/simulacao_1_beta_0.02_mu_0.1.csv"),
    ("b", 0.02, 0.4, "resultados/simulacao_2_beta_0.02_mu_0.4.csv"),
    ("c", 0.02, 0.5, "resultados/simulacao_3_beta_0.02_mu_0.5.csv"),
]

# Questão 2 – rede livre de escala
Q2_CENARIOS = [
    ("a", 0.01, 0.1, "resultados_questao2/simulacao_1_beta_0.01_mu_0.1.csv"),
    ("b", 0.01, 0.2, "resultados_questao2/simulacao_2_beta_0.01_mu_0.2.csv"),
    ("c", 0.01, 0.3, "resultados_questao2/simulacao_3_beta_0.01_mu_0.3.csv"),
]

# Questão 3 – imunização (resultados do arquivo que geramos antes)
Q3_ARQUIVOS = [
    ("random",    "Imunização aleatória",
     "resultados_questao3/resultados_random.csv"),
    ("hubs",      "Imunização de hubs (maior grau)",
     "resultados_questao3/resultados_hubs.csv"),
    ("neighbors", "Imunização de vizinhos aleatórios",
     "resultados_questao3/resultados_neighbors.csv"),
]


# -------------------------------------------------------------------
# Rotinas específicas para cada questão
# -------------------------------------------------------------------

def analisar_questoes_1_e_2():
    """
    Lê os CSVs das questões 1 e 2, calcula R0, prevalência simulada e teórica
    e gera:
      - um CSV resumo_q1_q2.csv com uma tabela comparando tudo;
      - gráficos das curvas de infecção para cada rede;
      - um gráfico comparando ER x livre de escala para os casos de R0 > 1.
    """
    resumos = []
    curvas_ER = []
    curvas_SF = []

    # Questão 1 – ER
    for cenario, beta, mu, csv_path in Q1_CENARIOS:
        resumo, (t, frac_inf) = resumo_sis(
            nome_questao="1",
            tipo_rede="Erdős-Rényi (ER)",
            cenario=cenario,
            beta=beta,
            mu=mu,
            csv_path=csv_path,
        )
        resumos.append(resumo)
        curvas_ER.append((cenario, beta, mu, t, frac_inf))

    # Questão 2 – livre de escala
    for cenario, beta, mu, csv_path in Q2_CENARIOS:
        resumo, (t, frac_inf) = resumo_sis(
            nome_questao="2",
            tipo_rede="Livre de escala",
            cenario=cenario,
            beta=beta,
            mu=mu,
            csv_path=csv_path,
        )
        resumos.append(resumo)
        curvas_SF.append((cenario, beta, mu, t, frac_inf))

    # Monta DataFrame de resumo
    df_resumo = pd.DataFrame(resumos)
    # Converte frações para porcentagem para ficar mais legível
    df_resumo["prevalencia_teorica_%"] = 100 * df_resumo["prevalencia_teorica_frac"]
    df_resumo["prevalencia_simulada_%"] = 100 * df_resumo["prevalencia_simulada_frac"]

    os.makedirs("resultados_comparacao", exist_ok=True)
    df_resumo.to_csv("resultados_comparacao/resumo_q1_q2.csv", index=False)

    print("\n================ RESUMO QUESTÕES 1 E 2 ================")
    print(
        df_resumo[
            [
                "questao", "rede", "cenario",
                "beta", "mu",
                "R0", "R0_maior_1",
                "prevalencia_teorica_%", "prevalencia_simulada_%"
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:6.2f}")
    )

    # Gráfico 1: curvas de infecção na rede ER (questão 1)
    plt.figure(figsize=(10, 6))
    for cenario, beta, mu, t, frac_inf in curvas_ER:
        plt.plot(
            t,
            frac_inf,
            label=f"Q1 {cenario}) β={beta}, μ={mu}, R0={sis_R0(beta, mu):.2f}"
        )
        # linha horizontal com prevalência teórica (se >0)
        i_teo = sis_prevalencia_teorica(beta, mu)
        if i_teo > 0:
            plt.axhline(i_teo, linestyle="--", linewidth=1)
    plt.xlabel("Tempo")
    plt.ylabel("Fração de infectados")
    plt.title("Questão 1 – Rede ER: curvas de infecção e previsão de campo médio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("resultados_comparacao/q1_curvas_ER.png")

    # Gráfico 2: curvas de infecção na rede livre de escala (questão 2)
    plt.figure(figsize=(10, 6))
    for cenario, beta, mu, t, frac_inf in curvas_SF:
        plt.plot(
            t,
            frac_inf,
            label=f"Q2 {cenario}) β={beta}, μ={mu}, R0={sis_R0(beta, mu):.2f}"
        )
        i_teo = sis_prevalencia_teorica(beta, mu)
        if i_teo > 0:
            plt.axhline(i_teo, linestyle="--", linewidth=1)
    plt.xlabel("Tempo")
    plt.ylabel("Fração de infectados")
    plt.title("Questão 2 – Rede livre de escala: curvas de infecção e previsão de campo médio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("resultados_comparacao/q2_curvas_livre_escala.png")

    # Gráfico 3: comparação ER x livre de escala para os cenários de R0 > 1
    plt.figure(figsize=(10, 6))
    for resumo in resumos:
        if not resumo["R0_maior_1"]:
            # só faz sentido comparar casos onde a previsão de campo médio
            # indica estado endêmico
            continue
        scen = resumo["cenario"]
        # Descobre de qual lista vem (ER ou SF) para recuperar a curva
        if resumo["rede"].startswith("Erdős"):
            lista = curvas_ER
            label_prefix = "ER"
        else:
            lista = curvas_SF
            label_prefix = "Livre de escala"

        for (cenario, beta, mu, t, frac_inf) in lista:
            if (
                cenario == scen and
                abs(beta - resumo["beta"]) < 1e-9 and
                abs(mu - resumo["mu"]) < 1e-9
            ):
                plt.plot(
                    t,
                    frac_inf,
                    label=f"{label_prefix} Q{resumo['questao']}{cenario}) β={beta}, μ={mu}"
                )
                break

    plt.xlabel("Tempo")
    plt.ylabel("Fração de infectados")
    plt.title("Comparação de curvas – casos com R0 > 1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("resultados_comparacao/comparacao_ER_vs_livre_escala_R0_maior_1.png")

    print("\nArquivos gerados em 'resultados_comparacao/':")
    print("  - resumo_q1_q2.csv")
    print("  - q1_curvas_ER.png")
    print("  - q2_curvas_livre_escala.png")
    print("  - comparacao_ER_vs_livre_escala_R0_maior_1.png")


def analisar_questao_3():
    """
    Lê os CSVs da questão 3 (imunização) e calcula, para cada estratégia:
      - a menor fração de vértices imunizados que leva a prevalência < 1%;
      - gera um gráfico de prevalência vs fração imunizada com os três casos.
    Isso ajuda a discutir as estratégias de imunização na comparação final.
    """
    resultados_q3 = []
    curvas_q3 = []

    for key, descricao, csv_path in Q3_ARQUIVOS:
        if not os.path.exists(csv_path):
            print(f"[AVISO] Arquivo da questão 3 não encontrado: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if not {"frac_imunizados", "prevalencia_media"}.issubset(df.columns):
            raise ValueError(
                f"Arquivo {csv_path} não contém as colunas esperadas "
                f"'frac_imunizados' e 'prevalencia_media'."
            )

        fracs = df["frac_imunizados"].values
        prevs = df["prevalencia_media"].values  # já é fração de infectados

        curvas_q3.append((descricao, fracs, prevs))

        limiar = None
        for f, p in zip(fracs, prevs):
            if p < 0.01:   # menos de 1% de infectados
                limiar = f
                break

        resultados_q3.append({
            "estrategia": key,
            "descricao": descricao,
            "limiar_frac_imunizados": limiar,
            "limiar_%_imunizados": None if limiar is None else 100 * limiar,
        })

    if resultados_q3:
        df_q3 = pd.DataFrame(resultados_q3)
        os.makedirs("resultados_comparacao", exist_ok=True)
        df_q3.to_csv("resultados_comparacao/resumo_q3_imunizacao.csv", index=False)

        print("\n================ RESUMO QUESTÃO 3 (IMUNIZAÇÃO) ================")
        print(df_q3.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

        # Gráfico: prevalência vs fração de imunizados
        plt.figure(figsize=(10, 6))
        for descricao, fracs, prevs in curvas_q3:
            plt.plot(fracs, prevs, marker="o", label=descricao)
        plt.xlabel("Fração de vértices imunizados")
        plt.ylabel("Prevalência média (fração de infectados)")
        plt.title("Questão 3 – Prevalência vs fração de vértices imunizados")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("resultados_comparacao/q3_prevalencia_vs_imunizados.png")

        print("  - resumo_q3_imunizacao.csv")
        print("  - q3_prevalencia_vs_imunizados.png")
    else:
        print(
            "\nNenhum arquivo válido da questão 3 foi encontrado; "
            "verifique se 'resultados_questao3/' contém os CSVs esperados."
        )


def main():
    analisar_questoes_1_e_2()
    analisar_questao_3()


if __name__ == "__main__":
    main()
