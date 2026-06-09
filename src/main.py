import sys
from pathlib import Path

# Ajuste de path para encontrar o utils
caminho_raiz = Path(__file__).resolve().parent.parent
if str(caminho_raiz) not in sys.path:
    sys.path.append(str(caminho_raiz))

# Gerais
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.print_customizado import cprint

# Métodos
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from src.metricas import entropia, coesao, separacao


# Métricas
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import rand_score, adjusted_rand_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix


def ler_dados():
    
    caminho_atual = Path(__file__).resolve()
    raiz = caminho_atual.parent.parent
    caminho_dados = raiz / "dados_raw" / "dados.csv"

    dados = pd.read_csv(caminho_dados)

    return dados

def printar_metricas(metricas, modelo):
    cprint(f"Métricas para o modelo: {modelo}", label="METRICAS")
    print("-" * 45)
    print(f"{'Métrica':<30} | {'Valor':<10}")
    print("-" * 45)
    for metrica, valor in metricas.items():
        if valor is not None:
            print(f"{metrica.replace('_', ' ').capitalize():<30} | {valor:.4f}")
        else:
            print(f"{metrica.replace('_', ' ').capitalize():<30} | N/A")
    print("-" * 45 + "\n")

def plotagraficos (X : pd.DataFrame, opiniao, modelo):

    nome_coluna1 = X.columns[0]
    nome_coluna2 = X.columns[1]
    nome_colunaClass = X.columns[-1]

    coluna1 = X[nome_coluna1]
    coluna2 = X[nome_coluna2]
    colunaClass = X[nome_colunaClass]
    
    # Criar diretório de resultados se não existir
    caminho_resultados = Path(__file__).resolve().parent.parent / "resultados"
    caminho_resultados.mkdir(exist_ok=True)

    f,(ax1,ax2)=plt.subplots(1,2,sharey=True,figsize=(20,7.5))
    ax1.set_title("Original")
    ax1.scatter(coluna1,coluna2,c=colunaClass,cmap="rainbow")
    ax2.set_title(label=modelo)
    ax2.scatter(coluna1,coluna2,c=opiniao,cmap="rainbow")
    
    # Salvar o gráfico
    nome_arquivo = f"resultado_{modelo.lower().replace('-', '_')}.png"
    plt.savefig(caminho_resultados / nome_arquivo)
    cprint(f"Gráfico salvo em: {caminho_resultados / nome_arquivo}", label="PLOT")
    
    plt.close(f) # Fecha a figura para liberar memória

def main():

    modelos = {}

    # 1. Ler dados
    cprint("Lendo dados...")
    dados = ler_dados()
    cprint(f"Dados carregados: {dados.shape[0]} amostras, {dados.shape[1]} colunas.")

    # Separação entre treino e classes reais
    X = dados.iloc[:, :-1]
    Y = dados.iloc[:, -1]

    # Rodando algoritmos

# ── KMEANS ────────────────────────────────────────────────────────────────────
    maior_ARI = -1
    melhor_modelo_kmeans = None
    for n_clusters in range(2, 10):
        for max_iter in [2, 5, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300]:
            modelo = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42, n_init=10)
            modelo.fit(X)
            ARI = adjusted_rand_score(Y, modelo.labels_)
            if ARI > maior_ARI:
                maior_ARI = ARI
                melhor_modelo_kmeans = modelo

    cprint(f"Melhor configuração: n_clusters={melhor_modelo_kmeans.n_clusters}, max_iter={melhor_modelo_kmeans.max_iter}, ARI={maior_ARI:.4f}", label="KMEANS")
    modelos["K-Means"] = melhor_modelo_kmeans

# ── AGNES ─────────────────────────────────────────────────────────────────────
    maior_ARI = -1
    melhor_modelo_agnes = None
    for n_clusters in range(2, 10):
        for linkage in ['ward', 'complete', 'average', 'single']:
            modelo = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            modelo.fit(X)
            ARI = adjusted_rand_score(Y, modelo.labels_)
            if ARI > maior_ARI:
                maior_ARI = ARI
                melhor_modelo_agnes = modelo

    cprint(f"Melhor configuração: n_clusters={melhor_modelo_agnes.n_clusters}, linkage={melhor_modelo_agnes.linkage}, ARI={maior_ARI:.4f}", label="AGNES")
    modelos["AGNES"] = melhor_modelo_agnes

# ── DBSCAN ────────────────────────────────────────────────────────────────────
    maior_ARI = -1
    melhor_modelo_dbscan = None
    for eps in range(1, 200, 1): # eps in range (0.1 ate 2.0)
        eps = eps/100
        for min_samples in range(5, 30, 1):
            modelo = DBSCAN(eps=eps, min_samples=min_samples)
            modelo.fit(X)
            ARI = adjusted_rand_score(Y, modelo.labels_)
            if ARI > maior_ARI:
                maior_ARI = ARI
                melhor_modelo_dbscan = modelo

    cprint(f"Melhor configuração: eps={melhor_modelo_dbscan.eps}, min_samples={melhor_modelo_dbscan.min_samples}, ARI={maior_ARI:.4f}", label="DBSCAN")
    modelos["DBSCAN"] = melhor_modelo_dbscan

    for nome, modelo in modelos.items():
        cprint(f"Rodando {nome}...", label=nome.upper())
        
        # Treino e Predição
        modelo.fit(X)
        opiniao = modelo.labels_

        # Cálculo de Métricas
        metricas = {
            # Coesao, Separação
            "homogeneidade": homogeneity_score(Y, opiniao),
            "completude": completeness_score(Y, opiniao),
            "v_measure": v_measure_score(Y, opiniao),
            "indice_randomico": rand_score(Y, opiniao),
            "indice_randomico_ajustado": adjusted_rand_score(Y, opiniao),
            "silhueta": silhouette_score(X, opiniao),
            "1 - entropia": 1 - entropia(Y, opiniao),          #None # implementar
            "1 - coesao": 1 - coesao(X, opiniao),                    #None # implementar
            "separacao": None # implementar
        }

        # Exibição
        printar_metricas(metricas, nome)
        plotagraficos(dados, opiniao, nome)

    cprint("Processamento concluído.")

if __name__ == "__main__":
    main()