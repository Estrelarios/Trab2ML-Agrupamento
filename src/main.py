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


# Métricas
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import rand_score, adjusted_rand_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.stats import entropy

def ler_dados():
    
    caminho_atual = Path(__file__).resolve()
    raiz = caminho_atual.parent.parent
    caminho_dados = raiz / "dados_raw" / "Banana.csv"

    dados = pd.read_csv(caminho_dados)

    return dados

def data_augmentation(dados, fator_escala=1):
    """Aumenta o dataset aplicando ruído aleatório entre 1% e 10%"""
    if fator_escala <= 1:
        return dados
    
    novos_dados_lista = [dados]
    atributos = dados.columns[:-1]
    
    for _ in range(fator_escala - 1):
        dados_ruido = dados.copy()
        for col in atributos:
            ruido_percentual = np.random.uniform(0.01, 0.10, size=len(dados_ruido))
            direcao = np.random.choice([-1, 1], size=len(dados_ruido))
            dados_ruido[col] = dados_ruido[col] * (1 + (direcao * ruido_percentual))
        novos_dados_lista.append(dados_ruido)
        
    return pd.concat(novos_dados_lista, ignore_index=True)

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

def plotagraficos (dados_treino : pd.DataFrame, opiniao, modelo):

    nome_coluna1 = dados_treino.columns[0]
    nome_coluna2 = dados_treino.columns[1]
    nome_colunaClass = dados_treino.columns[-1]

    coluna1 = dados_treino[nome_coluna1]
    coluna2 = dados_treino[nome_coluna2]
    colunaClass = dados_treino[nome_colunaClass]
    
    f,(ax1,ax2)=plt.subplots(1,2,sharey=True,figsize=(20,7.5))
    ax1.set_title("Original")
    ax1.scatter(coluna1,coluna2,c=colunaClass,cmap="rainbow")
    ax2.set_title(label=modelo)
    ax2.scatter(coluna1,coluna2,c=opiniao,cmap="rainbow")
    plt.show()

def main():

    # 1. Ler dados
    cprint("Lendo dados...")
    dados = ler_dados()
    cprint(f"Dados carregados: {dados.shape[0]} amostras, {dados.shape[1]} colunas.")

    # # 2. Augmentation
    # cprint("Aplicando Data Augmentation...", label="MAIN")
    # dados = data_augmentation(dados, fator_escala=2)
    # cprint(f"Dataset após augmentation: {dados.shape[0]} amostras.")

    # Separação de features e labels reais
    dados_treino = dados.iloc[:, :-1]
    y_true = dados.iloc[:, -1]

    # 3. Modelos para rodar
    modelos = {
        "K-Means": KMeans(n_clusters=2, max_iter=300, random_state=42),
        "AGNES": AgglomerativeClustering(n_clusters=2, linkage='complete'),
        "DBSCAN": DBSCAN(eps=0.8, min_samples=20)
    }

    for nome, modelo in modelos.items():
        cprint(f"Rodando {nome}...", label=nome.upper())
        
        # Treino e Predição
        modelo.fit(dados_treino)
        opiniao = modelo.labels_

        # Cálculo de Métricas
        metricas = {
            "homogeneidade": homogeneity_score(y_true, opiniao),
            "completude": completeness_score(y_true, opiniao),
            "v_measure": v_measure_score(y_true, opiniao),
            "indice_randomico": rand_score(y_true, opiniao),
            "indice_randomico_ajustado": adjusted_rand_score(y_true, opiniao),
            "silhueta": silhouette_score(dados_treino, opiniao),
            "entropia": None # implementar
        }

        # Exibição
        printar_metricas(metricas, nome)
        plotagraficos(dados, opiniao, nome)

    cprint("Processamento concluído.")

if __name__ == "__main__":
    main()