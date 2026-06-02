# Gerais
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ajuste de path para encontrar o utils
caminho_raiz = Path(__file__).resolve().parent.parent
if str(caminho_raiz) not in sys.path:
    sys.path.append(str(caminho_raiz))

# Métodos
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn import metrics 
from utils.print_customizado import cprint


def ler_dados():
    
    caminho_atual = Path(__file__).resolve()
    raiz = caminho_atual.parent.parent
    caminho_dados = raiz / "dados_raw" / "Banana.csv"

    dados = pd.read_csv(caminho_dados)

    return dados

def plotagraficos (dados_treino : pd.DataFrame, opiniao, modelo):

    nome_coluna1 = dados_treino.columns[0]
    nome_coluna2 = dados_treino.columns[1]
    nome_colunaClass = dados_treino.columns[-1]

    coluna1 = dados_treino[nome_coluna1]
    coluna2 = dados_treino[nome_coluna2]
    colunaClass = dados_treino[nome_colunaClass]
    
    f,(ax1,ax2)=plt.subplots(1,2,sharey=True,figsize=(20,7.5))
    ax1.set_title(label=modelo)
    ax1.scatter(coluna1,coluna2,c=opiniao,cmap="rainbow")
    ax2.set_title("Original")
    ax2.scatter(coluna1,coluna2,c=colunaClass,cmap="rainbow")
    plt.show()

def main():

    # 1. Ler dados
    cprint("Lendo dados...")
    dados = ler_dados()
    cprint(f"Dados carregados: {dados.shape[0]} amostras, {dados.shape[1]} colunas.")

    # 2. Processar dados se necessario

    cprint("Rodando K-Means...", label="KMEANS")
    DadosTreino = pd.DataFrame(dados,columns=dados.columns[:-1])
    kmeans = KMeans(n_clusters=2, max_iter=300,random_state=42)
    kmeans.fit(DadosTreino)
    X = kmeans.labels_
    plotagraficos(dados, X,'K-Means')

    cprint("Rodando AGNES...", label="AGNES")
    DadosTreino = pd.DataFrame(dados,columns=dados.columns[:-1])
    Agnes = AgglomerativeClustering(n_clusters=2,linkage='complete')
    Agnes.fit(DadosTreino)
    X = Agnes.labels_
    plotagraficos(dados, X,'AGNES')
    
    cprint("Rodando DBSCAN...", label="DBSCAN")
    DadosTreino = pd.DataFrame(dados,columns=dados.columns[:-1])
    db = DBSCAN(eps=0.8, min_samples=20)
    db.fit(DadosTreino)
    X = db.labels_
    plotagraficos(dados, X,'DBScan')

    cprint("Processamento concluído.")
    
    # Para cada metodo:
        # 3. Buscar hiperparametros
        # 4. Rodar metodo e catar saída
        # 5. Avaliar resultados


if __name__ == "__main__":
    main()