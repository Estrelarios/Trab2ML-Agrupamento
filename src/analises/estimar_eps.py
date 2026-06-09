import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Ajuste de path para encontrar o projeto raiz
caminho_projeto = Path(__file__).resolve().parent.parent.parent
if str(caminho_projeto) not in sys.path:
    sys.path.append(str(caminho_projeto))

def estimar_eps():
    # 1. Ler dados
    caminho_dados = caminho_projeto / "dados_raw" / "dados.csv"
    if not caminho_dados.exists():
        print(f"Erro: Arquivo não encontrado em {caminho_dados}")
        return

    dados = pd.read_csv(caminho_dados)
    X = dados.iloc[:, :-1] # Apenas as colunas de atributos

    print(f"Dados carregados: {X.shape[0]} amostras.")

    # 2. Configurar k (MinPts)
    # Regra de ouro: k = 2 * dimensões (para 2D, k=4 ou 5)
    k_valores = [4, 5, 10, 20]
    
    plt.figure(figsize=(12, 8))

    for k in k_valores:
        # Calcular vizinhos mais próximos
        vizinhos = NearestNeighbors(n_neighbors=k)
        fit_vizinhos = vizinhos.fit(X)
        distancias, indices = fit_vizinhos.kneighbors(X)

        # Ordenar as distâncias do k-ésimo vizinho (última coluna)
        distancias_ordenadas = np.sort(distancias[:, k-1], axis=0)

        plt.plot(distancias_ordenadas, label=f'k={k}')

    plt.ylabel("Distância ao k-ésimo vizinho (Eps)")
    plt.xlabel("Pontos ordenados por distância")
    plt.title("Gráfico K-Dist para Estimativa de Eps (DBSCAN)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. Salvar resultado
    caminho_script = Path(__file__).resolve().parent
    caminho_plot = caminho_script / "estimativa_eps_kdist.png"
    
    plt.savefig(caminho_plot)
    print(f"\nGráfico salvo em: {caminho_plot}")
    print("DICA: Procure pelo 'cotovelo' (elbow) no gráfico. O valor de Y nesse ponto é um excelente ponto de partida para o Eps.")
    
    # Tenta mostrar o gráfico se houver interface gráfica
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    estimar_eps()
