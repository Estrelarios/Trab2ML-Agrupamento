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
from src.metricas import entropia, coesao, separacao_clusters, separacao_pontoAponto


# Métricas
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import rand_score, adjusted_rand_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import StandardScaler

# Configuração de pesos para avaliação
pesos_config = {
    "ari": 1,
    # "v_measure" : .5,
    # "sil" : .25,
}

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

def plotagraficos (X : pd.DataFrame, opiniao, modelo, params, pesos):

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
    
    # Formata pesos para o título
    pesos_str = ", ".join([f"{k}={v}" for k, v in pesos.items()])
    ax2.set_title(label=f"{modelo} | {params}\nPesos: {pesos_str}", fontsize=10)
    
    # Tratamento para destacar o ruído se existir
    if -1 in list(opiniao):
        # Pontos normais
        mask_normal = opiniao != -1
        ax2.scatter(coluna1[mask_normal], coluna2[mask_normal], c=opiniao[mask_normal], cmap="rainbow")
        # Pontos de ruído em preto
        ax2.scatter(coluna1[~mask_normal], coluna2[~mask_normal], c='black', label='Ruído', marker='x', s=50)
        ax2.legend(loc='upper right')
    else:
        ax2.scatter(coluna1, coluna2, c=opiniao, cmap="rainbow")
    
    # Salvar o gráfico com iterador para não sobrescrever
    base_nome = f"resultado_{modelo.lower().replace('-', '_')}"
    extensao = ".png"
    
    contador = 1
    caminho_save = caminho_resultados / f"{base_nome}_{contador}{extensao}"
    while caminho_save.exists():
        contador += 1
        caminho_save = caminho_resultados / f"{base_nome}_{contador}{extensao}"
        
    plt.savefig(caminho_save)
    cprint(f"Gráfico salvo em: {caminho_save}", label="PLOT")
    
    plt.close(f) # Fecha a figura para liberar memória


    
def avaliar(Y_real, Y_pred, X):
    # Evita erro se houver apenas 1 cluster ou apenas ruído
    n_clusters = len(set(Y_pred)) - (1 if -1 in Y_pred else 0)
    if n_clusters <= 1:
        return -1

    # Número de classes reais para normalizar entropia
    n_classes_reais = len(np.unique(Y_real))
    log2_n_classes = np.log2(n_classes_reais) if n_classes_reais > 1 else 1.0
    
    pesos = pesos_config

    valores = {}

    if "ari" in pesos:
        valores["ari"] = adjusted_rand_score(Y_real, Y_pred)

    if "homog" in pesos:
        valores["homog"] = homogeneity_score(Y_real, Y_pred)

    if "sil" in pesos:
        try: 
            sil = silhouette_score(X, Y_pred)
            valores["sil"] = (sil + 1) / 2 # Normalização
        except:
            valores["sil"] = 0

    if "1-entropia" in pesos:
        ent = entropia(Y_real, Y_pred)
        valores["1-entropia"] = 1 - (ent / log2_n_classes) # Normalização

    if "1-coesao" in pesos:
        valores["1-coesao"] = 1 - coesao(X, Y_pred)

    if "v_measure" in pesos:
        valores["v_measure"] = v_measure_score(Y_real, Y_pred)

    if "completeness" in pesos:
        valores["completeness"] = completeness_score(Y_real, Y_pred)
        
    if "separacao" in pesos:
        valores["separacao"] = separacao_clusters(X, Y_pred)

    # Calculo score final
    score = sum(valores[m] * pesos[m] for m in pesos if m in valores)
    return score


def main():

    modelos = {}

    # 1. Ler dados
    cprint("Lendo dados...")
    dados = ler_dados()
    cprint(f"Dados carregados: {dados.shape[0]} amostras, {dados.shape[1]} colunas.")

    # Separação entre treino e classes reais
    X_raw = dados.iloc[:, :-1]
    Y = dados.iloc[:, -1]

    # Normalização
    cprint("Normalizando dados (StandardScaler)...")
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_raw)
    X = pd.DataFrame(X_np, columns=X_raw.columns)

    # Rodando algoritmos

# ── KMEANS ────────────────────────────────────────────────────────────────────
    melhor_score = -1
    melhor_modelo_kmeans = None
    for n_clusters in range(2, 10):
        for max_iter in [2, 5, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300]:
            modelo = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
            modelo.fit(X)
            score = avaliar(Y, modelo.labels_, X)
            if score > melhor_score:
                melhor_score = score
                melhor_modelo_kmeans = modelo

    cprint(f"Melhor configuração: n_clusters={melhor_modelo_kmeans.n_clusters}, max_iter={melhor_modelo_kmeans.max_iter}, score={melhor_score:.4f}", label="KMEANS")
    modelos["K-Means"] = melhor_modelo_kmeans

# ── AGNES ─────────────────────────────────────────────────────────────────────
    melhor_score = -1
    melhor_modelo_agnes = None
    for n_clusters in range(2, 10):
        for linkage in ['ward', 'complete', 'average', 'single']:
            modelo = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            modelo.fit(X)
            score = avaliar(Y, modelo.labels_, X)
            if score > melhor_score:
                melhor_score = score
                melhor_modelo_agnes = modelo

    cprint(f"Melhor configuração: n_clusters={melhor_modelo_agnes.n_clusters}, linkage={melhor_modelo_agnes.linkage}, score={melhor_score:.4f}", label="AGNES")
    modelos["AGNES"] = melhor_modelo_agnes

# ── DBSCAN ────────────────────────────────────────────────────────────────────
    melhor_score = -1
    melhor_modelo_dbscan = None
    for eps in range(1, 200, 5): # eps in range (0.01 ate 2.0)
        eps = eps/100
        for min_samples in range(5, 30, 1):
            modelo = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1, algorithm="kd_tree")
            modelo.fit(X)
            score = avaliar(Y, modelo.labels_, X)
            if score > melhor_score:
                melhor_score = score
                melhor_modelo_dbscan = modelo

    cprint(f"Melhor configuração: eps={melhor_modelo_dbscan.eps}, min_samples={melhor_modelo_dbscan.min_samples}, score={melhor_score:.4f}", label="DBSCAN")
    modelos["DBSCAN"] = melhor_modelo_dbscan

    for nome, mod in modelos.items():
        cprint(f"Rodando {nome}...", label=nome.upper())
        opiniao = mod.labels_

        if nome == "K-Means": params = f"n_clusters={mod.n_clusters}, max_iter={mod.max_iter}"
        elif nome == "AGNES": params = f"n_clusters={mod.n_clusters}, linkage={mod.linkage}"
        else: params = f"eps={mod.eps}, ms={mod.min_samples}"

        metricas = {
            "homogeneidade": homogeneity_score(Y, opiniao),
            "completude": completeness_score(Y, opiniao),
            "v_measure": v_measure_score(Y, opiniao),
            "indice_randomico": rand_score(Y, opiniao),
            "indice_randomico_ajustado": adjusted_rand_score(Y, opiniao),
            "silhueta": silhouette_score(X, opiniao),
            "1 - entropia": 1 - entropia(Y, opiniao),          #None # implementar
            "1 - coesao": 1 - coesao(X, opiniao),                    #None # implementar
            "separacao": separacao_clusters(X, opiniao)
        }

        printar_metricas(metricas, nome)
        plotagraficos(dados, opiniao, nome, params, pesos_config)

    cprint("Processamento concluído.")

if __name__ == "__main__":
    main()