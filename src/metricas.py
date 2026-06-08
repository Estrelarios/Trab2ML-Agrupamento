import pandas as pd
import numpy as np
from utils.print_customizado import cprint
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.stats import entropy


def entropia(Y_real, labels_preditos):
    """
    Calcula a entropia média ponderada dos clusters.
    
    Para cada cluster, mede o quão "impuro" ele é em relação
    às classes reais — entropia 0 significa cluster puro.

    Parâmetros:
        Y_real: rótulos verdadeiros
        labels_preditos: rótulos atribuídos pelo modelo

    Retorna:
        Entropia média ponderada (float)
    """
    Y_real = np.array(Y_real)
    labels = np.array(labels_preditos)
    
    n_total = len(Y_real)
    entropia_ponderada = 0.0

    for cluster_id in np.unique(labels):
        if cluster_id == -1:  # ignora ruído do DBSCAN
            continue

        mascara = labels == cluster_id
        classes_no_cluster = Y_real[mascara]
        n_cluster = len(classes_no_cluster)

        # Frequência relativa de cada classe dentro do cluster
        _, contagens = np.unique(classes_no_cluster, return_counts=True)
        proporcoes = contagens / n_cluster

        # Entropia do cluster (base 2 → bits)
        entropia_cluster = entropy(proporcoes, base=2)

        # Ponderação pelo tamanho do cluster
        entropia_ponderada += (n_cluster / n_total) * entropia_cluster

    return entropia_ponderada

def coesao(X, labels_preditos):
    """
    Calcula a coesão média ponderada dos clusters.

    Para cada cluster, mede a distância média de cada ponto
    ao centroide do seu cluster (inércia normalizada).
    Coesão 0 significa que todos os pontos estão no centroide.

    Parâmetros:
        X: DataFrame ou array com os dados
        labels_preditos: rótulos atribuídos pelo modelo

    Retorna:
        Coesão média ponderada (float) — quanto menor, melhor
    """
    X_array = np.array(X)
    labels = np.array(labels_preditos)

    n_total = len(X_array)
    coesao_ponderada = 0.0

    for cluster_id in np.unique(labels):
        if cluster_id == -1:  # ignora ruído do DBSCAN
            continue

        pontos_cluster = X_array[labels == cluster_id]
        """
        labels        = [0, 1, 0, 2, 1, 0]
        cluster_id    = 0

        labels == cluster_id  →  [True, False, True, False, False, True]

        bruxaria pura, filtra so os pontos do cluster_id da vez

        # Jeito manual (equivalente, mas verboso)
            pontos_cluster = []
            for i in range(len(X_array)):
                if labels[i] == cluster_id:
                    pontos_cluster.append(X_array[i])
            pontos_cluster = np.array(pontos_cluster)
        """
        n_cluster = len(pontos_cluster)

        # Centroide do cluster
        centroide = pontos_cluster.mean(axis=0)

        # Distância euclidiana média de cada ponto ao centroide
        distancias = np.linalg.norm(pontos_cluster - centroide, axis=1)
        # np.linalg.norm calcula o comprimento (norma) de um vetor
        # mean tira a média dessas distâncias para obter a coesão do cluster
        coesao_cluster = distancias.mean()

        # Ponderação pelo tamanho do cluster
        coesao_ponderada += (n_cluster / n_total) * coesao_cluster

    # Normaliza pelo diâmetro máximo dos dados
    distancia_maxima = np.linalg.norm(X_array.max(axis=0) - X_array.min(axis=0))
    return coesao_ponderada / distancia_maxima

def separacao_clusters(X, labels_preditos):
    """
    Calcula a separação média entre clusters.

    Mede a distância média entre todos os pares de centroides.
    Separação alta significa clusters bem afastados entre si.

    Parâmetros:
        X: DataFrame ou array com os dados
        labels_preditos: rótulos atribuídos pelo modelo

    Retorna:
        Separação média entre centroides (float) — quanto maior, melhor
    """
    X_array = np.array(X)
    labels = np.array(labels_preditos)

    # Calcula o centroide de cada cluster
    ids_clusters = [c for c in np.unique(labels) if c != -1]  # ignora ruído DBSCAN
    centroides = []

    for cluster_id in ids_clusters:
        pontos_cluster = X_array[labels == cluster_id]
        centroide = pontos_cluster.mean(axis=0)
        centroides.append(centroide)

    centroides = np.array(centroides)

    # Calcula a distância entre todos os pares de centroides
    distancias = []
    for i in range(len(centroides)):
        for j in range(i + 1, len(centroides)):  # i+1 evita repetir pares e comparar com si mesmo
            distancia = np.linalg.norm(centroides[i] - centroides[j])
            distancias.append(distancia)

    return np.mean(distancias)

def separacao_pontoAponto(X, labels_preditos):
    """
    Calcula a separação média entre todos as combinações de pontos entre 2 clusters.

    Para cada par de clusters, mede a distância média entre
    todos os pontos de um cluster e todos os pontos do outro.
    Separação alta significa clusters bem afastados entre si.

    Parâmetros:
        X: DataFrame ou array com os dados
        labels_preditos: rótulos atribuídos pelo modelo

    Retorna:
        Separação média entre pares de clusters (float) — quanto maior, melhor
    """
    X_array = np.array(X)
    labels = np.array(labels_preditos)

    ids_clusters = [c for c in np.unique(labels) if c != -1]  # ignora ruído DBSCAN

    distancias_pares = []

    for i in range(len(ids_clusters)):
        for j in range(i + 1, len(ids_clusters)):  # evita repetir pares

            pontos_a = X_array[labels == ids_clusters[i]]
            pontos_b = X_array[labels == ids_clusters[j]]

            # Distância de cada ponto de A para cada ponto de B
            distancias = []
            for ponto_a in pontos_a:
                for ponto_b in pontos_b:
                    distancia = np.linalg.norm(ponto_a - ponto_b)
                    distancias.append(distancia)

            distancias_pares.append(np.mean(distancias))

    return np.mean(distancias_pares)