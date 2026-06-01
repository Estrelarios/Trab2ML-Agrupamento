# -*- coding: utf-8 -*-
"""Aula 01/06/26 Agrupamentos.ipynb - CORRIGIDO

Importante bibliotecas
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dados = pd.read_csv("Banana.csv")
df_dados = pd.DataFrame(dados)

plt.figure(figsize=(10, 7.5))
plt.scatter(df_dados['A1'], df_dados['A2'], c=df_dados['Class'], cmap="rainbow")
plt.show()  # ✅ já estava correto


def plotagraficos(opiniao, modelo):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20, 7.5))
    ax1.set_title(label=modelo)
    ax1.scatter(DadosTreino['A1'], DadosTreino['A2'], c=opiniao, cmap="rainbow")
    ax2.set_title("Original")
    ax2.scatter(DadosTreino['A1'], DadosTreino['A2'], c=dados['Class'], cmap="rainbow")
    plt.show()  # ✅ CORRIGIDO: adicionado plt.show()


# # ── DBSCAN ──────────────────────────────────────────────────────────────────
# from sklearn.cluster import DBSCAN

# DadosTreino = pd.DataFrame(dados, columns=dados.columns[:-1])
# db = DBSCAN(eps=0.8, min_samples=20)
# db.fit(DadosTreino)
# X = db.labels_
# plotagraficos(X, 'DBScan')

# # ── KMeans ───────────────────────────────────────────────────────────────────
# from sklearn.cluster import KMeans
# from sklearn import metrics

# kmeans = KMeans(n_clusters=2, max_iter=100, random_state=12)
# DadosTreino = pd.DataFrame(dados, columns=dados.columns[:-1])
# kmeans.fit(DadosTreino)
# X = kmeans.labels_
# plotagraficos(X, 'K-Means')

# ── AGNES ────────────────────────────────────────────────────────────────────
from sklearn.cluster import AgglomerativeClustering

DadosTreino = pd.DataFrame(dados, columns=dados.columns[:-1])
Agnes = AgglomerativeClustering(n_clusters=2, linkage='complete')
Agnes.fit(DadosTreino)
X = Agnes.labels_
plotagraficos(X, 'AGNES')

"""Gerando conjunto de dados"""

from sklearn.datasets import make_blobs

data = make_blobs(n_samples=2000, n_features=2, centers=4, cluster_std=1.7, random_state=29)
plt.figure(figsize=(10, 7.5))
plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap="rainbow")
plt.show()  # ✅ CORRIGIDO: adicionado plt.show()
print(data)

"""Executando o KMeans"""

from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import rand_score, adjusted_rand_score

DadosTreino = pd.DataFrame(dados, columns=dados.columns[:-1])
kmeans = KMeans(n_clusters=2, max_iter=300, random_state=42)
kmeans.fit(DadosTreino)
X = kmeans.labels_
plotagraficos(X, 'K-Means')

centro = kmeans.cluster_centers_
print(centro)
print("\nMatriz de contingência kmeans\n", contingency_matrix(dados['Class'], kmeans.labels_))
print("\nHomogeneidade : ", metrics.homogeneity_score(dados['Class'], kmeans.labels_))
print("\nCompletude : ", metrics.completeness_score(dados['Class'], kmeans.labels_))
print("Rand Index:", rand_score(dados['Class'], kmeans.labels_))
print("ARI =", adjusted_rand_score(dados['Class'], kmeans.labels_))

"""Comparando dados originais com a saída do KMeans"""

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 5))
ax1.set_title("KMeans")
ax1.scatter(data[0][:, 0], data[0][:, 1], c=X, cmap="rainbow")
ax1.scatter(centro[:, 0], centro[:, 1], color='black')
ax2.set_title("Original")
ax2.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap="rainbow")
plt.show()  # ✅ CORRIGIDO: adicionado plt.show()

"""Executando o DBSCAN"""

db = DBSCAN(eps=2, min_samples=10)
db.fit(data[0])
X = db.labels_
print(X)

"""Plotando os resultados do DBScan"""

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 5))
ax1.set_title("DBScan")
ax1.scatter(data[0][:, 0], data[0][:, 1], c=X, cmap="rainbow")
ax2.set_title("Original")
ax2.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap="rainbow")
plt.show()  # ✅ CORRIGIDO: adicionado plt.show()

"""Executando uma estratégia Aglomerativa (AGNES)"""

Agnes = AgglomerativeClustering(n_clusters=5, linkage='ward')
Agnes.fit(data[0])
X = Agnes.labels_

"""Plotando os resultados do AGNES"""

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 5))
ax1.set_title("AGNES")
ax1.scatter(data[0][:, 0], data[0][:, 1], c=X, cmap="rainbow")
ax2.set_title("Original")
ax2.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap="rainbow")
plt.show()  # ✅ CORRIGIDO: adicionado plt.show()