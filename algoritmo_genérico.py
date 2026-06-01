# Arquivo para mostrar um algoritmo generico que poderemos usar depois no trabalho

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dados = pd.read_csv("Banana.csv")
df_dados = pd.DataFrame(dados)

# plt.figure(figsize=(10, 7.5))
# plt.scatter(df_dados['A1'], df_dados['A2'], c=df_dados['Class'], cmap="rainbow")
# plt.show()  # ✅ já estava correto


def plotagraficos(opiniao, modelo):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20, 7.5))
    ax1.set_title(label=modelo)
    ax1.scatter(DadosTreino['A1'], DadosTreino['A2'], c=opiniao, cmap="rainbow")
    ax2.set_title("Original")
    ax2.scatter(DadosTreino['A1'], DadosTreino['A2'], c=dados['Class'], cmap="rainbow")
    plt.show()  # ✅ CORRIGIDO: adicionado plt.show()

# # ── GENERICO ──────────────────────────────────────────────────────────────────
# from sklearn.cluster import Metodo # Import

# DadosTreino = pd.DataFrame(dados, columns=dados.columns)
# metodoGenerico = Metodo(parametros)
# metodoGenerico.fit(DadosTreino)
# X = metodoGenerico.labels_
# plotagraficos(X, f"{metodoGenerico.__str__()}")


# ── DBSCAN ──────────────────────────────────────────────────────────────────
from sklearn.cluster import DBSCAN

DadosTreino = pd.DataFrame(dados, columns=dados.columns[:-1])
db = DBSCAN(eps=0.8, min_samples=20)

db.fit(DadosTreino)
X = db.labels_
plotagraficos(X, 'DBScan')

# ── KMeans ───────────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn import metrics

kmeans = KMeans(n_clusters=2, max_iter=100, random_state=12)
DadosTreino = pd.DataFrame(dados, columns=dados.columns[:-1])
kmeans.fit(DadosTreino)
X = kmeans.labels_
plotagraficos(X, kmeans.__str__())

# ── AGNES ────────────────────────────────────────────────────────────────────
from sklearn.cluster import AgglomerativeClustering

DadosTreino = pd.DataFrame(dados, columns=dados.columns[:-1])
Agnes = AgglomerativeClustering(n_clusters=2, linkage='complete')
Agnes.fit(DadosTreino)
X = Agnes.labels_
plotagraficos(X, Agnes.__str__())