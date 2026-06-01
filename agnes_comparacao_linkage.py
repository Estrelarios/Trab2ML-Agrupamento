# -*- coding: utf-8 -*-
"""Comparação dos 4 tipos de linkage do AGNES em grid 2x2"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

from sklearn.neighbors import kneighbors_graph


# ── Carregando dados ──────────────────────────────────────────────────────────
dados = pd.read_csv("Banana.csv")
DadosTreino = pd.DataFrame(dados, columns=dados.columns[:-1])

# ── Configuração ──────────────────────────────────────────────────────────────
linkages = ['ward', 'complete', 'average', 'single']
n_clusters = 2

# ── Grid 2x2 ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('AGNES — Comparação dos 4 tipos de Linkage', fontsize=16, fontweight='bold', y=1.01)

# 1. Cria uma matriz de conectividade para evitar pontes de ruído (ex: 10 vizinhos)
# 'X' representa seus dados no formato de banana
connectivity_matrix = kneighbors_graph(DadosTreino, n_neighbors=10, include_self=False)

for ax, linkage in zip(axes.flatten(), linkages):
    agnes = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric='euclidean')
    agnes.fit(DadosTreino)
    labels = agnes.labels_

    ax.scatter(DadosTreino['A1'], DadosTreino['A2'], c=labels, cmap='rainbow', s=10, alpha=0.7)
    ax.set_title(f"Linkage: {linkage.upper()}", fontsize=13, fontweight='bold')
    ax.set_xlabel('A1')
    ax.set_ylabel('A2')

plt.tight_layout()
plt.show()