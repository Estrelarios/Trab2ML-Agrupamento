# Estudando

arquivo para anotações e entender os algs

## DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

- Decide quantos clusters formar
- Agrupamento por densidade

Parâmetros:
- eps = Epsilon -> distância (raio) mínimo para considerar que 2 amostras são vizinhas
- num_samples -> número mínimo de amostras para criar um cluster

## K-Means

- Agrupamento por centralidade

Parâmetros:
- n_clusters: Número de clusters que deseja-se formar
- max_iter: Número máximo de iterações desejado. O algortimo para se convergir antes.

## AGNES (Aglomerative Nesting)

Vai de baixo pra cima (Bottom-up) formando grupos. No começo, se existem N amostras, haverá N grupos (cada amostra é um grupo). Em cada iteração ele junta os grupos mais próximos (Precisa-se definir o método para definir a proximidade)

- Agrupamento por hierarquização

Parâmetros:
- n_clusters
- linkage: Critério para definir a distância (ward, complete, average e single)

Critérios:
- Complete (ou Max): Distância dos clusters é igual à distância entre os pontos mais distantes dos clusters (P1 do cluster A e P2 do cluster B)
- Ward : Padrão do sklearn. Minimiza a variância total dentro dos grupos. Ele tende a criar clusters de tamanhos mais homogêneos e arredondados.
- Average: Distância dos clusters é igual à distância entre a média dos pontos de cada cluster
- single (ou Min): Calcula a distância entre los dois pontos mais próximos de cada grupo. Pode criar grupos longos em formato de "corrente", ligando pontos distantes por meio de intermediários