# Avaliação de Clusters

Arquivo para entender como fazer a avaliação dos algoritmos e encontrar a função de avaliação composta/funcao de fitness (mistura de vários índices de avaliação).

Penso em 2 abordagens:
1. Separar os invasores: Nos 2 clusters "abraçados", há amostras que estão invadindo o espaço do outro cluster. A ideia é tentar de alguma forma separar essas amostras corretamente para cada clusters;
2. Área de incerteza : Fazer as amostras invasoras dos clusters abraçados viraram um cluster só, uma área de dúvida (é do cluster A ou do cluster B?) e dividir aquela região em 3 clusters, tendo no total 4 clusters. Poderia ser expandido para transformar as amostras mais disperas em um cluster (isso criaria agrupamentos compactos e agrupamentos de amostras esparsas);


## 1. Separação completa

A ideia é maximizar a "taxa de acerto" e colocar corretamente as amostras em seus devidos clusters, o que requer remover impurezas. Para isso, é melhor maximizar (um menos) a entropia, a homogeneidade e a completude.

## 2. Área de dúvida

Trabalhar a ideia de que podem haver mais clusters, assumindo uma área de dúvida em que não há certeza sobre qual cluster a amostra pertence. Para isso é ideal maximizar 