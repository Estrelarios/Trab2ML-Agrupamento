# Guia de Métricas de Agrupamento

Este documento descreve as métricas utilizadas para avaliar o desempenho dos algoritmos de clustering no projeto.

## Homogeneidade (Homogeneity)

Mede se cada cluster contém apenas membros de uma única classe real. É uma métrica de "pureza" do cluster. Se um cluster tiver amostras de múltiplas classes, a homogeneidade diminui.

**Máximo (1.0):** Cada cluster contém apenas amostras de uma única classe (os clusters são perfeitamente "puros").

**Mínimo (0.0):** As amostras de diferentes classes estão distribuídas aleatoriamente entre os clusters, não havendo especialização.

## Completude (Completeness)

Mede se todos os membros de uma determinada classe real foram atribuídos ao mesmo cluster. Enquanto a homogeneidade olha para o conteúdo do cluster, a completude olha para a dispersão da classe original.

**Máximo (1.0):** Todos os pontos de uma mesma classe estão reunidos em um único cluster.

**Mínimo (0.0):** Os membros de uma mesma classe original estão espalhados por muitos clusters diferentes.

## V-Measure

É a média harmônica entre a Homogeneidade e a Completude. Funciona de forma análoga ao F1-Score na classificação, garantindo que ambas as qualidades (pureza e integridade) sejam equilibradas.

**Máximo (1.0):** O agrupamento é perfeito, sendo simultaneamente homogêneo (clusters puros) e completo (classes não fragmentadas).

**Mínimo (0.0):** O agrupamento falha em ambas as frentes ou é equivalente a uma atribuição aleatória.

## Índice Randômico Ajustado (ARI)

Mede a similaridade entre o agrupamento realizado e as classes reais, analisando todos os pares de amostras e contando quantos foram agrupados da mesma forma (ou de forma diferente) em ambos. O termo "Ajustado" significa que ele desconta a pontuação que seria obtida por mero acaso.

**Máximo (1.0):** O agrupamento realizado é idêntico às classes originais (concordância perfeita).

**Mínimo (0.0 ou negativo):** A concordância é igual ou inferior ao que seria esperado por um sorteio aleatório de clusters.

## Coeficiente de Silhueta (Silhouette)

Mede o quão próximo cada ponto está do seu próprio cluster (coesão) em comparação com o cluster vizinho mais próximo (separação). É uma métrica interna, ou seja, não depende do conhecimento das classes reais.

**Máximo (1.0):** Os clusters estão bem separados uns dos outros e os pontos estão muito próximos dos centros de seus respectivos clusters.

**Mínimo (-1.0):** Indica que as amostras foram atribuídas ao cluster errado, estando mais perto de um cluster vizinho do que do seu próprio.

## Entropia

Mede o grau de desordem ou incerteza dentro dos clusters em relação às classes reais. É uma medida de impureza: quanto mais misturadas as classes dentro de um cluster, maior a entropia.

**Máximo (Depende do número de classes):** O cluster está totalmente "sujo", contendo uma mistura equilibrada de todas as classes possíveis (máxima incerteza).

**Mínimo (0.0):** O cluster é perfeitamente puro, contendo apenas amostras de uma única classe (certeza absoluta).

## Coesão (Compactness)

Avalia quão próximos os pontos de um mesmo cluster estão entre si. Geralmente calculada pela soma dos quadrados das distâncias internas (WCSS). No código, usamos `1 - coesão` para facilitar a visualização onde valores maiores são melhores.

**Máximo (1.0 no formato 1-coesão):** Todos os pontos de um cluster estão exatamente na mesma posição (distância zero entre eles).

**Mínimo (0.0 no formato 1-coesão):** Os pontos estão o mais espalhados possível dentro do cluster, indicando baixa densidade interna.

## Separação

Mede a distância entre os diferentes clusters formados. Um bom agrupamento deve manter os clusters o mais longe possível uns dos outros para evitar sobreposição e ambiguidade.

**Máximo (Elevado):** Os clusters estão localizados em regiões muito distintas do espaço, sem qualquer proximidade entre suas fronteiras.

**Mínimo (0.0):** Os clusters estão sobrepostos ou seus centros são idênticos, tornando impossível distingui-los espacialmente.
