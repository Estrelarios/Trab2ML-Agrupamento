# TODO - Trabalho 2 de Aprendizagem de Máquina (Agrupamento)

## Implementação

- [] **Dados**
    - [] Verificar se precisam de pre processamento (normalização)

- [ ] **Métricas Intrínsecas:**
    - [x] Implementar cálculo de **Coesão** (ex: WCSS - Within-Cluster Sum of Squares).
    - [ ] Implementar cálculo de **Separação** (ex: BSS - Between-Cluster Sum of Squares).
        - [ ] Perguntar pro andrezao qual tipo é pra usar, elemento de um cluster para todos os outros ou centroide para centroide
    - [x] Validar **Coeficiente de Silhueta** (já presente no código).

- [ ] **Métricas Extrínsecas:**
    - [] Implementar cálculo de **Entropia** (pode ser a média da entropia de cada cluster em relação às classes reais).
        - [] Conferir se é para deixar a entropia prox de 0 mesmo ou se é para fazer 1 - entropia. Por que? -> entropia é minimização, os outros não
    - [x] Validar **Homogeneidade** (já presente).
    - [x] Validar **Completude** (já presente).
    - [x] Validar **V-Measure** (já presente).
    - [x] Validar **Índice Randômico** (já presente).
    - [x] Validar **Índice Randômico Ajustado (ARI)** (já presente).

- [ ] **Calibração de Modelos:**
    - [ ] Ajustar a lógica de escolha do "melhor" modelo e calibração de parametros: Ver sobre função de fitness com soma ponderada das métricas.
    - [x] Garantir que os hiperparâmetros específicos da Tabela 1 estão sendo calibrados:
        - [x] K-means: `n_clusters`, `max_iter`.
        - [x] DBScan: `eps`, `min_samples`.
        - [x] AGNES: `n_clusters`, `linkage`.

- [ ] **Análise e Visualização:**
    - [ ] Gerar gráficos comparativos (Original vs. Clusterizado) para os melhores modelos de cada tipo.
    - [ ] Exportar as métricas obtidas em um formato fácil de copiar para o relatório (ex: CSV ou tabela formatada no log).
    - [ ] Adicionar visualização dos centroides no K-means (conforme exemplo da aula).

- [ ] **Organização do Código:**
    - [ ] Refatorar `src/main.py` para separar a lógica de carregamento, treinamento, cálculo de métricas e plotagem.
    - [ ] Remover ou isolar funções de `data_augmentation` que não foram pedidas explicitamente no PDF do trabalho.

## Relatório (Documentação)

- [ ] Elaborar o relatório técnico seguindo o formato do PDF original.
- [ ] Detalhar a faixa de valores variada para cada parâmetro.
- [ ] Realizar análise detalhada das métricas.
- [ ] Identificar qual métrica melhor representa o desempenho para este dataset.
- [ ] Comparar as três estratégias.
- [ ] Análise dos clusters formados e insights.
