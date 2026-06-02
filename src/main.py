import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
import numpy as np
from pathlib import Path


def ler_dados():
    
    caminho_atual = Path(__file__).resolve()
    raiz = caminho_atual.parent.parent
    caminho_dados = raiz / "dados_raw" / "Banana.csv"

    dados = pd.read_csv(caminho_dados)

    return dados




def main():

    # 1. Ler dados

    dados = ler_dados()
    df_dados = pd.DataFrame(dados)

    
    # 2. Processar dados se necessario
    # Para cada metodo:
        # 3. Buscar hiperparametros
        # 4. Rodar metodo e catar saída
        # 5. Avaliar resultados


if __name__ == "__main__":
    main()