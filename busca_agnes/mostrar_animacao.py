import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from pathlib import Path

def extrair_pesos(nome_arquivo):
    """Extrai os valores numéricos do nome do arquivo para ordenação."""
    match = re.findall(r'\d+', nome_arquivo)
    return [int(x) for x in match] if match else [0, 0, 0, 0]

import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

def extrair_pesos(nome_arquivo):
    """Extrai os valores numéricos do nome do arquivo para ordenação."""
    match = re.findall(r'\d+', nome_arquivo)
    return [int(x) for x in match] if match else [0, 0, 0, 0]

class AnimacaoInterativa:
    def __init__(self, arquivos, caminho_resultados):
        self.arquivos = arquivos
        self.caminho_resultados = caminho_resultados
        self.indice = 0
        self.pausado = False
        
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        plt.subplots_adjust(left=0, bottom=0.1, right=1, top=0.95)
        self.ax.axis('off')
        
        self.img_display = self.ax.imshow(mpimg.imread(str(self.caminho_resultados / self.arquivos[0])))
        self.titulo = self.ax.set_title(self.arquivos[0], fontsize=14)
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Texto de instrução
        self.fig.text(0.5, 0.02, "ESPAÇO: Play/Pause | SETAS: <- Anterior / Próximo ->", 
                      ha="center", fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        self.timer = self.fig.canvas.new_timer(interval=200)
        self.timer.add_callback(self.proximo_quadro)
        self.timer.start()

    def atualizar_imagem(self):
        img = mpimg.imread(str(self.caminho_resultados / self.arquivos[self.indice]))
        self.img_display.set_data(img)
        self.titulo.set_text(f"Quadro {self.indice+1}/{len(self.arquivos)}: {self.arquivos[self.indice]}")
        self.fig.canvas.draw_idle()

    def proximo_quadro(self):
        if not self.pausado:
            self.indice = (self.indice + 1) % len(self.arquivos)
            self.atualizar_imagem()

    def on_key(self, event):
        if event.key == ' ':
            self.pausado = not self.pausado
            if self.pausado:
                print("Pausado")
            else:
                print("Reproduzindo")
        elif event.key == 'right':
            self.pausado = True
            self.indice = (self.indice + 1) % len(self.arquivos)
            self.atualizar_imagem()
        elif event.key == 'left':
            self.pausado = True
            self.indice = (self.indice - 1) % len(self.arquivos)
            self.atualizar_imagem()

def mostrar_animacao():
    caminho_resultados = Path(__file__).resolve().parent / "resultados"
    
    if not caminho_resultados.exists():
        print(f"Erro: Pasta de resultados não encontrada em {caminho_resultados}")
        return

    arquivos = [f for f in os.listdir(caminho_resultados) if f.endswith('.png')]
    arquivos.sort(key=extrair_pesos)

    if not arquivos:
        print("Nenhuma imagem encontrada na pasta de resultados.")
        return

    print(f"Iniciando animação interativa com {len(arquivos)} quadros...")
    print("CONTROLES:")
    print("  - Barra de Espaço: Play / Pause")
    print("  - Seta Direita: Próximo quadro (Pausa automaticamente)")
    print("  - Seta Esquerda: Quadro anterior (Pausa automaticamente)")
    
    anim = AnimacaoInterativa(arquivos, caminho_resultados)
    plt.show()

if __name__ == "__main__":
    mostrar_animacao()
