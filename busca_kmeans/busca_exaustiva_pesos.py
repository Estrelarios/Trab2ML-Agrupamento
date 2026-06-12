import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from joblib import Parallel, delayed

# Path adjustment
caminho_projeto = Path(__file__).resolve().parent.parent
if str(caminho_projeto) not in sys.path:
    sys.path.append(str(caminho_projeto))

# Methods and Metrics
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, adjusted_rand_score, v_measure_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from src.metricas import entropia, coesao, separacao_clusters

def ler_dados():
    caminho_dados = caminho_projeto / "dados_raw" / "dados.csv"
    dados = pd.read_csv(caminho_dados)
    return dados

def plotar_e_salvar(X, opiniao, modelo_info, nome_arquivo):
    coluna1 = X.iloc[:, 0]
    coluna2 = X.iloc[:, 1]

    f, ax = plt.subplots(figsize=(10, 7))
    
    pesos_str = (f"ARI={modelo_info['w_ari']:.1f}, Ent={modelo_info['w_entr']:.1f}, "
                 f"Comp={modelo_info['w_comp']:.1f}, Hom={modelo_info['w_homog']:.1f}, "
                 f"Sil={modelo_info['w_sil']:.1f}, Coe={modelo_info['w_coesao']:.1f}, "
                 f"Sep={modelo_info['w_sep']:.1f}, V={modelo_info['w_v']:.1f}")
    
    ax.set_title(f"KMeans: {modelo_info['params']}\nScore: {modelo_info['fitness']:.4f} | Pesos: {pesos_str}", fontsize=9)

    if -1 in list(opiniao):
        mask_normal = (opiniao != -1)
        ax.scatter(coluna1[mask_normal], coluna2[mask_normal], c=opiniao[mask_normal], cmap="rainbow")
        ax.scatter(coluna1[~mask_normal], coluna2[~mask_normal], c='black', label='Ruído', marker='x', s=50)   
        ax.legend(loc='upper right')
    else:
        ax.scatter(coluna1, coluna2, c=opiniao, cmap="rainbow")

    caminho_save_dir = Path(__file__).resolve().parent / "resultados"
    caminho_save_dir.mkdir(exist_ok=True)
    plt.savefig(caminho_save_dir / nome_arquivo)
    plt.close(f)

def calcular_modelo(n_clusters, max_iter, init, X, Y, log2_n_classes):
    try:
        modelo = KMeans(n_clusters=n_clusters, max_iter=max_iter, init=init, n_init=10)
        labels = modelo.fit_predict(X)

        if len(set(labels)) > 1:
            sil = silhouette_score(X, labels)
            ent = entropia(Y, labels)
            return {
                "labels": labels,
                "ari": adjusted_rand_score(Y, labels),
                "homog": homogeneity_score(Y, labels),
                "comp": completeness_score(Y, labels),
                "entr": 1 - (ent / log2_n_classes),
                "sil": (sil + 1) / 2,
                "coesao": 1 - coesao(X, labels),
                "sep": separacao_clusters(X, labels),
                "v_measure": v_measure_score(Y, labels),
                "params": f"n_clusters={n_clusters}, max_iter={max_iter}"
            }
    except: pass
    return None

def main():
    print("Busca Exaustiva KMeans (8 Métricas)...")
    dados = ler_dados()
    X_raw = dados.iloc[:, :-1]
    Y = dados.iloc[:, -1]

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)
    
    n_classes_reais = len(np.unique(Y))
    log2_n = np.log2(n_classes_reais) if n_classes_reais > 1 else 1.0

    print("Passo 1: Pré-calculando modelos...")
    n_clusters_range = range(2, 9)
    max_iter_range = [2, 5, 10, 20, 50]
    init_range = ["k-means++", "random"]
    parametros = [(n, m, i) for n in n_clusters_range for m in max_iter_range for i in init_range]

    res_raw = Parallel(n_jobs=-1)(delayed(calcular_modelo)(n, m, i, X, Y, log2_n) for n, m, i in parametros)
    pre_calculados = [r for r in res_raw if r is not None]
    print(f"Modelos válidos: {len(pre_calculados)}")

    print("Passo 2: Busca de Pesos (8D)...")
    log_resultados = []
    resultados_salvos = set()
    imagens_geradas = 0
    step = 0.1
    r = np.arange(0, 1.1, step)
    count = 0

    for w_ari in r:
        for w_entr in r:
            if w_ari + w_entr > 1.001: break
            for w_comp in r:
                if w_ari + w_entr + w_comp > 1.001: break
                for w_homog in r:
                    if w_ari + w_entr + w_comp + w_homog > 1.001: break
                    for w_sil in r:
                        if w_ari + w_entr + w_comp + w_homog + w_sil > 1.001: break
                        for w_coesao in r:
                            if w_ari + w_entr + w_comp + w_homog + w_sil + w_coesao > 1.001: break
                            for w_sep in r:
                                total = w_ari + w_entr + w_comp + w_homog + w_sil + w_coesao + w_sep
                                if total > 1.001: break
                                
                                w_v = round(1.0 - total, 2)
                                if w_v < 0: continue

                                melhor_fit = -999
                                vencedor = None

                                for pc in pre_calculados:
                                    score = (w_ari * pc["ari"]) + (w_entr * pc["entr"]) + \
                                            (w_comp * pc["comp"]) + (w_homog * pc["homog"]) + \
                                            (w_sil * pc["sil"]) + (w_coesao * pc["coesao"]) + \
                                            (w_sep * pc["sep"]) + (w_v * pc["v_measure"])
                                    if score > melhor_fit:
                                        melhor_fit = score
                                        vencedor = pc

                                if vencedor:
                                    res_id = vencedor["params"]
                                    if res_id not in resultados_salvos:
                                        v_info = vencedor.copy()
                                        v_info.update({"w_ari":w_ari,"w_entr":w_entr,"w_comp":w_comp,"w_homog":w_homog,
                                                      "w_sil":w_sil,"w_coesao":w_coesao,"w_sep":w_sep,"w_v":w_v,"fitness":melhor_fit})
                                        fname = f"unique_{imagens_geradas:03d}_{res_id.replace('=','').replace(', ','_')}.png"
                                        plotar_e_salvar(X, vencedor["labels"], v_info, fname)
                                        resultados_salvos.add(res_id)
                                        imagens_geradas += 1

                                    log_resultados.append({"w_ari":w_ari,"w_entr":w_entr,"w_sep":w_sep,"best_params":res_id,"fitness":melhor_fit})
                                    count += 1
                                    if count % 20000 == 0: print(f"Iter: {count} | Img: {imagens_geradas}")

    pd.DataFrame(log_resultados).to_csv(Path(__file__).resolve().parent / "log_busca_7d.csv", index=False)
    print(f"Fim. Iterações: {count}, Únicas: {imagens_geradas}")

if __name__ == "__main__":
    main()
