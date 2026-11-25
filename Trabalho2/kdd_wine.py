# Dependencias cientificas, algoritmos de aprendizado e visualizacao em alta resolucao
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelSpreading, LabelPropagation, SelfTrainingClassifier
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)


# Calcula o joelho da curva de distancias k-NN para orientar a escolha do raio eps no DBSCAN
def diagnosticar_eps_dbscan(X_treino_padronizado, min_samples=4, eps_escolhido=2.3):
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_treino_padronizado)
    distancias, _ = nn.kneighbors(X_treino_padronizado)
    k_dist = np.sort(distancias[:, -1])

    # Estima o ponto de maior curvatura pela distancia perpendicular a reta dos extremos
    n_pontos = len(k_dist)
    coords = np.vstack((np.arange(n_pontos), k_dist)).T
    vetor_reta = coords[-1] - coords[0]
    vetor_reta_norm = vetor_reta / np.linalg.norm(vetor_reta)
    vetores_origem = coords - coords[0]
    projecoes = np.dot(vetores_origem, vetor_reta_norm)
    dist_perpendicular = np.linalg.norm(
        vetores_origem - np.outer(projecoes, vetor_reta_norm), axis=1
    )
    eps_joelho = float(k_dist[np.argmax(dist_perpendicular)])

    print(f"\nDiagnostico k-NN (min_samples={min_samples}) em {X_treino_padronizado.shape[1]} dimensoes:")
    print(f"Joelho geometrico estimado: {eps_joelho:.3f} | eps calibrado adotado: {eps_escolhido:.2f}")

    # Avalia uma grade estreita em torno do eps escolhido para confirmar estabilidade dos clusters
    candidatos = sorted(set([round(eps_escolhido - 0.1, 2), eps_escolhido, round(eps_escolhido + 0.1, 2), round(eps_joelho, 2)]))
    registros = []
    for eps in candidatos:
        rotulos = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_treino_padronizado)
        n_grupos = len(set(rotulos)) - (1 if -1 in rotulos else 0)
        pct_ruido = np.mean(rotulos == -1) * 100.0
        registros.append({
            "eps": eps,
            "clusters_encontrados": n_grupos,
            "ruido_%": round(pct_ruido, 2),
        })
    print(pd.DataFrame(registros).to_string(index=False))


# Mascara os rotulos de forma estratificada garantindo um piso minimo por classe em bases pequenas
def mascarar_rotulos_estratificado(y, taxa, min_por_classe=2, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    n_total = len(y)
    classes, contagens = np.unique(y, return_counts=True)
    n_classes = len(classes)

    # Garante pelo menos min_por_classe amostras rotuladas em cada cultivar antes do rateio proporcional
    n_rotulados_alvo = max(int(round(n_total * taxa)), n_classes * min_por_classe)
    n_rotulados_alvo = min(n_rotulados_alvo, n_total)

    alocacao = np.full(n_classes, min_por_classe, dtype=int)
    restante = n_rotulados_alvo - alocacao.sum()

    if restante > 0:
        proporcoes = contagens / contagens.sum()
        extras = np.floor(proporcoes * restante).astype(int)
        alocacao += extras
        sobra = restante - extras.sum()
        if sobra > 0:
            prioridade = np.argsort(-(proporcoes * restante - extras))
            for idx in prioridade[:sobra]:
                if alocacao[idx] < contagens[idx]:
                    alocacao[idx] += 1

    y_mascarado = np.full(n_total, -1, dtype=int)
    for cls, qtd in zip(classes, alocacao):
        indices_cls = np.where(y == cls)[0]
        selecionados = rng.choice(indices_cls, size=min(qtd, len(indices_cls)), replace=False)
        y_mascarado[selecionados] = cls

    return y_mascarado


# Encapsula as etapas de pre-processamento, mineracao e visualizacao do processo KDD
def executar_pipeline_kdd(
    X,
    y,
    feature_names,
    target_names,
    nome_dataset,
    n_clusters,
    top_3_features,
    eps_dbscan=2.3,
    min_samples_dbscan=4,
    taxas_rotulados=(0.05, 0.10, 0.20, 0.30, 0.50, 0.70),
    n_repeticoes=10,
    taxa_matriz=0.05,
):
    # Prepara diretorios de saida relativos a localizacao do script no projeto
    pasta_base = Path(__file__).resolve().parent
    pasta_figuras = pasta_base / "results" / "figures"
    pasta_metricas = pasta_base / "results" / "metrics"
    pasta_figuras.mkdir(parents=True, exist_ok=True)
    pasta_metricas.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")
    df_completo = pd.DataFrame(X, columns=feature_names)
    df_completo["classe"] = [target_names[i] for i in y]

    # Separa conjunto de teste estratificado antes de qualquer transformacao estatistica
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # Ajusta o scaler apenas no treino para evitar vazamento de dados para o teste
    scaler = StandardScaler()
    X_treino_sc = scaler.fit_transform(X_treino)
    X_teste_sc = scaler.transform(X_teste)

    # Projeta os atributos padronizados em duas componentes principais para inspecao visual
    pca = PCA(n_components=2, random_state=42)
    X_treino_pca = pca.fit_transform(X_treino_sc)
    var_explicada = pca.explained_variance_ratio_ * 100.0

    # Executa o agrupamento como caso limite sem rotulos (0% supervisionado)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    clusters_kmeans = kmeans.fit_predict(X_treino_sc)

    dbscan = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
    clusters_dbscan = dbscan.fit_predict(X_treino_sc)
    pct_ruido_dbscan = np.mean(clusters_dbscan == -1) * 100.0

    # Calcula metricas externas e internas dos algoritmos de agrupamento
    sil_dbscan = (
        silhouette_score(X_treino_sc, clusters_dbscan)
        if len(set(clusters_dbscan)) > 1
        else np.nan
    )
    df_clustering = pd.DataFrame([
        {
            "Algoritmo": f"K-Means (k={n_clusters})",
            "Clusters_Validos": n_clusters,
            "Ruido_%": 0.0,
            "ARI": round(adjusted_rand_score(y_treino, clusters_kmeans), 4),
            "NMI": round(normalized_mutual_info_score(y_treino, clusters_kmeans), 4),
            "Silhouette": round(silhouette_score(X_treino_sc, clusters_kmeans), 4),
        },
        {
            "Algoritmo": f"DBSCAN (eps={eps_dbscan}, min={min_samples_dbscan})",
            "Clusters_Validos": len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0),
            "Ruido_%": round(pct_ruido_dbscan, 2),
            "ARI": round(adjusted_rand_score(y_treino, clusters_dbscan), 4),
            "NMI": round(normalized_mutual_info_score(y_treino, clusters_dbscan), 4),
            "Silhouette": round(sil_dbscan, 4) if not np.isnan(sil_dbscan) else np.nan,
        },
    ])

    print(f"\nResultados de Agrupamento - {nome_dataset} (0% rotulado)")
    print(df_clustering.to_string(index=False))
    df_clustering.to_csv(pasta_metricas / f"{nome_dataset}_clustering.csv", index=False)

    # Avalia baselines supervisionados e algoritmos semi-supervisionados variando a taxa de rotulos
    gamma_rbf = 1.0 / X_treino_sc.shape[1]
    nomes_modelos = [
        "KNN (Baseline)",
        "Reg. Logistica (Baseline)",
        "Label Spreading",
        "Label Propagation",
        "Self-Training (LogReg)",
    ]
    historico = []
    predicoes_matriz = {}

    for taxa in taxas_rotulados:
        for rep in range(n_repeticoes):
            rng = np.random.default_rng(100 + rep)
            y_semi = mascarar_rotulos_estratificado(y_treino, taxa, min_por_classe=2, rng=rng)
            mascara_rot = y_semi != -1
            X_rot = X_treino_sc[mascara_rot]
            y_rot = y_semi[mascara_rot]

            # Ajusta k dinamicamente quando a amostra rotulada minima e muito reduzida
            k_knn = min(5, max(1, len(X_rot) // 2))

            modelos = {
                "KNN (Baseline)": KNeighborsClassifier(n_neighbors=k_knn),
                "Reg. Logistica (Baseline)": LogisticRegression(max_iter=1000, random_state=42),
                "Label Spreading": LabelSpreading(kernel="rbf", gamma=gamma_rbf, alpha=0.2, max_iter=1000),
                "Label Propagation": LabelPropagation(kernel="rbf", gamma=gamma_rbf, max_iter=1000),
                "Self-Training (LogReg)": SelfTrainingClassifier(
                    LogisticRegression(max_iter=1000, random_state=42), threshold=0.80
                ),
            }

            for nome_modelo, clf in modelos.items():
                if "Baseline" in nome_modelo:
                    clf.fit(X_rot, y_rot)
                else:
                    clf.fit(X_treino_sc, y_semi)

                y_pred = clf.predict(X_teste_sc)
                historico.append({
                    "Taxa_Rotulados": taxa,
                    "Amostras_Rotuladas": int(mascara_rot.sum()),
                    "Modelo": nome_modelo,
                    "Acuracia": accuracy_score(y_teste, y_pred),
                    "F1_Macro": f1_score(y_teste, y_pred, average="macro"),
                })

                if rep == 0 and np.isclose(taxa, taxa_matriz):
                    predicoes_matriz[nome_modelo] = y_pred

    # Consolida medias e desvios padrao por taxa de rotulacao para exportacao e apresentacao
    df_resultados = pd.DataFrame(historico)
    df_resumo = (
        df_resultados.groupby(["Taxa_Rotulados", "Amostras_Rotuladas", "Modelo"], as_index=False)
        .agg(
            Acuracia_Media=("Acuracia", "mean"),
            Acuracia_DP=("Acuracia", "std"),
            F1_Macro_Media=("F1_Macro", "mean"),
            F1_Macro_DP=("F1_Macro", "std"),
        )
    )
    df_resumo.to_csv(pasta_metricas / f"{nome_dataset}_semi_supervisionado.csv", index=False)

    tabela_pivoteada = df_resumo.assign(
        Acc_Formatada=df_resumo.apply(
            lambda r: f"{r['Acuracia_Media']:.3f} +/- {r['Acuracia_DP']:.3f}", axis=1
        ),
        F1_Formatado=df_resumo.apply(
            lambda r: f"{r['F1_Macro_Media']:.3f} +/- {r['F1_Macro_DP']:.3f}", axis=1
        ),
    )

    print(f"\nAcuracia no Teste - {nome_dataset} (Media +/- DP em {n_repeticoes} execucoes)")
    print(
        tabela_pivoteada.pivot(
            index=["Taxa_Rotulados", "Amostras_Rotuladas"], columns="Modelo", values="Acc_Formatada"
        )[nomes_modelos].to_string()
    )

    print(f"\nF1-Score Macro no Teste - {nome_dataset} (Media +/- DP em {n_repeticoes} execucoes)")
    print(
        tabela_pivoteada.pivot(
            index=["Taxa_Rotulados", "Amostras_Rotuladas"], columns="Modelo", values="F1_Formatado"
        )[nomes_modelos].to_string()
    )

    # Grafico 1: Compara a distribuicao dos tres atributos mais discriminantes entre as classes
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    paleta_classes = sns.color_palette("Set2", n_colors=n_clusters)
    for ax, feat in zip(axes, top_3_features):
        sns.boxplot(
            data=df_completo,
            x="classe",
            y=feat,
            hue="classe",
            palette=paleta_classes,
            legend=False,
            ax=ax,
        )
        ax.set_title(feat, fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Valor original")
    fig.suptitle(
        f"{nome_dataset} - Distribuicao dos Principais Atributos por Classe",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(pasta_figuras / f"{nome_dataset}_1_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Grafico 2: Contrasta classes reais com os agrupamentos K-Means e DBSCAN no espaco PCA 2D
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    paineis_pca = [
        ("Classes Reais (Ground Truth)", [target_names[i] for i in y_treino]),
        (f"K-Means (ARI={df_clustering.loc[0, 'ARI']:.2f})", [f"Cluster {c}" for c in clusters_kmeans]),
        (
            f"DBSCAN (ARI={df_clustering.loc[1, 'ARI']:.2f} | Ruido={pct_ruido_dbscan:.1f}%)",
            ["Ruido (-1)" if c == -1 else f"Cluster {c}" for c in clusters_dbscan],
        ),
    ]
    for ax, (titulo, rotulos_painel) in zip(axes, paineis_pca):
        sns.scatterplot(
            x=X_treino_pca[:, 0],
            y=X_treino_pca[:, 1],
            hue=rotulos_painel,
            palette="tab10",
            s=75,
            alpha=0.88,
            edgecolor="black",
            linewidth=0.4,
            ax=ax,
        )
        ax.set_title(titulo, fontsize=12, fontweight="bold")
        ax.set_xlabel(f"PC1 ({var_explicada[0]:.1f}% var.)")
        ax.set_ylabel(f"PC2 ({var_explicada[1]:.1f}% var.)")
        ax.legend(title="", loc="best", fontsize=9)
    fig.suptitle(
        f"{nome_dataset} - Projecao PCA 2D: Estrutura Real vs. Agrupamento Nao Supervisionado",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(pasta_figuras / f"{nome_dataset}_2_pca_clustering.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Grafico 3: Mostra a curva de aprendizado de Acuracia e F1-Macro de 5% a 70% de rotulos
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    estilos_linha = {
        "KNN (Baseline)": ("--", "o"),
        "Reg. Logistica (Baseline)": ("--", "s"),
        "Label Spreading": ("-", "^"),
        "Label Propagation": ("-", "D"),
        "Self-Training (LogReg)": ("-", "P"),
    }
    metricas_curva = [("Acuracia_Media", "Acuracia no Teste"), ("F1_Macro_Media", "F1-Score Macro no Teste")]

    for ax, (col_metrica, titulo_eixo) in zip(axes, metricas_curva):
        for nome_modelo in nomes_modelos:
            sub = df_resumo[df_resumo["Modelo"] == nome_modelo].sort_values("Taxa_Rotulados")
            estilo, marcador = estilos_linha[nome_modelo]
            ax.plot(
                sub["Taxa_Rotulados"] * 100.0,
                sub[col_metrica],
                linestyle=estilo,
                marker=marcador,
                linewidth=2.2,
                markersize=7,
                label=nome_modelo,
            )
        ax.set_title(titulo_eixo, fontsize=13, fontweight="bold")
        ax.set_xlabel("Percentual de Amostras Rotuladas no Treino (%)")
        ax.set_ylabel("Pontuacao")
        ax.set_xticks([t * 100.0 for t in taxas_rotulados])
        ax.set_ylim(0.65, 1.03)
        ax.legend(fontsize=9.5, loc="lower right")

    fig.suptitle(
        f"{nome_dataset} - Desempenho Supervisionado vs. Semi-Supervisionado (0.05 a 0.70)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(pasta_figuras / f"{nome_dataset}_3_curva_aprendizado.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Grafico 4: Exibe as matrizes de confusao 3x3 no regime critico de escassez de rotulos
    modelos_matriz = [
        "KNN (Baseline)",
        "Reg. Logistica (Baseline)",
        "Label Spreading",
        "Self-Training (LogReg)",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    nomes_curtos = [nome.replace("class_", "Cultivar ") for nome in target_names]

    for ax, nome_modelo in zip(axes, modelos_matriz):
        cm = confusion_matrix(y_teste, predicoes_matriz[nome_modelo])
        acc_local = accuracy_score(y_teste, predicoes_matriz[nome_modelo])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=nomes_curtos,
            yticklabels=nomes_curtos,
            ax=ax,
        )
        ax.set_title(f"{nome_modelo}\n(Acc = {acc_local:.3f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")

    fig.suptitle(
        f"{nome_dataset} - Matrizes de Confusao 3x3 com {int(taxa_matriz * 100)}% de Rotulos",
        fontsize=14,
        fontweight="bold",
        y=1.04,
    )
    plt.tight_layout()
    fig.savefig(pasta_figuras / f"{nome_dataset}_4_matrizes_confusao.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nGraficos salvos em: {pasta_figuras}")
    print(f"Tabelas CSV salvas em: {pasta_metricas}")


# Carrega o dataset Wine, diagnostica o parametro eps e executa a pipeline completa
if __name__ == "__main__":
    wine = load_wine()

    # Executa pre-diagnostico de vizinhanca no conjunto padronizado para validar eps=2.3
    X_tr_diag, _, _, _ = train_test_split(
        wine.data, wine.target, test_size=0.25, stratify=wine.target, random_state=42
    )
    X_tr_diag_sc = StandardScaler().fit_transform(X_tr_diag)
    diagnosticar_eps_dbscan(X_tr_diag_sc, min_samples=4, eps_escolhido=2.3)

    executar_pipeline_kdd(
        X=wine.data,
        y=wine.target,
        feature_names=list(wine.feature_names),
        target_names=list(wine.target_names),
        nome_dataset="Wine",
        n_clusters=3,
        top_3_features=["flavanoids", "color_intensity", "alcohol"],
        eps_dbscan=2.3,
        min_samples_dbscan=4,
    )