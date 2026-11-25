# Dependencias de sistema, manipulacao numerica, visualizacao e modelagem do scikit-learn
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# Encapsula todas as etapas do fluxo KDD para reutilizacao entre diferentes datasets
def executar_pipeline_kdd(
    X,
    y,
    nomes_features,
    nomes_classes,
    nome_dataset,
    n_clusters,
    eps_dbscan,
    min_samples_dbscan,
    top_3_features,
):
    # Define tamanhos de fonte legiveis para exportacao direta em slides
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })

    # Prepara o diretorio de saida das figuras para manter a raiz do projeto organizada
    pasta_figuras = os.path.join(f"results_{nome_dataset}", "figures")
    os.makedirs(pasta_figuras, exist_ok=True)

    # Separa 25% dos dados para avaliacao final mantendo a proporcao original das classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # Ajusta o scaler apenas no treino para evitar vazamento de dados para o teste
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Projeta os atributos padronizados em 2 componentes principais exclusivamente para visualizacao
    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    var_explicada = pca.explained_variance_ratio_
    var_acumulada = np.sum(var_explicada) * 100

    # Exibe o diagnostico inicial da particao e da variancia retida na projecao 2D
    print(f"\nResumo do Dataset: {nome_dataset}")
    print(
        f"Amostras treino: {X_train.shape[0]} | "
        f"Amostras teste: {X_test.shape[0]} | "
        f"Atributos: {X_train.shape[1]}"
    )
    print(
        f"PCA (2 componentes) - Variancia explicada: "
        f"PC1={var_explicada[0] * 100:.2f}%, PC2={var_explicada[1] * 100:.2f}% "
        f"(Acumulada: {var_acumulada:.2f}%)"
    )

    # Avalia o cenario limite sem rotulos (0%) comparando particionamento por centroide e densidade
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters_kmeans = kmeans.fit_predict(X_train_scaled)

    dbscan = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
    clusters_dbscan = dbscan.fit_predict(X_train_scaled)

    # Calcula metricas externas (ARI e NMI) e a taxa de pontos isolados como ruido pelo DBSCAN
    perc_ruido = np.mean(clusters_dbscan == -1) * 100
    n_clusters_dbscan = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)

    df_cluster = pd.DataFrame([
        {
            "Algoritmo": "K-Médias",
            "Clusters Encontrados": n_clusters,
            "Ruído (%)": 0.0,
            "ARI": adjusted_rand_score(y_train, clusters_kmeans),
            "NMI": normalized_mutual_info_score(y_train, clusters_kmeans),
        },
        {
            "Algoritmo": f"DBSCAN (eps={eps_dbscan}, min_samples={min_samples_dbscan})",
            "Clusters Encontrados": n_clusters_dbscan,
            "Ruído (%)": perc_ruido,
            "ARI": adjusted_rand_score(y_train, clusters_dbscan),
            "NMI": normalized_mutual_info_score(y_train, clusters_dbscan),
        },
    ])

    print("\nResultados Etapa 5 - Agrupamento (0% Rotulado):")
    print(df_cluster.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Configura a grade de taxas de rotulagem e as repeticoes para medir estabilidade estatistica
    taxas_rotulados = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    sementes = [10, 20, 30, 40, 50]
    nomes_modelos = [
        "Label Spreading (Semi)",
        "Self-Training LR (Semi)",
        "KNN (Baseline)",
        "Regressão Logística (Baseline)",
    ]

    registros = []
    predicoes_por_modelo = {m: {t: [] for t in taxas_rotulados} for m in nomes_modelos}

    # Mascara os rotulos excedentes como -1 e compara propagacao semi-supervisionada contra baselines
    for taxa in taxas_rotulados:
        for seed in sementes:
            idx_rotulados, _ = train_test_split(
                np.arange(len(y_train)),
                train_size=taxa,
                stratify=y_train,
                random_state=seed,
            )

            y_train_semi = np.full_like(y_train, -1)
            y_train_semi[idx_rotulados] = y_train[idx_rotulados]

            modelos = {
                "Label Spreading (Semi)": LabelSpreading(kernel="knn", n_neighbors=7),
                "Self-Training LR (Semi)": SelfTrainingClassifier(
                    LogisticRegression(max_iter=1000, random_state=seed)
                ),
                "KNN (Baseline)": KNeighborsClassifier(n_neighbors=5),
                "Regressão Logística (Baseline)": LogisticRegression(
                    max_iter=1000, random_state=seed
                ),
            }

            for nome_modelo, modelo in modelos.items():
                if "(Semi)" in nome_modelo:
                    modelo.fit(X_train_scaled, y_train_semi)
                else:
                    modelo.fit(X_train_scaled[idx_rotulados], y_train[idx_rotulados])

                y_pred = modelo.predict(X_test_scaled)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="macro")

                registros.append({
                    "Taxa": taxa,
                    "Modelo": nome_modelo,
                    "Seed": seed,
                    "Acurácia": acc,
                    "F1_Macro": f1,
                })
                predicoes_por_modelo[nome_modelo][taxa].append((f1, y_pred))

    # Consolida media e desvio padrao de acuracia e F1-macro por taxa de rotulagem
    df_resultados = pd.DataFrame(registros)
    df_resumo = (
        df_resultados.groupby(["Taxa", "Modelo"], sort=False)
        .agg(
            Acuracia_Media=("Acurácia", "mean"),
            Acuracia_DP=("Acurácia", "std"),
            F1_Media=("F1_Macro", "mean"),
            F1_DP=("F1_Macro", "std"),
        )
        .reset_index()
    )

    tabela_f1 = df_resumo.pivot(index="Taxa", columns="Modelo", values="F1_Media")[nomes_modelos]
    tabela_acc = df_resumo.pivot(index="Taxa", columns="Modelo", values="Acuracia_Media")[nomes_modelos]

    print("\nResultados Etapa 5 - F1-Score Macro Médio no Teste (5 repetições):")
    print(tabela_f1.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nResultados Etapa 5 - Acurácia Média no Teste (5 repetições):")
    print(tabela_acc.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nTabela Detalhada (Média +/- Desvio Padrão):")
    df_formatado = df_resumo.copy()
    df_formatado["Acurácia"] = df_formatado.apply(
        lambda r: f"{r['Acuracia_Media']:.4f} +/- {r['Acuracia_DP']:.4f}", axis=1
    )
    df_formatado["F1_Macro"] = df_formatado.apply(
        lambda r: f"{r['F1_Media']:.4f} +/- {r['F1_DP']:.4f}", axis=1
    )
    print(df_formatado[["Taxa", "Modelo", "Acurácia", "F1_Macro"]].to_string(index=False))

    # Grafico 1: compara a distribuicao das tres variaveis mais discriminativas entre as classes
    df_box = pd.DataFrame(X, columns=nomes_features)[top_3_features].copy()
    df_box["Classe"] = [nomes_classes[idx] for idx in y]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, feat in zip(axes, top_3_features):
        sns.boxplot(
            data=df_box,
            x="Classe",
            y=feat,
            hue="Classe",
            dodge=False,
            palette="Set2",
            ax=ax,
        )
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_title(feat, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Valor original")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        f"{nome_dataset} - Distribuição dos 3 Atributos Mais Discriminativos",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(pasta_figuras, f"{nome_dataset}_1_boxplots.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Grafico 2: contrasta os rotulos verdadeiros com os agrupamentos obtidos sem supervisao no espaco PCA
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    paleta_clusters = sns.color_palette("tab10", n_colors=max(n_clusters, n_clusters_dbscan, 3))

    for cls_idx, cls_nome in enumerate(nomes_classes):
        mask = y_train == cls_idx
        axes[0].scatter(
            X_train_pca[mask, 0],
            X_train_pca[mask, 1],
            label=cls_nome,
            color=paleta_clusters[cls_idx],
            alpha=0.8,
            edgecolors="w",
            s=55,
        )
    axes[0].set_title("(a) Rótulos Reais", fontweight="bold")
    axes[0].legend(title="Classe", frameon=True)

    for c_id in range(n_clusters):
        mask = clusters_kmeans == c_id
        axes[1].scatter(
            X_train_pca[mask, 0],
            X_train_pca[mask, 1],
            label=f"Cluster {c_id}",
            color=paleta_clusters[c_id],
            alpha=0.8,
            edgecolors="w",
            s=55,
        )
    axes[1].set_title(f"(b) K-Médias (k={n_clusters})", fontweight="bold")
    axes[1].legend(title="Cluster", frameon=True)

    clusters_unicos_db = sorted(set(clusters_dbscan))
    for c_id in clusters_unicos_db:
        mask = clusters_dbscan == c_id
        if c_id == -1:
            axes[2].scatter(
                X_train_pca[mask, 0],
                X_train_pca[mask, 1],
                label="Ruído (-1)",
                color="#333333",
                marker="x",
                alpha=0.85,
                s=60,
            )
        else:
            axes[2].scatter(
                X_train_pca[mask, 0],
                X_train_pca[mask, 1],
                label=f"Cluster {c_id}",
                color=paleta_clusters[c_id % len(paleta_clusters)],
                alpha=0.8,
                edgecolors="w",
                s=55,
            )
    axes[2].set_title(f"(c) DBSCAN (eps={eps_dbscan})", fontweight="bold")
    axes[2].legend(title="Agrupamento", frameon=True)

    for ax in axes:
        ax.set_xlabel(f"PC1 ({var_explicada[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_explicada[1] * 100:.1f}%)")
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(
        f"{nome_dataset} - Projeção PCA 2D: Rótulos Reais vs. Agrupamento (0% Rotulado)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(pasta_figuras, f"{nome_dataset}_2_pca_clustering.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Grafico 3: traca a evolucao do F1-macro medio e faixa de variabilidade conforme cresce a rotulagem
    fig, ax = plt.subplots(figsize=(11, 6.5))
    estilos = {
        "Label Spreading (Semi)": {"cor": "#1f77b4", "estilo": "-", "marcador": "o"},
        "Self-Training LR (Semi)": {"cor": "#2ca02c", "estilo": "-", "marcador": "s"},
        "KNN (Baseline)": {"cor": "#ff7f0e", "estilo": "--", "marcador": "^"},
        "Regressão Logística (Baseline)": {"cor": "#d62728", "estilo": "--", "marcador": "d"},
    }

    for nome_modelo in nomes_modelos:
        dados_m = df_resumo[df_resumo["Modelo"] == nome_modelo].sort_values("Taxa")
        x_vals = dados_m["Taxa"].values
        y_vals = dados_m["F1_Media"].values
        y_std = dados_m["F1_DP"].values
        cfg = estilos[nome_modelo]

        ax.plot(
            x_vals,
            y_vals,
            label=nome_modelo,
            color=cfg["cor"],
            linestyle=cfg["estilo"],
            marker=cfg["marcador"],
            linewidth=2.4,
            markersize=7,
        )
        ax.fill_between(
            x_vals,
            np.clip(y_vals - y_std, 0, 1),
            np.clip(y_vals + y_std, 0, 1),
            color=cfg["cor"],
            alpha=0.15,
        )

    ax.set_title(
        f"{nome_dataset} - Curva de Aprendizado (F1-Score Macro vs. Taxa de Rotulagem)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Proporção de Dados Rotulados no Treino")
    ax.set_ylabel("F1-Score Macro (Conjunto de Teste)")
    ax.set_xticks(taxas_rotulados)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(pasta_figuras, f"{nome_dataset}_3_curva_aprendizado.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Grafico 4: compara os erros por classe do melhor modelo semi-supervisionado nos extremos de 5% e 70%
    modelos_semi = ["Label Spreading (Semi)", "Self-Training LR (Semi)"]
    medias_semi = (
        df_resumo[df_resumo["Modelo"].isin(modelos_semi)]
        .groupby("Modelo")["F1_Media"]
        .mean()
    )
    melhor_semi = medias_semi.idxmax()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, taxa_extrema in zip(axes, [0.05, 0.70]):
        lista_runs = predicoes_por_modelo[melhor_semi][taxa_extrema]
        f1_medio_taxa = np.mean([item[0] for item in lista_runs])
        _, y_pred_rep = min(lista_runs, key=lambda item: abs(item[0] - f1_medio_taxa))

        cm = confusion_matrix(y_test, y_pred_rep)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nomes_classes)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        f1_rep = f1_score(y_test, y_pred_rep, average="macro")
        ax.set_title(
            f"Taxa Rotulada = {taxa_extrema:.2f} (F1 = {f1_rep:.4f})",
            fontweight="bold",
        )
        ax.set_xlabel("Classe Predita")
        ax.set_ylabel("Classe Real")

    fig.suptitle(
        f"{nome_dataset} - Matrizes de Confusão: {melhor_semi} (5% vs. 70% Rotulado)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(pasta_figuras, f"{nome_dataset}_4_matrizes_confusao.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"\nGráficos salvos em '{pasta_figuras}' em 300 DPI.")
    return df_cluster, df_resumo


# Carrega o conjunto Breast Cancer Wisconsin, executa a pipeline e exporta os CSVs de metricas
if __name__ == "__main__":
    dados_cancer = load_breast_cancer()
    df_cluster, df_resumo = executar_pipeline_kdd(
        X=dados_cancer.data,
        y=dados_cancer.target,
        nomes_features=list(dados_cancer.feature_names),
        nomes_classes=["Maligno", "Benigno"],
        nome_dataset="breast_cancer",
        n_clusters=2,
        eps_dbscan=4.5,
        min_samples_dbscan=5,
        top_3_features=["mean radius", "mean concave points", "mean perimeter"],
    )

    pasta_metricas = os.path.join("results_breast_cancer", "metrics")
    os.makedirs(pasta_metricas, exist_ok=True)
    df_cluster.to_csv(
        os.path.join(pasta_metricas, "Breast_Cancer_clustering.csv"), index=False
    )
    df_resumo.to_csv(
        os.path.join(pasta_metricas, "Breast_Cancer_semi_supervisionado.csv"), index=False
    )
    print(f"Tabelas CSV salvas em '{pasta_metricas}'.")