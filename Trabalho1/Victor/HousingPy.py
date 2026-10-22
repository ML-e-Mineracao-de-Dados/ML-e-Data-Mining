"""California Housing — laboratório interativo: regressão linear vs. PCR (e variantes).

Uso:
    pip install streamlit numpy pandas matplotlib scikit-learn
    streamlit run app_housing.py
Coloque o housing.csv na mesma pasta ou envie o arquivo pela barra lateral.
"""

# ============================================================================
# 1. IMPORTS E CONFIGURAÇÃO GERAL
# ----------------------------------------------------------------------------
# Bibliotecas usadas, constantes globais (nome do alvo e caminho do CSV padrão)
# e configuração da página do Streamlit.
# ============================================================================
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "median_house_value"
CSV_PADRAO = Path(__file__).resolve().parent / "housing.csv"

st.set_page_config(page_title="California Housing Lab", layout="wide")

# ----------------------------------------------------------------------------
# GUIA: (título, o que é, efeito prático). Usado nos tooltips e na aba Guia.
# ----------------------------------------------------------------------------
# Dicionário com a explicação de cada parâmetro da barra lateral. A mesma fonte
# alimenta os tooltips (?) dos controles e a aba "Guia", evitando texto duplicado.
GUIA = {
    "test_size": ("Proporção de teste",
                  "Fração das linhas reservada para avaliar o modelo; ele nunca vê esses dados no treino.",
                  "Teste maior = estimativa de desempenho mais estável, mas menos dados para treinar. 0,2 é o padrão."),
    "seed": ("Semente aleatória",
             "Número que fixa o sorteio da divisão treino/teste.",
             "Mesma semente = mesmos resultados. Mude-a para ver o quanto o R² oscila só por sorte da divisão."),
    "imput": ("Estratégia de imputação",
              "Como preencher os valores ausentes (só existem em total_bedrooms).",
              "Mediana é robusta a outliers; média é sensível a eles. O valor é calculado só no treino (sem vazamento)."),
    "ocean": ("Tratar ocean_proximity",
              "Coluna categórica (perto do mar, interior...). Pode ser removida ou transformada em colunas 0/1 (one-hot).",
              "Mantê-la costuma melhorar bastante o R², pois captura o efeito litoral que lat/long lineares não pegam."),
    "feats": ("Engenharia de atributos",
              "Cria razões como cômodos por domicílio, quartos por cômodo e pessoas por domicílio.",
              "Reduz a colinearidade (os totais crescem juntos com o tamanho do bloco) e enfraquece a motivação do PCA."),
    "logy": ("Log do alvo",
             "Treina o modelo em log(preço) e converte de volta para dólares na previsão.",
             "Costuma reduzir o 'funil' dos resíduos e eliminar previsões negativas. Altera o alvo: sai da linha de base 'crua'."),
    "scale": ("Padronizar atributos",
              "Deixa cada coluna com média 0 e desvio 1.",
              "Não muda a regressão linear simples, mas é essencial para PCA, Ridge e Lasso, que dependem da escala."),
    "modelo": ("Regressor final",
               "Linear: mínimos quadrados puros. Ridge: penaliza coeficientes grandes (L2). Lasso: pode zerar coeficientes (L1).",
               "Ridge é a alternativa 'suave' ao PCR para colinearidade; Lasso faz seleção de atributos."),
    "alpha": ("Alpha (regularização)",
              "Força da penalidade em Ridge/Lasso.",
              "Alpha ~0 = regressão linear comum. Alpha alto = coeficientes encolhidos, mais viés e menos variância."),
    "pca": ("Usar PCA (PCR)",
            "Insere a projeção em componentes principais entre a padronização e a regressão.",
            "Troca atributos originais por combinações ortogonais. Elimina colinearidade, mas perde interpretabilidade."),
    "kmodo": ("Como escolher k",
              "Manual: você define. Variância: menor k que atinge o corte. Validação cruzada: k com melhor R² médio no treino.",
              "Só a validação cruzada olha o alvo. Escolher k olhando o teste seria vazamento."),
    "k": ("Número de componentes (k)",
          "Quantos componentes principais entram na regressão.",
          "k pequeno = modelo simples, mas pode descartar sinal. k = nº de atributos equivale à regressão sem PCA."),
    "var": ("Corte de variância acumulada",
            "Fração mínima da variância total que os k componentes devem reter.",
            "0,90 é comum. Atenção: variância alta nos preditores não garante poder preditivo sobre o alvo."),
    "folds": ("Dobras da validação cruzada",
              "Em quantas partes o treino é dividido para avaliar cada configuração.",
              "Mais dobras = estimativa mais fiel, porém mais lenta. 5 é o padrão."),
    "sel": ("Atributos usados",
            "Subconjunto de colunas que alimenta o modelo.",
            "Remover colunas testa o quanto cada uma contribui. Retirar latitude/longitude, por exemplo, derruba o R²."),
}


# Formata o texto do tooltip (?) de um controle: "o que é" + "efeito prático".
def ajuda(chave):
    return f"{GUIA[chave][1]}\n\n{GUIA[chave][2]}"


# ============================================================================
# 2. FUNÇÕES AUXILIARES
# ----------------------------------------------------------------------------
# Leitura do CSV, preparação dos atributos, montagem do pipeline de modelagem
# e cálculo das métricas. Nenhuma delas desenha nada na tela.
# ============================================================================

# Lê o CSV uma única vez; o cache do Streamlit evita reler a cada interação.
@st.cache_data
def ler_csv(origem):
    return pd.read_csv(origem)


# Prepara X e y: remove linhas sem alvo, cria atributos derivados (opcional) e
# trata a coluna categórica ocean_proximity (one-hot ou remoção).
def preparar(df, cfg):
    df = df.dropna(subset=[TARGET]).copy()
    if cfg["feats"]:
        df["rooms_per_household"] = df["total_rooms"] / df["households"]
        df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
        df["population_per_household"] = df["population"] / df["households"]
    if "ocean_proximity" in df:
        if cfg["ocean"] == "One-hot":
            df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True, dtype=float)
        else:
            df = df.drop(columns="ocean_proximity")
    return df.drop(columns=TARGET), df[TARGET]


# Monta o pipeline: imputação -> (padronização) -> (PCA com k componentes) -> regressor.
# Se k=None, não há PCA (modelo base). Se logy=True, o alvo é treinado em log
# e convertido de volta para dólares na previsão.
def montar(cfg, k=None):
    passos = [("imputer", SimpleImputer(strategy=cfg["imput"]))]
    if cfg["scale"] or k is not None:        # PCA exige padronização
        passos.append(("scaler", StandardScaler()))
    if k is not None:
        passos.append(("pca", PCA(n_components=k)))
    reg = {"Linear": LinearRegression(), "Ridge": Ridge(alpha=cfg["alpha"]),
           "Lasso": Lasso(alpha=cfg["alpha"], max_iter=20000)}[cfg["modelo"]]
    passos.append(("reg", reg))
    pipe = Pipeline(passos)
    if cfg["logy"]:
        return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
    return pipe


# Calcula R², RMSE e MAE de um modelo no treino e no teste, devolvendo uma
# linha (dict) pronta para virar uma linha da tabela de resultados.
def metricas(nome, modelo, Xtr, ytr, Xte, yte):
    linha = {"Modelo": nome}
    for rot, Xs, ys in [("treino", Xtr, ytr), ("teste", Xte, yte)]:
        p = modelo.predict(Xs)
        linha[f"R² ({rot})"] = r2_score(ys, p)
        linha[f"RMSE ({rot})"] = np.sqrt(mean_squared_error(ys, p))
        linha[f"MAE ({rot})"] = mean_absolute_error(ys, p)
    return linha


# ============================================================================
# 3. BARRA LATERAL: CARREGAMENTO DOS DADOS E FORMULÁRIO DE PARÂMETROS
# ----------------------------------------------------------------------------
# Define de onde vêm os dados (upload ou housing.csv local) e reúne todos os
# hiperparâmetros num st.form: nada é recalculado até o botão ser clicado.
# ============================================================================
st.sidebar.title("⚙️ Parâmetros")
arquivo = st.sidebar.file_uploader("housing.csv (opcional)", type="csv")
if arquivo is None and not CSV_PADRAO.exists():
    st.error("Coloque o housing.csv na mesma pasta do app ou envie-o pela barra lateral.")
    st.stop()
df_raw = ler_csv(arquivo if arquivo is not None else CSV_PADRAO)
todas = [c for c in df_raw.columns if c not in (TARGET, "ocean_proximity")]

with st.sidebar.form("form"):
    st.caption("Passe o mouse no (?) de cada controle para ver o que ele faz. Aplique no botão final.")

    # --- Opções de dados e pré-processamento ---
    st.subheader("Dados")
    test_size = st.slider("Proporção de teste", 0.1, 0.5, 0.2, 0.05, help=ajuda("test_size"))
    seed = st.number_input("Semente aleatória", 0, 9999, 42, help=ajuda("seed"))
    imput = st.selectbox("Imputação", ["median", "mean", "most_frequent"], help=ajuda("imput"))
    ocean = st.radio("ocean_proximity", ["Remover", "One-hot"], horizontal=True, help=ajuda("ocean"))
    feats = st.checkbox("Criar atributos derivados", help=ajuda("feats"))
    logy = st.checkbox("Aplicar log no alvo", help=ajuda("logy"))
    sel = st.multiselect("Atributos numéricos usados", todas, default=todas, help=ajuda("sel"))

    # --- Opções do regressor final ---
    st.subheader("Modelo")
    scale = st.checkbox("Padronizar atributos", True, help=ajuda("scale"))
    modelo = st.selectbox("Regressor final", ["Linear", "Ridge", "Lasso"], help=ajuda("modelo"))
    alpha = st.number_input("Alpha (Ridge/Lasso)", 0.0, 1e6, 1.0, help=ajuda("alpha"))

    # --- Opções de PCA e escolha do número de componentes (k) ---
    st.subheader("PCA")
    usa_pca = st.checkbox("Usar PCA (PCR)", True, help=ajuda("pca"))
    kmodo = st.radio("Escolha de k", ["Manual", "Variância acumulada", "Validação cruzada"], help=ajuda("kmodo"))
    k_manual = st.slider("k (manual)", 1, 14, 5, help=ajuda("k"))
    var_corte = st.slider("Corte de variância", 0.5, 0.999, 0.90, 0.01, help=ajuda("var"))
    folds = st.slider("Dobras da CV", 3, 10, 5, help=ajuda("folds"))
    st.form_submit_button("▶ Treinar / atualizar", use_container_width=True)

# Agrupa as opções que afetam a construção do pipeline (usadas por preparar() e montar()).
cfg = dict(imput=imput, ocean=ocean, feats=feats, logy=logy, scale=scale, modelo=modelo, alpha=alpha)

# ============================================================================
# 4. PREPARAÇÃO DOS DADOS E TREINO DOS MODELOS
# ----------------------------------------------------------------------------
# Gera X/y, divide treino/teste, roda a PCA exploratória (scree plot, VIF, k por
# variância), escolhe k (manual, variância ou CV) e treina o modelo base e o PCR.
# ============================================================================

# --- Construção de X e y, respeitando os atributos selecionados ---
X, y = preparar(df_raw, cfg)
extras = [c for c in X.columns if c not in todas]          # derivados e dummies
X = X[[c for c in sel if c in X.columns] + extras]
if X.shape[1] < 1:
    st.warning("Selecione ao menos um atributo.")
    st.stop()
n_feat = X.shape[1]

# --- Divisão treino/teste ---
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=int(seed))

# --- PCA exploratória (ajustada só no treino): variância explicada por componente ---
pre = Pipeline([("i", SimpleImputer(strategy=imput)), ("s", StandardScaler())])
Xs = pre.fit_transform(X_tr)
pca_full = PCA().fit(Xs)
evr = pca_full.explained_variance_ratio_
cum = np.cumsum(evr)

# --- Escolha de k: pelo corte de variância e, se pedido, por validação cruzada ---
k_var = int(np.searchsorted(cum, var_corte) + 1)
k_cv = None
if usa_pca and kmodo == "Validação cruzada":
    with st.spinner("Buscando k por validação cruzada..."):
        gs = GridSearchCV(montar(cfg, 1), {} if False else
                          {("regressor__pca__n_components" if logy else "pca__n_components"): range(1, n_feat + 1)},
                          cv=KFold(folds, shuffle=True, random_state=int(seed)), scoring="r2")
        gs.fit(X_tr, y_tr)
        k_cv = int(list(gs.best_params_.values())[0])
K = {"Manual": min(k_manual, n_feat), "Variância acumulada": k_var, "Validação cruzada": k_cv}[kmodo]

# --- Treino final: modelo sem PCA (base) e, se ativado, o PCR com k = K ---
base = montar(cfg).fit(X_tr, y_tr)
pcr = montar(cfg, K).fit(X_tr, y_tr) if usa_pca else None
modelos = [("Sem PCA", base)] + ([(f"PCR (k = {K})", pcr)] if usa_pca else [])

# ============================================================================
# 5. INTERFACE: CABEÇALHO E ABAS
# ----------------------------------------------------------------------------
# Título, resumo da configuração atual e criação das seis abas. Cada bloco
# "with t_xxx" abaixo preenche uma aba.
# ============================================================================
st.title("🏠 California Housing Lab")
st.caption(f"{X_tr.shape[0]} linhas de treino · {X_te.shape[0]} de teste · {n_feat} atributos · "
           f"regressor: {modelo}" + (f" · k = {K}" if usa_pca else ""))
t_guia, t_dados, t_diag, t_res, t_k, t_cv = st.tabs(
    ["📖 Guia", "📊 Dados", "🔬 Diagnóstico/PCA", "📈 Resultados", "🎚️ Varredura de k", "🔁 Validação cruzada"])

# ----------------------------------------------------------------------------
# Aba Guia: lista expansível com a explicação de cada parâmetro (vinda de GUIA).
# ----------------------------------------------------------------------------
with t_guia:
    st.markdown("### O que cada parâmetro faz\nExpanda um item para ver o que ele é e o que muda ao ajustá-lo.")
    for _, (titulo, o_que, efeito) in GUIA.items():
        with st.expander(titulo):
            st.markdown(f"**O que é:** {o_que}\n\n**Efeito prático:** {efeito}")
    st.info("Fluxo: imputação → padronização → (PCA) → regressão. Tudo é ajustado só no treino e apenas aplicado "
            "no teste, o que evita vazamento de dados.")

# ----------------------------------------------------------------------------
# Aba Dados: visão geral do CSV (amostra, estatísticas, ausentes e censura do alvo).
# ----------------------------------------------------------------------------
with t_dados:
    st.dataframe(df_raw.head(20), use_container_width=True)
    st.dataframe(df_raw.describe().T, use_container_width=True)
    aus = df_raw.isna().sum()
    st.write("Valores ausentes:", aus[aus > 0].to_dict() or "nenhum")
    teto = y.max()
    st.write(f"**Censura:** {(y == teto).mean():.2%} dos alvos estão exatamente em {teto:,.0f}.")

# ----------------------------------------------------------------------------
# Aba Diagnóstico/PCA: colinearidade (VIF), scree plot e cargas dos componentes.
# ----------------------------------------------------------------------------
with t_diag:
    # VIF: calculado pela diagonal da inversa da matriz de correlação
    corr = np.corrcoef(Xs, rowvar=False)
    if n_feat > 1:
        vif = pd.Series(np.diag(np.linalg.pinv(corr)), index=X.columns, name="VIF").sort_values(ascending=False)
        st.markdown("**VIF** (acima de 5 indica colinearidade relevante)")
        st.dataframe(vif.to_frame().style.format("{:.2f}"), use_container_width=True)
    c1, c2 = st.columns(2)

    # Scree plot: variância individual (barras) e acumulada (linha)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(range(1, len(evr) + 1), evr, label="Individual")
    ax.plot(range(1, len(evr) + 1), cum, "o-", color="tab:red", label="Acumulada")
    ax.axhline(var_corte, ls="--", color="gray")
    ax.set(xlabel="Componente", ylabel="Variância", title="Scree plot")
    ax.legend()
    c1.pyplot(fig)

    # Cargas: quanto cada atributo contribui para os primeiros componentes
    cargas = pd.DataFrame(pca_full.components_.T, index=X.columns,
                          columns=[f"PC{i+1}" for i in range(len(evr))])
    c2.markdown("**Cargas** (peso de cada atributo em cada componente)")
    c2.dataframe(cargas.iloc[:, :min(4, n_feat)].style.format("{:.3f}"), use_container_width=True)
    st.write(f"k pelo corte de {var_corte:.0%}: **{k_var}**" + (f" · k por CV: **{k_cv}**" if k_cv else ""))

# ----------------------------------------------------------------------------
# Aba Resultados: métricas, coeficientes, real vs. predito e resíduos.
# ----------------------------------------------------------------------------
with t_res:
    # Tabela de métricas (treino e teste) para cada modelo
    tab = pd.DataFrame([metricas(n, m, X_tr, y_tr, X_te, y_te) for n, m in modelos])
    st.dataframe(tab.set_index("Modelo").style.format("{:,.4f}"), use_container_width=True)

    # Coeficientes do modelo sem PCA (com log no alvo, o regressor fica dentro de regressor_)
    if modelo != "Linear":
        coefs = base.regressor_[-1].coef_ if logy else base[-1].coef_
    else:
        coefs = base.regressor_[-1].coef_ if logy else base[-1].coef_
    st.markdown("**Coeficientes do modelo sem PCA**" + (" (escala do log do alvo)" if logy else ""))
    st.bar_chart(pd.Series(coefs, index=X.columns).sort_values())

    # Gráficos de diagnóstico: um painel por modelo, com eixos compartilhados
    preds = [(n, m.predict(X_te)) for n, m in modelos]
    lo, hi = min(y_te.min(), *(p.min() for _, p in preds)), max(y_te.max(), *(p.max() for _, p in preds))
    for titulo, tipo in [("Real vs. predito", "rp"), ("Resíduos", "res")]:
        fig, axes = plt.subplots(1, len(preds), figsize=(5.5 * len(preds), 4), sharex=True, sharey=True,
                                 squeeze=False)
        for ax, (n, p) in zip(axes[0], preds):
            if tipo == "rp":
                ax.scatter(y_te, p, s=4, alpha=0.25)
                ax.plot([lo, hi], [lo, hi], "r--")
                ax.set(xlabel="Real", title=f"{n} · R² = {r2_score(y_te, p):.3f}")
            else:
                ax.scatter(p, y_te - p, s=4, alpha=0.25)
                ax.axhline(0, color="r", ls="--")
                ax.set(xlabel="Predito", title=n)
            ax.grid(alpha=0.25)
        axes[0][0].set_ylabel("Predito" if tipo == "rp" else "Resíduo")
        fig.suptitle(titulo)
        st.pyplot(fig)

    # Fração de previsões negativas do modelo base (preço negativo não faz sentido)
    pb = preds[0][1]
    st.caption(f"Previsões negativas ({preds[0][0]}): {(pb < 0).mean():.2%}")

# ----------------------------------------------------------------------------
# Aba Varredura de k: R² de teste para cada k (apenas ilustrativo, sob demanda).
# ----------------------------------------------------------------------------
with t_k:
    st.write("Mostra o R² de teste para cada k. É **ilustrativo**: não use esta curva para escolher k.")
    if st.checkbox("Calcular varredura"):
        # Treina um PCR para cada k de 1 até o nº de atributos
        linhas = []
        for k in range(1, n_feat + 1):
            p = montar(cfg, k).fit(X_tr, y_tr).predict(X_te)
            linhas.append({"k": k, "R² teste": r2_score(y_te, p), "RMSE teste": np.sqrt(mean_squared_error(y_te, p))})
        v = pd.DataFrame(linhas)
        st.dataframe(v.set_index("k").style.format("{:,.4f}"))

        # Curva do R² por k, comparada ao modelo sem PCA e ao k atual
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot(v["k"], v["R² teste"], "o-", label="PCR")
        ax.axhline(r2_score(y_te, base.predict(X_te)), ls="--", color="tab:green", label="Sem PCA")
        ax.axvline(K, ls=":", color="gray", label=f"k atual = {K}")
        ax.set(xlabel="k", ylabel="R² teste")
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig)

# ----------------------------------------------------------------------------
# Aba Validação cruzada: compara os modelos (média ± desvio) só com dados de
# treino, para saber se a diferença entre eles é maior que o ruído.
# ----------------------------------------------------------------------------
with t_cv:
    st.write("Média ± desvio padrão em validação cruzada **só no treino**, para ver se a diferença é maior que o ruído.")
    if st.checkbox("Rodar validação cruzada"):
        kf = KFold(folds, shuffle=True, random_state=int(seed))
        linhas = []
        for n, m in [("Sem PCA", montar(cfg))] + ([(f"PCR (k = {K})", montar(cfg, K))] if usa_pca else []):
            r = cross_validate(m, X_tr, y_tr, cv=kf, scoring={
                "r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"})
            linhas.append({"Modelo": n,
                           "R²": f"{r['test_r2'].mean():.4f} ± {r['test_r2'].std():.4f}",
                           "RMSE": f"{-r['test_rmse'].mean():,.0f} ± {r['test_rmse'].std():,.0f}",
                           "MAE": f"{-r['test_mae'].mean():,.0f} ± {r['test_mae'].std():,.0f}"})
        st.dataframe(pd.DataFrame(linhas).set_index("Modelo"), use_container_width=True)