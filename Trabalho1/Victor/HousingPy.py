"""California Housing — laboratório interativo: regressão linear vs. PCR (e variantes).

Uso:
    pip install streamlit numpy pandas matplotlib scikit-learn
    streamlit run app_housing.py
Coloque o housing.csv na mesma pasta ou envie o arquivo pela barra lateral.
"""
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


def ajuda(chave):
    return f"{GUIA[chave][1]}\n\n{GUIA[chave][2]}"


# ----------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------
@st.cache_data
def ler_csv(origem):
    return pd.read_csv(origem)


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


def montar(cfg, k=None):
    passos = [("imputer", SimpleImputer(strategy=cfg["imput"]))]
    if cfg["scale"] or k is not None:
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


def metricas(nome, modelo, Xtr, ytr, Xte, yte):
    linha = {"Modelo": nome}
    for rot, Xs, ys in [("treino", Xtr, ytr), ("teste", Xte, yte)]:
        p = modelo.predict(Xs)
        linha[f"R² ({rot})"] = r2_score(ys, p)
        linha[f"RMSE ({rot})"] = np.sqrt(mean_squared_error(ys, p))
        linha[f"MAE ({rot})"] = mean_absolute_error(ys, p)
    return linha


# ----------------------------------------------------------------------------
# Barra lateral: formulário de hiperparâmetros
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Parâmetros")
arquivo = st.sidebar.file_uploader("housing.csv (opcional)", type="csv")
if arquivo is None and not CSV_PADRAO.exists():
    st.error("Coloque o housing.csv na mesma pasta do app ou envie-o pela barra lateral.")
    st.stop()
df_raw = ler_csv(arquivo if arquivo is not None else CSV_PADRAO)
todas = [c for c in df_raw.columns if c not in (TARGET, "ocean_proximity")]

with st.sidebar.form("form"):
    st.caption("Passe o mouse no (?) de cada controle para ver o que ele faz. Aplique no botão final.")
    st.subheader("Dados")
    test_size = st.slider("Proporção de teste", 0.1, 0.5, 0.2, 0.05, help=ajuda("test_size"))
    seed = st.number_input("Semente aleatória", 0, 9999, 42, help=ajuda("seed"))
    imput = st.selectbox("Imputação", ["median", "mean", "most_frequent"], help=ajuda("imput"))
    ocean = st.radio("ocean_proximity", ["Remover", "One-hot"], horizontal=True, help=ajuda("ocean"))
    feats = st.checkbox("Criar atributos derivados", help=ajuda("feats"))
    logy = st.checkbox("Aplicar log no alvo", help=ajuda("logy"))
    sel = st.multiselect("Atributos numéricos usados", todas, default=todas, help=ajuda("sel"))

    st.subheader("Modelo")
    scale = st.checkbox("Padronizar atributos", True, help=ajuda("scale"))
    modelo = st.selectbox("Regressor final", ["Linear", "Ridge", "Lasso"], help=ajuda("modelo"))
    alpha = st.number_input("Alpha (Ridge/Lasso)", 0.0, 1e6, 1.0, help=ajuda("alpha"))

    st.subheader("PCA")
    usa_pca = st.checkbox("Usar PCA (PCR)", True, help=ajuda("pca"))
    kmodo = st.radio("Escolha de k", ["Manual", "Variância acumulada", "Validação cruzada"], help=ajuda("kmodo"))
    k_manual = st.slider("k (manual)", 1, 14, 5, help=ajuda("k"))
    var_corte = st.slider("Corte de variância", 0.5, 0.999, 0.90, 0.01, help=ajuda("var"))
    folds = st.slider("Dobras da CV", 3, 10, 5, help=ajuda("folds"))
    st.form_submit_button("▶ Treinar / atualizar", use_container_width=True)

cfg = dict(imput=imput, ocean=ocean, feats=feats, logy=logy, scale=scale, modelo=modelo, alpha=alpha)

# ----------------------------------------------------------------------------
# Pipeline de dados e modelos
# ----------------------------------------------------------------------------
X, y = preparar(df_raw, cfg)
extras = [c for c in X.columns if c not in todas]          # derivados e dummies
X = X[[c for c in sel if c in X.columns] + extras]
if X.shape[1] < 1:
    st.warning("Selecione ao menos um atributo.")
    st.stop()
n_feat = X.shape[1]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=int(seed))

pre = Pipeline([("i", SimpleImputer(strategy=imput)), ("s", StandardScaler())])
Xs = pre.fit_transform(X_tr)
pca_full = PCA().fit(Xs)
evr = pca_full.explained_variance_ratio_
cum = np.cumsum(evr)

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

base = montar(cfg).fit(X_tr, y_tr)
pcr = montar(cfg, K).fit(X_tr, y_tr) if usa_pca else None
modelos = [("Sem PCA", base)] + ([(f"PCR (k = {K})", pcr)] if usa_pca else [])

# ----------------------------------------------------------------------------
# Abas
# ----------------------------------------------------------------------------
st.title("🏠 California Housing Lab")
st.caption(f"{X_tr.shape[0]} linhas de treino · {X_te.shape[0]} de teste · {n_feat} atributos · "
           f"regressor: {modelo}" + (f" · k = {K}" if usa_pca else ""))
t_guia, t_dados, t_diag, t_res, t_k, t_cv = st.tabs(
    ["📖 Guia", "📊 Dados", "🔬 Diagnóstico/PCA", "📈 Resultados", "🎚️ Varredura de k", "🔁 Validação cruzada"])

with t_guia:
    st.markdown("### O que cada parâmetro faz\nExpanda um item para ver o que ele é e o que muda ao ajustá-lo.")
    for _, (titulo, o_que, efeito) in GUIA.items():
        with st.expander(titulo):
            st.markdown(f"**O que é:** {o_que}\n\n**Efeito prático:** {efeito}")
    st.info("Fluxo: imputação → padronização → (PCA) → regressão. Tudo é ajustado só no treino e apenas aplicado "
            "no teste, o que evita vazamento de dados.")

with t_dados:
    st.dataframe(df_raw.head(20), use_container_width=True)
    st.dataframe(df_raw.describe().T, use_container_width=True)
    aus = df_raw.isna().sum()
    st.write("Valores ausentes:", aus[aus > 0].to_dict() or "nenhum")
    teto = y.max()
    st.write(f"**Censura:** {(y == teto).mean():.2%} dos alvos estão exatamente em {teto:,.0f}.")

with t_diag:
    corr = np.corrcoef(Xs, rowvar=False)
    if n_feat > 1:
        vif = pd.Series(np.diag(np.linalg.pinv(corr)), index=X.columns, name="VIF").sort_values(ascending=False)
        st.markdown("**VIF** (acima de 5 indica colinearidade relevante)")
        st.dataframe(vif.to_frame().style.format("{:.2f}"), use_container_width=True)
    c1, c2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(range(1, len(evr) + 1), evr, label="Individual")
    ax.plot(range(1, len(evr) + 1), cum, "o-", color="tab:red", label="Acumulada")
    ax.axhline(var_corte, ls="--", color="gray")
    ax.set(xlabel="Componente", ylabel="Variância", title="Scree plot")
    ax.legend()
    c1.pyplot(fig)
    cargas = pd.DataFrame(pca_full.components_.T, index=X.columns,
                          columns=[f"PC{i+1}" for i in range(len(evr))])
    c2.markdown("**Cargas** (peso de cada atributo em cada componente)")
    c2.dataframe(cargas.iloc[:, :min(4, n_feat)].style.format("{:.3f}"), use_container_width=True)
    st.write(f"k pelo corte de {var_corte:.0%}: **{k_var}**" + (f" · k por CV: **{k_cv}**" if k_cv else ""))

with t_res:
    tab = pd.DataFrame([metricas(n, m, X_tr, y_tr, X_te, y_te) for n, m in modelos])
    st.dataframe(tab.set_index("Modelo").style.format("{:,.4f}"), use_container_width=True)
    if modelo != "Linear":
        coefs = base.regressor_[-1].coef_ if logy else base[-1].coef_
    else:
        coefs = base.regressor_[-1].coef_ if logy else base[-1].coef_
    st.markdown("**Coeficientes do modelo sem PCA**" + (" (escala do log do alvo)" if logy else ""))
    st.bar_chart(pd.Series(coefs, index=X.columns).sort_values())

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
    pb = preds[0][1]
    st.caption(f"Previsões negativas ({preds[0][0]}): {(pb < 0).mean():.2%}")

with t_k:
    st.write("Mostra o R² de teste para cada k. É **ilustrativo**: não use esta curva para escolher k.")
    if st.checkbox("Calcular varredura"):
        linhas = []
        for k in range(1, n_feat + 1):
            p = montar(cfg, k).fit(X_tr, y_tr).predict(X_te)
            linhas.append({"k": k, "R² teste": r2_score(y_te, p), "RMSE teste": np.sqrt(mean_squared_error(y_te, p))})
        v = pd.DataFrame(linhas)
        st.dataframe(v.set_index("k").style.format("{:,.4f}"))
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot(v["k"], v["R² teste"], "o-", label="PCR")
        ax.axhline(r2_score(y_te, base.predict(X_te)), ls="--", color="tab:green", label="Sem PCA")
        ax.axvline(K, ls=":", color="gray", label=f"k atual = {K}")
        ax.set(xlabel="k", ylabel="R² teste")
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig)

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