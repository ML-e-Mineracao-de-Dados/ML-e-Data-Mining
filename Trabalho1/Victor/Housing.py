"""California Housing (Kaggle) — regressão linear crua vs. PCR.

Uso: coloque este arquivo e o housing.csv na mesma pasta e rode:
    python california_housing_pessoa1.py
Dependências: pip install numpy pandas matplotlib scipy scikit-learn
"""


# %% [markdown]
# # California Housing (Kaggle) — Regressão linear crua vs. regressão sobre componentes principais
#
# **Pessoa 1 · Mineração de Dados e Aprendizado de Máquina**
#
# Decisões de projeto (fixadas antes de olhar os resultados):
#
# | Decisão | Escolha |
# |---|---|
# | Versão do conjunto | Kaggle (`housing.csv`) |
# | Valores ausentes (`total_bedrooms`) | Imputação pela **mediana calculada apenas no treino** |
# | `ocean_proximity` | **Removida** por completo |
# | Linha de base | Regressão linear **crua**: sem engenharia de atributos, sem tocar no alvo |
# | Comparação | Regressão linear vs. PCR (padronização → PCA → regressão linear) |
# | Divisão | 80/20, semente fixa; tudo encadeado em `Pipeline` para evitar vazamento de dados |
#
# Correspondência com as etapas do processo de descoberta de conhecimento: **etapas 2–3** (seleção e pré-processamento) nas células 2–4; **etapa 4** (transformação, PCA) na seção de diagnóstico; **etapa 5** (mineração) nos dois modelos; **etapa 6** (avaliação e interpretação) nas métricas, nas figuras e nas observações finais.

# %% [markdown]
# ## 1. Configuração

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import os
from pathlib import Path

try:
    PASTA = Path(__file__).resolve().parent
except NameError:   # ex.: console interativo
    PASTA = Path.cwd()
os.chdir(PASTA)            # figuras e tabelas são salvas nesta pasta
CSV_PATH = PASTA / "housing.csv"
TARGET = "median_house_value"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAR_CORTE = 0.90           # variância acumulada mínima para escolher k no PCR

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25})
pd.options.display.float_format = "{:,.4f}".format
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def display(obj):
    print(obj.to_string() if hasattr(obj, "to_string") else obj)


# %% [markdown]
# ## 2. Carregamento e inspeção (etapa 2)

# %%
df = pd.read_csv(CSV_PATH)
print("Dimensões:", df.shape)
display(df.head())
nulos = df.isna().sum()
print("Valores ausentes por coluna:")
print(nulos[nulos > 0] if (nulos > 0).any() else "nenhum")
display(df.describe().T)

# %% [markdown]
# ## 3. Seleção de atributos e divisão treino/teste (etapas 2–3)

# %%
colunas_com_nulos = list(nulos[nulos > 0].index)

df = df.drop(columns=["ocean_proximity"], errors="ignore")   # decisão: coluna removida
X = df.drop(columns=[TARGET])
y = df[TARGET]
print("Preditores:", list(X.columns))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
n_nulos_treino = int(X_train.isna().sum().sum())
n_nulos_teste = int(X_test.isna().sum().sum())
print(f"Treino: {X_train.shape[0]} linhas · Teste: {X_test.shape[0]} linhas")
print(f"Nulos nos preditores — treino: {n_nulos_treino} · teste: {n_nulos_teste}")

# %% [markdown]
# ## 4. Pipelines (etapas 3–5)
#
# Uma única função monta os dois modelos. Sem `k`, temos a **linha de base**; com `k`, inserimos o PCA entre a padronização e a regressão. A imputação, a padronização e o PCA são ajustados **somente no treino** e apenas aplicados ao teste.
#
# A padronização não altera as previsões da regressão linear ordinária; ela é mantida na linha de base para que os coeficientes fiquem comparáveis e para que as duas pontas do experimento difiram **apenas** pelo PCA.

# %%
def make_pipe(k=None):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if k is not None:
        steps.append(("pca", PCA(n_components=k, random_state=RANDOM_STATE)))
    steps.append(("lr", LinearRegression()))
    return Pipeline(steps)


base = make_pipe().fit(X_train, y_train)
pred_base = base.predict(X_test)


def metricas(nome, pipe):
    linha = {"Modelo": nome}
    for rot, Xs, ys in [("treino", X_train, y_train), ("teste", X_test, y_test)]:
        p = pipe.predict(Xs)
        mse = mean_squared_error(ys, p)
        linha[f"MSE ({rot})"] = mse
        linha[f"RMSE ({rot})"] = np.sqrt(mse)
        linha[f"R² ({rot})"] = r2_score(ys, p)
    return linha

# %% [markdown]
# ## 5. Diagnóstico de colinearidade e PCA (etapa 4)
#
# O PCA é **não supervisionado**: opera só sobre os preditores, já imputados e padronizados no treino. Aqui medimos quanta redundância existe (fatores de inflação da variância, VIF), quanta variância cada componente captura e o que cada um representa (cargas).

# %%
pre = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
Xtr_s = pre.fit_transform(X_train)

# VIF: diagonal da inversa da matriz de correlação
corr = np.corrcoef(Xtr_s, rowvar=False)
vif = pd.Series(np.diag(np.linalg.inv(corr)), index=X.columns, name="VIF").sort_values(ascending=False)
print("Fatores de inflação da variância (VIF > 5 indica colinearidade relevante):")
display(vif.to_frame())

pca_full = PCA(random_state=RANDOM_STATE).fit(Xtr_s)
evr = pca_full.explained_variance_ratio_
cum = np.cumsum(evr)
K = int(np.searchsorted(cum, VAR_CORTE) + 1)   # menor k com variância acumulada >= VAR_CORTE
print(f"k escolhido pelo critério de {VAR_CORTE:.0%} de variância acumulada: {K}")

variancia = pd.DataFrame({
    "componente": [f"PC{i+1}" for i in range(len(evr))],
    "variância explicada": evr,
    "acumulada": cum,
})
display(variancia)

cargas = pd.DataFrame(pca_full.components_.T, index=X.columns,
                      columns=[f"PC{i+1}" for i in range(len(evr))])
print("Cargas (loadings) — como cada componente combina os atributos:")
display(cargas.iloc[:, :4])

fig, ax = plt.subplots(figsize=(6.4, 4))
ax.bar(range(1, len(evr) + 1), evr, label="Variância explicada")
ax.plot(range(1, len(evr) + 1), cum, "o-", color="tab:red", label="Acumulada")
ax.axhline(VAR_CORTE, ls="--", color="gray")
ax.set_xlabel("Componente principal")
ax.set_ylabel("Proporção da variância")
ax.set_title("Scree plot")
ax.legend()
fig.tight_layout()
fig.savefig("fig_scree.png")

# %% [markdown]
# ## 6. Quantos componentes? Curva de R² de teste em função de k
#
# Em vez de arbitrar o corte, varremos k = 1…8. Com k = 8 o PCR coincide exatamente com a linha de base, o que serve de **verificação de sanidade** do pipeline.

# %%
n_feat = X.shape[1]
linhas = []
for k in range(1, n_feat + 1):
    m = make_pipe(k).fit(X_train, y_train)
    p = m.predict(X_test)
    linhas.append({"k": k, "R² teste": r2_score(y_test, p),
                   "RMSE teste": np.sqrt(mean_squared_error(y_test, p))})
varredura = pd.DataFrame(linhas)
display(varredura)

pcr_full = make_pipe(n_feat).fit(X_train, y_train)
print("Sanidade — PCR com todos os componentes == linha de base?",
      np.allclose(pcr_full.predict(X_test), pred_base))

r2_base = r2_score(y_test, pred_base)
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.plot(varredura["k"], varredura["R² teste"], "o-", label="PCR")
ax.axhline(r2_base, ls="--", color="tab:green", label="Linha de base (k = 8)")
ax.axvline(K, ls=":", color="gray", label=f"k escolhido = {K}")
ax.set_xlabel("Número de componentes (k)")
ax.set_ylabel("R² no teste")
ax.set_title("Desempenho do PCR vs. número de componentes")
ax.legend()
fig.tight_layout()
fig.savefig("fig_curva_k.png")

# %% [markdown]
# ## 7. Modelo final e tabela de métricas (etapa 5)

# %%
pcr = make_pipe(K).fit(X_train, y_train)
pred_pcr = pcr.predict(X_test)

tabela = pd.DataFrame([
    metricas("Regressão linear (crua)", base),
    metricas(f"PCR (k = {K})", pcr),
])
display(tabela)
tabela.to_csv("tabela_metricas.csv", index=False)

coef = pd.Series(base.named_steps["lr"].coef_, index=X.columns,
                 name="coeficiente (padronizado)").sort_values()
print("Coeficientes da linha de base (atributos padronizados):")
display(coef.to_frame())

# %% [markdown]
# ## 8. Figuras pedidas (etapa 6)
#
# Cada figura tem dois painéis: à esquerda a linha de base, à direita o PCR.

# %%
modelos = [("Regressão linear (crua)", pred_base), (f"PCR (k = {K})", pred_pcr)]

# Figura 1 — Valores reais vs. preditos
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
lo = min(y_test.min(), min(p.min() for _, p in modelos))
hi = max(y_test.max(), max(p.max() for _, p in modelos))
for ax, (nome, p) in zip(axes, modelos):
    ax.scatter(y_test, p, s=5, alpha=0.25)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="Predição perfeita")
    ax.set_title(f"{nome}\nR² = {r2_score(y_test, p):.3f}")
    ax.set_xlabel("Valor real")
axes[0].set_ylabel("Valor predito")
axes[0].legend(loc="upper left")
fig.suptitle("Valores reais vs. preditos (conjunto de teste)")
fig.tight_layout()
fig.savefig("fig_real_vs_predito.png")

# Figura 2 — Análise de resíduos
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
for ax, (nome, p) in zip(axes, modelos):
    res = y_test - p
    ax.scatter(p, res, s=5, alpha=0.25)
    ax.axhline(0, color="r", ls="--", lw=1.2)
    ax.set_title(nome)
    ax.set_xlabel("Valor predito")
axes[0].set_ylabel("Resíduo (real − predito)")
fig.suptitle("Análise de resíduos (conjunto de teste)")
fig.tight_layout()
fig.savefig("fig_residuos.png")

# %% [markdown]
# ## 9. Números de apoio ao passo 6

# %%
teto = y.max()
res_base = y_test - pred_base
rho = spearmanr(pred_base, np.abs(res_base)).correlation

print(f"Teto do alvo: {teto:,.0f} — {(y == teto).mean():.2%} dos registros estão exatamente no teto (censura).")
print(f"Previsões negativas na linha de base: {(pred_base < 0).mean():.2%}.")
print(f"Correlação de Spearman entre previsão e |resíduo|: {rho:.3f} "
      "(positiva e relevante sugere heterocedasticidade).")
print(f"R² teste — linha de base: {r2_base:.4f} · PCR (k = {K}): {r2_score(y_test, pred_pcr):.4f} "
      f"· diferença: {r2_base - r2_score(y_test, pred_pcr):+.4f}")

# %% [markdown]
# ### Observações (preencher após rodar)
#
# Roteiro para o passo 6 — responda com os números das células anteriores:
#
# 1. **Colinearidade:** quais atributos têm VIF alto e o que o PC1 representa (ver cargas)?
# 2. **Desempenho:** o PCR com k componentes supera, iguala ou perde para a linha de base? Qual o custo em R² e RMSE?
# 3. **Resíduos:** há funil (heterocedasticidade) ou curvatura? Qual a faixa horizontal visível no gráfico real vs. predito (censura em ~500 mil)?
# 4. **Interpretabilidade:** o que se ganha e o que se perde ao trocar atributos por componentes?
# 5. **Hipótese inicial (etapa 1) revisada?** O modelo linear é uma boa primeira aproximação ou o padrão dos resíduos pede outro modelo?

# %% [markdown]
# ## 10. Texto de duas linhas para a entrega

# %%
cols_txt = ", ".join(colunas_com_nulos) if colunas_com_nulos else "nenhuma coluna"
print(f"Valores ausentes ({cols_txt}) imputados pela mediana do treino "
      f"({n_nulos_treino} no treino, {n_nulos_teste} no teste); coluna ocean_proximity removida; "
      "atributos padronizados; nenhum registro apagado nem alvo alterado.")
print(f"Comparamos a regressão linear crua com o PCR ({K} componentes, ≥ {VAR_CORTE:.0%} da variância), "
      f"mesma divisão 80/20 (semente {RANDOM_STATE}).")

# %% Mostrar todas as figuras de uma vez
plt.show()
