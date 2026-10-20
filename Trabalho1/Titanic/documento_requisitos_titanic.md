# Documento de Requisitos e Plano de Execução
## Aprendizado de Máquina — Titanic com Árvore de Decisão

**Versão:** 1.0  
**Tipo de documento:** especificação do projeto.

## 1. Objetivo

Desenvolver, individualmente, um notebook Jupyter (`.ipynb`) que utilize o dataset Titanic para treinar e avaliar um classificador baseado em **Árvore de Decisão**, com `Survived` como variável-alvo. O trabalho deve documentar o tratamento dos dados, apresentar uma tabela com acurácia e F1-score e produzir duas imagens: matriz de confusão e desenho da árvore treinada.

## 2. Problema de aprendizado de máquina

- **Tipo:** aprendizado supervisionado, classificação binária.
- **Objetivo previsto:** `Survived` (0 = não sobreviveu; 1 = sobreviveu).
- **Dataset:** `train.csv` da competição Titanic do Kaggle: https://www.kaggle.com/competitions/titanic/data
- **Unidade de análise:** cada linha representa um passageiro.
- **Observação:** o `test.csv` oficial não possui a coluna `Survived`; portanto, a avaliação das métricas deve usar uma divisão de treino/validação do `train.csv`.

## 3. Escopo

### Incluído

1. Carregar e inspecionar o dataset.
2. Investigar tipos de dados, valores ausentes, duplicatas e distribuição do alvo.
3. Justificar o tratamento aplicado às colunas selecionadas.
4. Separar dados de treino e validação.
5. Treinar um modelo `DecisionTreeClassifier`.
6. Calcular acurácia e F1-score para a classe sobrevivente.
7. Exibir uma tabela de métricas.
8. Gerar e salvar a matriz de confusão e a visualização da árvore.
9. Incluir uma explicação de **duas frases** sobre o tratamento de valores nulos e uma conclusão sobre os resultados.

## 4. Requisitos funcionais e critérios de aceite

| ID | Requisito | Como verificar que foi cumprido |
|---|---|---|
| RF-01 | Carregar `train.csv` no notebook. | O notebook abre o arquivo por caminho relativo e mostra as primeiras linhas. |
| RF-02 | Fazer análise inicial dos dados. | São mostrados dimensões, tipos, estatísticas básicas e contagem de valores ausentes. |
| RF-03 | Examinar a variável-alvo. | É confirmado que `Survived` contém as classes 0 e 1 e é apresentada sua distribuição. |
| RF-04 | Definir as variáveis de entrada. | `Survived` fica em `y`; as colunas preditoras ficam em `X`. A justificativa para excluir identificadores/textos brutos é documentada. |
| RF-05 | Tratar dados ausentes. | O método aplicado a cada coluna com nulos é explicitado e não usa informações do conjunto de validação. |
| RF-06 | Codificar atributos categóricos. | Variáveis como `Sex` e `Embarked` são transformadas em formato aceito pelo modelo. |
| RF-07 | Separar treino e validação. | É usada uma divisão reproduzível, recomendada em 80% treino e 20% validação, com `stratify=y` e `random_state` fixo. |
| RF-08 | Treinar uma Árvore de Decisão. | O notebook treina `DecisionTreeClassifier` e registra os parâmetros escolhidos. |
| RF-09 | Avaliar o modelo. | São calculados accuracy e F1-score usando previsões sobre dados não usados no ajuste. |
| RF-10 | Apresentar tabela de métricas. | Há uma tabela com pelo menos as métricas solicitadas e seus valores resultantes. |
| RF-11 | Gerar matriz de confusão. | Uma imagem legível é salva como `matriz_confusao.png`, com rótulos para as classes 0 e 1. |
| RF-12 | Desenhar a árvore. | Uma imagem legível é salva como `arvore_decisao.png`; nomes das variáveis e classes estão identificados. |
| RF-13 | Criar script(s) de imagens. | Um arquivo `.py` pode ser executado para gerar as duas imagens com o mesmo conjunto de dados, divisão e configuração do modelo documentados no notebook. |
| RF-14 | Documentar os nulos em duas frases. | O notebook contém duas frases objetivas que identificam as colunas e os métodos realmente utilizados. |
| RF-15 | Concluir a análise. | A conclusão comenta as métricas e a matriz de confusão, sem afirmar que o resultado vale para todos os passageiros. |

## 5. Requisitos não funcionais

- O notebook deve ser executado de cima para baixo sem erros, em um ambiente com as dependências instaladas.
- O experimento deve ser reproduzível, utilizando `random_state` fixo na divisão e no modelo.
- O tratamento (imputação e codificação) deve ser ajustado somente com dados de treino. Recomenda-se usar `Pipeline` e `ColumnTransformer` do scikit-learn para evitar vazamento de dados.
- As imagens precisam ter resolução e tamanho suficientes para leitura; uma árvore completa pode ficar ilegível, então a visualização pode limitar a profundidade desenhada, desde que isso seja informado.
- Os resultados numéricos devem ser obtidos ao executar o notebook com o dataset real. Não preencher métricas com exemplos inventados.

## 6. Dados: o que investigar e como decidir

| Coluna | O que fazer no projeto inicial | Justificativa / cuidado |
|---|---|---|
| `Survived` | Usar como variável-alvo `y`. | Não deve ser incluída em `X`, pois isso revelaria a resposta ao modelo. |
| `PassengerId` | Excluir das variáveis preditoras iniciais. | É principalmente um identificador, não uma característica útil por si só. |
| `Pclass` | Manter como atributo numérico ordinal ou testar como categoria. | Registrar qual representação foi escolhida. |
| `Sex` | Manter e codificar. | O modelo não recebe strings diretamente; usar codificação categórica apropriada. |
| `Age` | Verificar nulos e imputar a mediana calculada no treino. | A mediana é menos sensível a valores extremos que a média. |
| `SibSp`, `Parch` | Manter; opcionalmente criar `FamilySize = SibSp + Parch + 1`. | Atributos derivados são opcionais; registrar se foram usados. |
| `Fare` | Verificar tipo, nulos e valores suspeitos; imputar mediana se houver nulos. | Não remover valores altos automaticamente: podem ser tarifas legítimas. |
| `Embarked` | Verificar nulos e imputar a moda calculada no treino; codificar categoria. | A moda corresponde ao valor mais frequente entre os exemplos de treino. |
| `Name`, `Ticket`, `Cabin` | Excluir no modelo inicial ou transformar em atributos derivados numa extensão. | São textos/campos com alta complexidade; a exclusão inicial deve ser registrada. `Cabin` costuma ter muitos nulos. |

**Importante:** primeiro medir a quantidade e a porcentagem de nulos de cada coluna; não presumir que o arquivo baixado é idêntico a uma cópia encontrada em outro lugar. Linhas não devem ser apagadas só porque uma coluna tem nulos. Duplicatas também devem ser investigadas antes de removidas.

## 7. Plano de execução — o que você precisa fazer

### Etapa 1 — Preparar o ambiente

1. Baixar o `train.csv` na página de dados da competição Titanic do Kaggle.
2. Criar a pasta do projeto e colocar o CSV em `data/train.csv` ou ajustar o caminho no notebook.
3. Abrir a pasta no VS Code e selecionar um ambiente Python com Jupyter habilitado.
4. Instalar, se necessário, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter` e `ipykernel`.

### Etapa 2 — Montar o notebook

Organizar o notebook em células Markdown e código nesta ordem:

1. **Título e objetivo:** explicar em poucas linhas o que será previsto.
2. **Bibliotecas e leitura:** importar bibliotecas e carregar o CSV.
3. **Conhecendo os dados:** mostrar `head()`, `shape`, `info()`, `describe()` e tipos das colunas.
4. **Diagnóstico:** mostrar valores ausentes por coluna e sua porcentagem; verificar duplicatas e distribuição de `Survived`.
5. **Decisões de tratamento:** documentar o que manter, excluir, imputar e codificar.
6. **Separação de X e y:** retirar `Survived` de `X`; não usar informações do alvo como preditoras.
7. **Divisão treino/validação:** executar a separação antes de ajustar imputadores e codificadores. Utilizar divisão estratificada e semente fixa.
8. **Pré-processamento:** imputar numéricas com mediana, categóricas com moda e codificar categorias. Ajustar esses passos somente nos dados de treino; preferencialmente usar pipeline.
9. **Treinamento:** criar e ajustar `DecisionTreeClassifier`. Registrar parâmetros como `max_depth`, `min_samples_leaf` e `random_state`; se testar profundidades, comparar os resultados na mesma validação.
10. **Previsão e métricas:** prever os rótulos da validação e calcular acurácia e F1-score.
11. **Tabela de resultados:** preencher a tabela com os valores realmente calculados.
12. **Matriz de confusão:** gerar figura com eixos/rótulos legíveis e salvar a imagem.
13. **Visualização da árvore:** gerar desenho da árvore treinada, incluindo nomes dos atributos e classes, e salvar a imagem.
14. **Explicação dos nulos e conclusão:** escrever as duas frases exigidas sobre os nulos e comentar o que as métricas e a matriz indicam.

### Etapa 3 — Criar o script para as imagens

1. Criar um arquivo, por exemplo, `scripts/gerar_imagens.py`.
2. Fazer o script carregar os mesmos dados e repetir a mesma divisão, pré-processamento e parâmetros do modelo documentados no notebook.
3. Gerar a matriz de confusão e o desenho da árvore com `matplotlib`/`scikit-learn`.
4. Salvar as figuras na pasta `resultados/` com os nomes especificados abaixo.
5. Executar o script a partir da raiz do projeto e verificar que as imagens foram criadas e estão legíveis.
6. Evitar divergência entre notebook e script: modelo, atributos, `random_state` e divisão de dados devem ser os mesmos.

## 8. Métricas obrigatórias

- **Acurácia:** proporção total de previsões corretas entre todas as observações de validação.
- **F1-score:** média harmônica entre precisão e recall. Para este trabalho, calcular explicitamente para a classe positiva `Survived = 1`.

A acurácia pode parecer boa se o modelo favorecer a classe majoritária; por isso, apresentar também F1-score e matriz de confusão. Explicitar que as métricas se referem apenas ao conjunto de validação escolhido.

### Tabela que você deverá preencher após executar o notebook

| Modelo | Conjunto avaliado | Acurácia | F1-score (Survived = 1) |
|---|---|---:|---:|
| Árvore de Decisão | Validação | preencher com o valor obtido | preencher com o valor obtido |

Não registrar uma métrica antes de executar a avaliação. Se comparar diferentes configurações de árvore, adicionar uma linha para cada configuração e identificar seus parâmetros.

## 9. Especificação dos arquivos de imagem

### `matriz_confusao.png`

- Deve ser calculada com `y_val` e as previsões feitas em `X_val`.
- Deve mostrar as classes “Não sobreviveu (0)” e “Sobreviveu (1)”.
- Deve permitir distinguir verdadeiros negativos, falsos positivos, falsos negativos e verdadeiros positivos.
- Deve incluir título e eixos identificados.

### `arvore_decisao.png`

- Deve representar o modelo efetivamente treinado.
- Deve mostrar nomes das variáveis e rótulos das classes.
- Deve ter tamanho/resolução suficientes para leitura.
- Se somente os primeiros níveis forem desenhados por legibilidade, informar isso no notebook; não alterar silenciosamente o modelo só para produzir a imagem.

## 10. Texto de duas frases sobre os nulos

Incluir no notebook uma nota de exatamente duas frases, ajustada ao que você realmente implementou. Exemplo de referência:

> Os valores ausentes de `Age` foram preenchidos com a mediana calculada a partir do conjunto de treino. Os valores ausentes de `Embarked` foram preenchidos com a categoria mais frequente do conjunto de treino; se houver valores ausentes em `Fare`, eles serão preenchidos com a mediana do treino.

Se o diagnóstico mostrar nulos em outras colunas ou você adotar métodos diferentes, revise o texto para corresponder ao código executado.


