# Atividade Prática: Comparação e Avaliação de Modelos de Machine Learning — SVM, Árvore de Decisão e Random Forest

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aj1no/ML_Atividade_Pratica_SVM_Arvore_RandomForest/blob/main/Rodolfo_Takemoto.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13%2B-388e3c?style=flat-square)](https://seaborn.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

Este repositório contém a resolução integral da atividade prática **Comparação e Avaliação de Modelos de Machine Learning** da disciplina de **Aprendizado de Máquina I**, ministrada pelo **Prof. Me. Mateus Guilherme Fuini**.

O objetivo do trabalho é implementar, comparar, auditar e interpretar o comportamento de três famílias fundamentais de algoritmos de aprendizado supervisionado — **Support Vector Machines (Linear e RBF)**, **Árvores de Decisão (CART)** e **Random Forests (Ensemble Bagging)** — aplicados à base de dados de diagnóstico médico de câncer de mama (**Breast Cancer Wisconsin Diagnostic**), com foco na prevenção de *Data Leakage*, diagnóstico do dilema viés-variância (*overfitting* vs *underfitting*), sensibilidade de hiperparâmetros e fundamentação técnica clínica na escolha do modelo ótimo.

---

## Informações Acadêmicas

* **Instituição:** Faculdade de Tecnologia (FATEC)
* **Curso:** Ciência de Dados
* **Disciplina:** Aprendizado de Máquina I
* **Professor:** Prof. Me. Mateus Guilherme Fuini
* **Aluno:** Rodolfo Vinicius Cima Takemoto

---

## Descrição do Problema e Dados

O conjunto de dados utilizado é o **Breast Cancer Wisconsin (Diagnostic)**, disponibilizado via `sklearn.datasets.load_breast_cancer()`:

* **Instâncias:** 569 amostras de biópsias por aspiração com agulha fina (PAAF);
* **Atributos:** 30 variáveis preditoras contínuas descrevendo características nucleares celulares (raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, simetria e dimensão fractal), computadas em três métricas: média (*mean*), erro padrão (*SE*) e pior valor observado (*worst*);
* **Variável Alvo (`target`):**
  * `0`: **Maligno** (212 instâncias, **37,26%** — classe minoritária e de alto interesse clínico);
  * `1`: **Benigno** (357 instâncias, **62,74%** — classe majoritária).
* **Valores Ausentes:** 0 (conjunto 100% íntegro).

---

## Estrutura do Pipeline Experimental

O notebook executado [`Rodolfo_Takemoto.ipynb`](Rodolfo_Takemoto.ipynb) contempla 7 etapas metodológicas estruturadas:

1. **Etapa 1 – Conhecendo os Dados:** Inspeção dimensional, tipos de dados, balanceamento das classes, estatísticas descritivas sumarizadas e análise visual bimodal dos atributos `mean radius`, `mean texture` e `mean concave points`.
2. **Etapa 2 – Preparação dos Dados:** Divisão disjunta de **75% treino (426 amostras)** e **25% teste (143 amostras)** com semente fixa (`random_state=42`) e partição estratificada (`stratify=y`), acoplada à arquitetura de `Pipeline` com `StandardScaler` para isolamento total contra *Data Leakage*.
3. **Etapa 3 – Implementação da SVM:** Construção de **SVM Linear** e **SVM RBF**, rastreamento analítico dos vetores de suporte (`n_support_`), varredura de regularização $C \in \{0.01, 0.1, 1.0, 10.0, 100.0\}$ e do raio radial $\gamma \in \{0.001, 0.01, 0.1, 1.0, \text{'scale'}\}$, diagnosticando o colapso por *overfitting* extremo em $\gamma=1.0$.
4. **Etapa 4 – Árvores de Decisão:** Treinamento de árvore CART inicial com `max_depth=3`, plotagem visual da árvore de decisão com `plot_tree()`, interpretação do nó raiz (`worst concave points <= 0.135`) e rastreamento de caminho clínico até a folha pura. Comparação entre critérios de impureza (**Gini vs Entropia**) e curva de acurácia por profundidade ($d \in \{2, 5, 10, \text{None}\}$).
5. **Etapa 5 – Random Forest:** Implementação do modelo ensemble (`n_estimators=100`, `max_features='sqrt'`), demonstração do ganho sobre a árvore única, curva de estabilização da validação ($n \in \{10, 50, 100, 200\}$) e ranqueamento dos **Top 10 atributos mais importantes** pelo método *Mean Decrease in Impurity (MDI)*.
6. **Etapa 6 – Comparação dos Modelos no Teste Final:** Avaliação neutra e cega dos 4 modelos candidatos sobre o conjunto de teste de 143 amostras, computando **Acurácia**, **Precisão**, **Recall** e **F1-score** para a classe crítica (**Maligno, `pos_label=0`**), além do painel $2 \times 2$ de matrizes de confusão destacando a contagem de Falsos Negativos (casos graves de falso diagnóstico benigno).
7. **Etapa 7 – Validação Cruzada Estratificada:** Execução de `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` estritamente sobre a base de treino via `cross_validate()` com múltiplos estimadores, apurando a média e o desvio-padrão de Acurácia, Recall e F1, atestando a robustez dos estimadores.

---

## Síntese dos Resultados Experimentais

### 1. Teste Final: Comparativo dos 4 Modelos (`pos_label=0: Maligno`)

| Modelo | Acurácia Global | Precisão (Maligno) | Recall (Maligno) | F1-Score (Maligno) | Falsos Negativos (FN) | Falsos Positivos (FP) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SVM Linear ($C=1$)** | **98,60%** | **98,11%** | **98,11%** | **98,11%** | **1** | **1** |
| **SVM RBF ($C=1, \gamma=\text{'scale'})$** | 97,90% | 96,30% | **98,11%** | 97,20% | **1** | 2 |
| **Random Forest ($100$ árvores)** | 95,80% | 96,08% | 92,45% | 94,23% | 4 | 2 |
| **Decision Tree ($depth=3$, Gini)** | 94,41% | 94,12% | 90,57% | 92,31% | 5 | 3 |

> **Nota Clínica Fundamental:** No conjunto de teste há **53 pacientes com tumor maligno real**. A SVM Linear e a SVM RBF deixaram de diagnosticar apenas **1 paciente** (98,11% de sensibilidade clínica), enquanto a Random Forest falhou em diagnosticar **4 pacientes** e a Árvore de Decisão simples falhou em **5 pacientes** (deixando quase 10% dos tumores malignos passarem despercebidos).

---

### 2. Validação Cruzada Estratificada no Treino (5-Fold Stratified CV)

| Modelo | Acurácia Média | Desvio-Padrão Acurácia | Recall Maligno Médio | F1-Score Maligno Médio | Diagnóstico de Estabilidade |
|:---|:---:|:---:|:---:|:---:|:---|
| **SVM Linear** | **97,19%** | $\pm 0,0118$ ($1,18\%$) | 94,35% | **0,9615** | **Maior acurácia média e melhor F1** |
| **SVM RBF** | 96,71% | **$\pm 0,0047$ ($0,47\%$)** | 94,37% | 0,9553 | **Menor variância absoluta entre dobras** |
| **Random Forest** | 96,25% | $\pm 0,0172$ ($1,72\%$) | **95,00%** | 0,9497 | Estabilidade intermediária |
| **Decision Tree** | 93,90% | $\pm 0,0216$ ($2,16\%$) | 93,10% | 0,9199 | Maior dispersão estocástica |

---

### 3. Diagnóstico de Hiperparâmetros

#### Efeito de $C$ (SVM Linear)
* $C=0.01$: Margem ampla e permissiva (*soft margin*), 87 vetores de suporte, acurácia de treino de $97,65\%$.
* **$C=1.0$ (Ótimo):** Equilíbrio de generalização, apenas 32 vetores de suporte (9,4% do treino), treino de $98,53\%$ e validação de $100\%$.
* $C=100.0$: Margem hiper-rígida (*hard margin*), 23 vetores de suporte, $100\%$ de acurácia de treino (enrijecimento excessivo).

#### Efeito de $\gamma$ (SVM RBF, $C=1.0$)
* $\gamma=0.001$: Gaussiana excessivamente plana, leve *underfitting* (treino $95,00\%$, 141 vetores de suporte).
* **$\gamma=\text{'scale'}$ ou $0.01$ (Ótimo):** Fronteira suave com raio equilibrado, validação de $100\%$ e 78 a 87 vetores de suporte.
* **$\gamma=1.0$ (Overfitting Extremo):** Raio de curvatura microscópico. A acurácia de treino atinge $100\%$, mas a validação **despenca para $62,79\%$** e o número de vetores de suporte **explode para 340 instâncias** (quase $100\%$ dos dados de treino tornaram-se vetores de suporte memorizados).

#### Profundidade da Árvore de Decisão
* $depth=2$: Modelo parcimonioso com $96,76\%$ de treino e $93,02\%$ de validação.
* $depth=3$: Ponto ótimo de equilíbrio interpretável ($97,94\%$ de treino e $91,86\%$ de validação).
* $depth=10$ e $\text{None}$: **Overfitting nítido**. Treino de $100\%$ e validação estagnada em $93,02\%$, gerando folhas puras com 1 ou 2 amostras.

#### Quantidade de Estimadores na Random Forest
* $n=10$: Acurácia de validação de $95,35\%$.
* $n=50, 100, 200$: Estabilização assintótica em $94,19\%$, demonstrando o limite inferior de redução de variância via *Bagging*.
* **Top 5 Features:** `worst concave points` ($13,8\%$), `worst perimeter` ($13,6\%$), `mean concave points` ($11,1\%$), `worst radius` ($8,8\%$) e `worst area` ($6,7\%$).

---

## Conclusão Técnica e Recomendação de Modelo

> Com base na criteriosa avaliação experimental conduzida sobre o conjunto de dados Breast Cancer Wisconsin, recomenda-se formalmente a seleção e o emprego do modelo **Support Vector Machine com Kernel Linear (SVM Linear com padronização via `StandardScaler` e parâmetro de regularização $C=1.0$)**.
>
> No conjunto de teste final, o SVM Linear liderou com folga todos os indicadores quantitativos, registrando **Acurácia de 98,60%**, **F1-score de 98,11%** e um decisivo **Recall de 98,11%** para a classe maligna (`pos_label=0`), identificando com êxito 52 dos 53 casos reais de câncer e cometendo apenas **um único falso negativo**. O alinhamento perfeito entre as métricas de treino (98,53%), validação cruzada em 5 dobras (**97,19% ± 1,18%**) e teste final (98,60%) comprova a ausência total de sobreajuste (*overfitting*) e uma notável estabilidade de generalização.
>
> Em termos de complexidade estrutural e explicabilidade, o classificador linear sobressai-se amplamente: fundamenta-se em apenas 32 vetores de suporte (meramente 9,4% da base de treino) e opera através da projeção analítica $f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$, propiciando inferência em escala de microssegundos e transparência direta sobre os coeficientes que penalizam a malignidade tecidual (destaque para concavidade e raio médio). Diante da gravidade extrema de um falso negativo na oncologia diagnóstica — cuja omissão atrasa intervenções vitais e compromete a sobrevida do paciente —, o SVM Linear supera a Random Forest (que incorreu em 4 falsos negativos com menor explicabilidade) e a Árvore de Decisão simples (5 falsos negativos com alta variância), consolidando-se como a solução de maior eficácia e segurança clínica.

---

## Estrutura do Repositório

```text
├── Rodolfo_Takemoto.ipynb                       # Notebook principal executado integralmente
├── Atividade_Pratica_SVM_Arvore_RandomForest.ipynb # Cópia idêntica com nomenclatura da atividade
├── requirements.txt                             # Dependências com versões fixadas
├── LICENSE                                      # Licença MIT
├── .gitignore                                   # Filtros de arquivos temporários e checkpoints
└── README.md                                    # Relatório técnico completo e documentação
```

---

## Como Reproduzir Este Projeto

### 1. Execução no Google Colab
Clique no botão abaixo para abrir diretamente o notebook no Google Colab com suporte a execução gratuita em nuvem:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aj1no/ML_Atividade_Pratica_SVM_Arvore_RandomForest/blob/main/Rodolfo_Takemoto.ipynb)

### 2. Execução Local

```bash
# 1. Clone o repositório
git clone https://github.com/aj1no/ML_Atividade_Pratica_SVM_Arvore_RandomForest.git
cd ML_Atividade_Pratica_SVM_Arvore_RandomForest

# 2. Crie e ative um ambiente virtual
python -m venv venv
# No Windows:
venv\Scripts\activate
# No Linux/macOS:
source venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Inicie o Jupyter Notebook
jupyter notebook Rodolfo_Takemoto.ipynb
```

---

## Licença

Este projeto está sob a licença [MIT](LICENSE).
