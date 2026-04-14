# Previsão de Churn de Clientes com Machine Learning

Este projeto constrói um pipeline de machine learning para prever o churn de clientes em uma empresa de telecomunicações. O objetivo é identificar clientes com maior probabilidade de cancelar o serviço, permitindo que o negócio tome ações preventivas antes que o cancelamento ocorra.

---

## Problema

O churn de clientes impacta diretamente a receita de empresas com modelo de assinatura, como telecomunicações, SaaS e streaming. Ao treinar um modelo de classificação com dados históricos de clientes, é possível sinalizar os perfis de maior risco e priorizar ações de retenção de forma proativa.

---

## Base de Dados

O projeto utiliza o dataset IBM Telco Customer Churn, uma base pública com 7.043 registros de clientes e 21 variáveis, incluindo tipo de contrato, cobranças mensais, tempo de relacionamento e uso de serviços. A variável alvo é `Churn` (Sim / Não).

Fonte: [IBM Telco Customer Churn no GitHub](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

---

## Fluxo do Projeto

1. Carregamento e inspeção inicial dos dados
2. Análise Exploratória de Dados (EDA)
3. Limpeza e correção de tipos
4. Engenharia de atributos
5. Codificação de variáveis categóricas
6. Divisão treino e teste (80/20)
7. Padronização com StandardScaler
8. Modelo baseline (DummyClassifier)
9. Treinamento e avaliação com Regressão Logística
10. Treinamento e avaliação com Random Forest
11. Comparação entre modelos
12. Curva ROC
13. Análise de importância das variáveis
14. Exportação do modelo para deploy

---

## Modelos Treinados

| Modelo | Descrição |
|---|---|
| DummyClassifier | Prevê sempre a classe majoritária; usado como baseline de desempenho |
| Regressão Logística | Classificador linear, interpretável e de baixo custo computacional |
| Random Forest | Ensemble de 300 árvores de decisão |

---

## Métricas de Avaliação

Cada modelo é avaliado com acurácia, relatório de classificação (precisão, recall e F1-score), matriz de confusão e curva ROC-AUC. 

---

## Engenharia de Atributos

Dois novos atributos foram criados a partir dos dados existentes:

- `AvgCharges`: valor médio pago por mês, calculado a partir do total cobrado e do tempo de relacionamento
- `NewCustomer`: flag binária que identifica clientes com menos de 12 meses de contrato

---

## Exportação do Modelo

O modelo de Regressão Logística treinado, o StandardScaler e os LabelEncoders são exportados como arquivos `.pkl` via `joblib`. Esses arquivos são destinados ao uso em uma aplicação Streamlit.

```
churn_model.pkl
scaler.pkl
encoders.pkl
```

---
## Interface Streamlit

Caso queira partir para testar diretamente o este classificador, foi desenvolvido uma interface web com Streamlit que, receberá os atributos através de um formulário, fará o preprocessamento com os modelos já treinados anteriormente e exibirá a previsão de Churn com sua probabilidade.

![screenshot](src\assets\screenshot.png)

---

## Tecnologias Utilizadas

- Python 3
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- streamlit

---

## Como Executar

1. Clone este repositório
2. Crie um ambiente virtual
3. Instale as dependências: `uv sync`
4. Abra os notebooks em sua IDE de preferência e execute todas as células em ordem
5. Caso queira ir diretamente para aplicação web, após `uv sync` execute no terminal:
    ```bash
    uv run streamlit run main.py
    ```
    A aplicação estará disponível em: http://localhost:8501

---

## Autor

Desenvolvido por, Wellington M. Santos - Cientista de Dados, como projeto de portfólio com foco em machine learning aplicado a um problema real de negócio.

## Contato
* Linkedin: [in/wellington-moreira-santos](https://www.linkedin.com/in/wellington-moreira-santos/)
* Email: [wsantos08@hotmail.com](mailto:wsantos08@hotmail.com)