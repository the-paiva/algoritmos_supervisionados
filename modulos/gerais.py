# Arquivo de funções que são usadas por mais de um algoritmo


# Imports necessários
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import ssl


# Avalia um modelo de acordo com as métricas exigidas
def avaliar_modelo(y_test, y_pred):
    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred, average="macro")
    revocacao = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    return acuracia, precisao, revocacao, f1


# Exibe os resultados encontrados nas métricas de avaliação
def exibir_resultado(acuracia, precisao, revocacao, f1, nome_do_algoritmo):
    print("=" * 50)
    print(f"MÉTRICAS DE AVALIAÇÃO - {nome_do_algoritmo}")
    print("=" * 50)
    print(f"Acurácia:  {acuracia:.4f}")
    print(f"Precisão:  {precisao:.4f}")
    print(f"Revocação: {revocacao:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("=" * 50)
    print("\n")


# Exibe um relatório sobre cada classe encontrada pelo algoritmo da árvore de decisão
def exibir_relatorio_de_classificacao(y_test, y_pred, target_names, descricao):
    print("=" * 20, f"RELATÓRIO DE CLASSIFICAÇÃO - {descricao}", "=" * 20)
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("=" * 80, '\n')


# Padroniza um conjunto de dados de treino para uma escala única
def padronizar_dados_de_treino(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


# Ajusta os valores iniciais necessários para as análises com Iris
def preparar_iris(iris):
    return iris.data, iris.target, iris.feature_names, iris.target_names


# Ajusta os valores iniciais necessários para as análises com Breast Cancer
def preparar_breast_cancer(breast_cancer):
    return breast_cancer.data, breast_cancer.target, breast_cancer.feature_names, breast_cancer.target_names


# Ajusta os valores iniciais necessários para as análises com Wine
def preparar_wine(wine):
    return wine.data, wine.target, wine.feature_names, wine.target_names


# Ajusta os valores iniciais necessários para as análises com Digits
def preparar_digits(digits):
    return digits.data, digits.target, digits.feature_names, digits.target_names


# Carrega a base Heart Disease diretamente do repositório original e realiza os ajustes
# necessários para a implementação dos algoritmos de análise
def preparar_heart_disease():
    # Coleta direta da base oficial da UCI Machine Learning Repository
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    colunas = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
    
    df = pd.read_csv(url, names=colunas)

    # Limpeza de Dados (Lidando com os '?' da base original)
    df = df.replace('?', np.nan)
    df = df.dropna()

    # Binarização dos resultados: A base de dados original possui resultados que vão de 0 (sem doença)
    # a 4 (doença grave). Aqui iremos trabalhar com os resultados 0 (saudável) e 1(com doença)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    # Separação de Features e Target
    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = X.columns.tolist()
    target_names = ['saudavel', 'com_doenca']

    return X, y, feature_names, target_names


# Carrega a base 20newsgroups, realizando também seu treinamento e ajustando os valores necessários
# para análises posteriores
def preparar_newsgroup():
    # Escolha das categorias dos textos que serão utilizados
    categorias = ['misc.forsale', 'talk.politics.misc', 'comp.graphics', 'sci.med']

    # Treinamento e teste com semente fixa de reprodutibilidade
    treino = fetch_20newsgroups(subset='train', categories=categorias, random_state=13)
    teste = fetch_20newsgroups(subset='test', categories=categorias, random_state=13)
    y_train = treino.target
    y_test = teste.target
    target_names = treino.target_names

    # Limita a tabela às 3000 palavras mais importantes e elimina palavras de ruído como 'the',
    # 'and', 'that', etc.
    vetorizador = TfidfVectorizer(max_features=3000, stop_words='english')
    
    # Aprendizado e transformação de vocabulário 
    X_train_tfidf = vetorizador.fit_transform(treino.data)
    X_test_tfidf = vetorizador.transform(teste.data)
    feature_names = vetorizador.get_feature_names_out()

    return X_train_tfidf.toarray(), X_test_tfidf.toarray(), y_train, y_test, feature_names, target_names
