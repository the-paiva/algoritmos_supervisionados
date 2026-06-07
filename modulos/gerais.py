# Arquivo de funções que são usadas por mais de um algoritmo


# Imports necessários
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report


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
def exibir_relatorio_de_classificacao(y_test, y_pred, iris):
    print("=" * 20, "RELATÓRIO DE CLASSIFICAÇÃO", "=" * 20)
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print("=" * 69)


# Ajuste de valores iniciais necessários para as análises com Iris
def preparar_iris(iris):
    return iris.data, iris.target, iris.feature_names, iris.target_names


# Ajuste de valores iniciais necessários para as análises com Breast Cancer
def preparar_breast_cancer(breast_cancer):
    return breast_cancer.data, breast_cancer.target, breast_cancer.feature_names, breast_cancer.target_names


# Ajuste de valores iniciais necessários para as análises com Wine
def preparar_wine(wine):
    return wine.data, wine.target, wine.feature_names, wine.target_names
