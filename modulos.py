# ================================== IMPORTS ==================================


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.neighbors import KNeighborsClassifier


# ================================== FUNÇÕES GERAIS ==================================


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


# Verifica se o valor atual de k tem um desempenho maior do que o melhor valor de k
# até então, e o atualiza caso isso aconteça.
def atualizar_melhor_desempenho_k(f1_knn, k, melhor_f1, melhor_k):
    if f1_knn > melhor_f1:
        return f1_knn, k
    
    return melhor_f1, melhor_k


# Exibe o texto informando o melhor valor de k usado pelo algoritmo KNN
def exibir_melhor_resultado_do_knn(melhor_k, melhor_f1):
    print("=" * 50)
    print(f"CONCLUSÃO: O melhor desempenho foi com K = {melhor_k} (F1-Score: {melhor_f1:.4f})")
    print("=" * 50)


# Realiza o treinamento e avaliação de um modelo que usa KNN de acordo com o valor escolhido de K
def treinar_k(k, resultados_acuracia, X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acuracia_knn, precisao_knn, revocacao_knn, f1_knn = avaliar_modelo(y_test, y_pred_knn)

    # Guardamos a acurácia para plotar o gráfico depois
    resultados_acuracia.append(acuracia_knn)

    return acuracia_knn, precisao_knn, revocacao_knn, f1_knn


# ================================== FUNÇÕES DO IRIS ==================================


# Ajuste de valores iniciais necessários para as análises com Iris
def preparar_iris(iris):
    return iris.data, iris.target, iris.feature_names, iris.target_names
