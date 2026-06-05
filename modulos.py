# ================================== IMPORTS ==================================


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB


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


# Gera o gráfico de uma árvore de decisão
def gerar_grafico_da_arvore_de_decisao(clf, feature_names, target_names, nome_da_base):
    plt.figure(figsize=(12, 8))

    plot_tree(
        clf, feature_names=feature_names, class_names=target_names,
        filled=True, rounded=True,
    )

    plt.title(f'Árvore de Decisão - Base {nome_da_base}')
    plt.show()


# Gera o gráfico comparativo dos valores de K
def exibir_grafico_comparativo(valores_k, resultados_acuracia, nome_da_base):
    plt.figure(figsize=(10, 6))
    plt.plot(valores_k, resultados_acuracia, marker='o', linestyle='-', color='purple', linewidth=2, markersize=8)
    plt.title(f'Impacto do Valor de K na Acurácia do KNN (Base {nome_da_base})', fontsize=14)
    plt.xlabel('Número de Vizinhos (K)', fontsize=12)
    plt.ylabel('Acurácia', fontsize=12)
    plt.xticks(valores_k)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


# Realiza todo o processo de preparação e avaliação de um modelo com Árvore de Decisão
def executar_arvore_de_decisao(X_train, y_train, X_test, y_test):
    # Limitando a profundidade para evitar overfit e gerar regras legíveis
    clf = DecisionTreeClassifier(max_depth=3, random_state=13)
    clf.fit(X_train, y_train)

    # Predição
    y_pred_arvore_de_decisao = clf.predict(X_test)

    # Avaliação das métricas usando Árvore de Decisão com Iris
    (acuracia_arvore_decisao, precisao_arvore_decisao,
    revocacao_arvore_decisao, f1_arvore_decisao) = avaliar_modelo(y_test, y_pred_arvore_de_decisao)

    return clf, y_pred_arvore_de_decisao, acuracia_arvore_decisao, precisao_arvore_decisao, revocacao_arvore_decisao, f1_arvore_decisao


# Realiza todo o processo de preparação e avaliação de um modelo com Naive Bayes
def executar_naive_bayes(X_train, y_train, X_test, y_test):
    # Aplicação do algoritmo Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # Predição
    y_pred_gnb = gnb.predict(X_test)

    # Avaliação das métricas usando Naive Bayes com Breast Cancer
    (acuracia_gaussian, precisao_gaussian,
    revocacao_gaussian, f1_gaussian) = avaliar_modelo(y_test, y_pred_gnb)

    return y_pred_gnb, acuracia_gaussian, precisao_gaussian, revocacao_gaussian, f1_gaussian


# Realiza todo o processo de preparação e avaliação de um modelo com Árvore de Decisão
def preparar_knn():
    valores_k = [1, 3, 5, 7, 9, 11]
    resultados_acuracia = []
    melhor_k = 1
    melhor_f1 = 0

    return valores_k, resultados_acuracia, melhor_k, melhor_f1



# ================================== FUNÇÕES DO IRIS ==================================


# Ajuste de valores iniciais necessários para as análises com Iris
def preparar_iris(iris):
    return iris.data, iris.target, iris.feature_names, iris.target_names


# ================================== FUNÇÕES DO BREAST CANCER ==================================


# Ajuste de valores iniciais necessários para as análises com Breast Cancer
def preparar_breast_cancer(breast_cancer):
    return breast_cancer.data, breast_cancer.target, breast_cancer.feature_names, breast_cancer.target_names
