# Arquivo de funções referentes ao algoritmo Árvore de Decisão


# Imports necessários
from modulos.gerais import avaliar_modelo
from sklearn.naive_bayes import GaussianNB


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

