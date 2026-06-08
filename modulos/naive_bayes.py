# Arquivo de funções referentes ao algoritmo Árvore de Decisão


# Imports necessários
from modulos.gerais import avaliar_modelo
from sklearn.naive_bayes import GaussianNB, MultinomialNB


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


# Realiza todo o processo de preparação e avaliação de um modelo com Naive Bayes Multinomial
def executar_naive_bayes_multinomial(X_train, y_train, X_test, y_test):
    # Aplicação do algoritmo Naive Bayes Multinomial
    multinomial = MultinomialNB()
    multinomial.fit(X_train, y_train)

    # Predição
    y_pred_multinomial = multinomial.predict(X_test)

    # Avaliação das métricas usando Naive Bayes com Breast Cancer
    (acuracia_multinomial, precisao_multinomial,
    revocacao_multinomial, f1_multinomial) = avaliar_modelo(y_test, y_pred_multinomial)

    return y_pred_multinomial, acuracia_multinomial, precisao_multinomial, revocacao_multinomial, f1_multinomial
