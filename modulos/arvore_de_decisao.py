# Arquivo de funções referentes ao algoritmo Árvore de Decisão


# Imports necessários
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from modulos.gerais import avaliar_modelo
import matplotlib.pyplot as plt


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


# Exibe as regras geradas pela Árvore de Decisãot 
def exibir_regras_da_arvore_de_decisao(clf, feature_names):
    print("\nREGRAS DA ÁRVORE DE DECISÃO (FORMATO TEXTO):")
    regras_texto = export_text(clf, feature_names=feature_names)
    print(regras_texto)
    print("=" * 50)



# Gera o gráfico de uma árvore de decisão
def gerar_grafico_da_arvore_de_decisao(clf, feature_names, target_names, nome_da_base):
    plt.figure(figsize=(12, 8))

    plot_tree(
        clf, feature_names=feature_names, class_names=target_names,
        filled=True, rounded=True,
    )

    plt.title(f'Árvore de Decisão - Base {nome_da_base}')
    plt.show()
