# Arquivo de funções referentes ao algoritmo Árvore de Decisão


# Imports necessários
from modulos.gerais import avaliar_modelo, exibir_resultado, exibir_relatorio_de_classificacao
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# Realiza todo o processo de preparação e avaliação de um modelo com Árvore de Decisão
def preparar_knn():
    valores_k = [1, 3, 5, 7, 9, 11]
    resultados_acuracia = []
    melhor_k = 1
    melhor_f1 = 0

    return valores_k, resultados_acuracia, melhor_k, melhor_f1


# Realiza o treinamento e avaliação de um modelo que usa KNN de acordo com o valor escolhido de K
def treinar_k(k, resultados_acuracia, X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acuracia_knn, precisao_knn, revocacao_knn, f1_knn = avaliar_modelo(y_test, y_pred_knn)

    # Guardamos a acurácia para plotar o gráfico depois
    resultados_acuracia.append(acuracia_knn)

    return y_pred_knn, acuracia_knn, precisao_knn, revocacao_knn, f1_knn


# Verifica se o valor atual de k tem um desempenho maior do que o melhor valor de k
# até então, e o atualiza caso isso aconteça.
def atualizar_melhor_desempenho_k(f1_knn, k, melhor_f1, melhor_k):
    if f1_knn > melhor_f1:
        return f1_knn, k
    
    return melhor_f1, melhor_k


# Função que agrega os resultados exibidos no loop principal do KNN
def exibir_resultados_do_knn(acuracia_knn, precisao_knn, revocacao_knn, f1_knn, k, y_test, y_pred_knn, target_names):
    exibir_resultado( acuracia_knn, precisao_knn, revocacao_knn, f1_knn, f"KNN (K = {k})")
    exibir_relatorio_de_classificacao(y_test, y_pred_knn, target_names, f'BASE IRIS (KNN COM K = {k})')
    print('\n')


# Exibe o texto informando o melhor valor de k usado pelo algoritmo KNN
def exibir_melhor_resultado_do_knn(melhor_k, melhor_f1):
    print("=" * 100)
    print(f"CONCLUSÃO: O melhor desempenho foi com K = {melhor_k} (F1-Score: {melhor_f1:.4f})")
    print("=" * 100)


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

