import numpy as np
import random

# Funções de distância
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

# Funções de normalização
def min_max_normalize(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

def z_score_normalize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Função k-NN
def k_nearest_neighbors(train_data, train_labels, test_instance, k, distance_func):
    distances = []
    for i in range(len(train_data)):
        dist = distance_func(test_instance, train_data[i])
        distances.append((dist, train_labels[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = [label for _, label in distances[:k]]
    return max(set(neighbors), key=neighbors.count)

# Função para avaliar o desempenho
def evaluate_knn(train_data, train_labels, test_data, test_labels, k_values, distance_func):
    accuracy_results = []
    for k in k_values:
        correct_predictions = 0
        for i in range(len(test_data)):
            prediction = k_nearest_neighbors(train_data, train_labels, test_data[i], k, distance_func)
            if prediction == test_labels[i]:
                correct_predictions += 1
        accuracy = correct_predictions / len(test_data)
        accuracy_results.append((k, accuracy))
        print(f"k={k}, Accuracy={accuracy:.4f}")
    return accuracy_results

# Função para dividir o conjunto de dados em treinamento e teste
def split_data(data, labels, train_size_ratio):
    indices = list(range(len(data)))
    random.shuffle(indices)
    train_size = int(len(data) * train_size_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_data = data[train_indices]
    train_labels = labels[train_indices]
    test_data = data[test_indices]
    test_labels = labels[test_indices]
    return train_data, train_labels, test_data, test_labels

# Carregar os dados e normalizar
# Exemplo de carregamento dos dados
data = np.load("dados.npy")  # Exemplo de arquivo numpy
labels = np.load("labels.npy")  # Exemplo de arquivo numpy

# Normalizar os dados
data_min_max = min_max_normalize(data)
data_z_score = z_score_normalize(data)

# Definir valores de k
k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

# Avaliação com diferentes porcentagens de dados de treinamento e normalizações
for train_size_ratio in [0.25, 0.5, 1.0]:
    print(f"\n--- Train Size Ratio: {train_size_ratio*100}% ---")
    train_data, train_labels, test_data, test_labels = split_data(data_min_max, labels, train_size_ratio)
    print("\nEuclidean Distance, Min-Max Normalization")
    evaluate_knn(train_data, train_labels, test_data, test_labels, k_values, euclidean_distance)

    train_data, train_labels, test_data, test_labels = split_data(data_z_score, labels, train_size_ratio)
    print("\nManhattan Distance, Z-score Normalization")
    evaluate_knn(train_data, train_labels, test_data, test_labels, k_values, manhattan_distance)

# Avaliação final com 100% das amostras de treinamento
print("\n--- Final Evaluation with 100% Training Data ---")
train_data, train_labels, test_data, test_labels = split_data(data_min_max, labels, 1.0)
evaluate_knn(train_data, train_labels, test_data, test_labels, k_values, euclidean_distance)
