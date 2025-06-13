import os
import random
import numpy as np
import librosa
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# ==========
# Processamento de Áudio (MFCC médio)
# ==========
def process_audio_file(file_path, sr=22050, duration=30, offset=3.0, n_mfcc=20):
    y, _ = librosa.load(file_path, sr=sr, offset=offset, duration=duration, mono=True)
    y = y / np.max(np.abs(y))  
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


def load_dataset(diretorio_base, n_train, n_test, seed=42):
    """
    Carrega um dataset com n_train e n_test arquivos por gênero.
    """
    random.seed(seed)
    X_train, y_train, X_test, y_test, test_files = [], [], [], [], []

    for genero in sorted(os.listdir(diretorio_base)):
        pasta_genero = os.path.join(diretorio_base, genero)
        if not os.path.isdir(pasta_genero):
            continue
        arquivos = [f for f in os.listdir(pasta_genero) if f.lower().endswith('.flac') and not f.startswith('.')]

        if len(arquivos) < n_train + n_test:
            raise ValueError(f"Gênero '{genero}' tem apenas {len(arquivos)} arquivos. Requeridos: {n_train + n_test}")

        arquivos_escolhidos = random.sample(arquivos, n_train + n_test)
        arquivos_treino = arquivos_escolhidos[:n_train]
        arquivos_teste = arquivos_escolhidos[n_train:]

        for arquivo in arquivos_treino:
            path = os.path.join(pasta_genero, arquivo)
            features = process_audio_file(path)
            X_train.append(features)
            y_train.append(genero)

        for arquivo in arquivos_teste:
            path = os.path.join(pasta_genero, arquivo)
            features = process_audio_file(path)
            X_test.append(features)
            y_test.append(genero)
            test_files.append(arquivo)

    # Normalização
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        np.array(X_train), np.array(y_train),
        np.array(X_test), np.array(y_test),
        test_files
    )

def pca_classification(X_train, y_train, X_test, y_test, n_components=10):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    centroides = {
        genero: X_train_pca[y_train == genero].mean(axis=0)
        for genero in np.unique(y_train)
    }

    y_pred = []
    for x in X_test_pca:
        dist = {g: np.linalg.norm(x - c) for g, c in centroides.items()}
        y_pred.append(min(dist, key=dist.get))

    return y_pred, X_train_pca, X_test_pca, pca

def classificar_com_testes_definidos_dict(diretorio_base, testes_dict, n_treino=5, n_componentes=10, seed=42):
    """
    Classifica músicas com PCA, com testes fixos e treinos sorteados.
    
    :param diretorio_base: Caminho para o dataset, com subpastas por gênero.
    :param testes_dict: Dicionário no formato {genero: [lista_de_arquivos.flac]}.
    :param n_treino: Quantidade de músicas para treino por gênero (exceto as definidas como teste).
    :param n_componentes: Número de componentes principais do PCA.
    :param seed: Semente para o sorteio.
    :return: acurácia, y_pred, y_teste, arquivos_de_teste
    """
    import random
    random.seed(seed)

    X_treino, y_treino, X_teste, y_teste, test_files = [], [], [], [], []

    for genero in os.listdir(diretorio_base):
        pasta_genero = os.path.join(diretorio_base, genero)
        if not os.path.isdir(pasta_genero):
            continue

        arquivos = [f for f in os.listdir(pasta_genero) if f.lower().endswith('.flac') and not f.startswith('.')]
        arquivos_testes = testes_dict.get(genero, [])

        arquivos_disponiveis = list(set(arquivos) - set(arquivos_testes))
        if len(arquivos_disponiveis) < n_treino:
            raise ValueError(f"Não há arquivos suficientes para treino em {genero}. Requeridos: {n_treino}, disponíveis: {len(arquivos_disponiveis)}")

        arquivos_treino = random.sample(arquivos_disponiveis, n_treino)

        # Processa os arquivos de treino
        for arquivo in arquivos_treino:
            caminho = os.path.join(pasta_genero, arquivo)
            try:
                features = process_audio_file(caminho)
                X_treino.append(features)
                y_treino.append(genero)
            except Exception as e:
                print(f"[Erro treino] {caminho}: {e}")

        # Processa os arquivos de teste
        for arquivo in arquivos_testes:
            caminho = os.path.join(pasta_genero, arquivo)
            try:
                features = process_audio_file(caminho)
                X_teste.append(features)
                y_teste.append(genero)
                test_files.append(arquivo)
            except Exception as e:
                print(f"[Erro teste] {caminho}: {e}")

    # Padronização
    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino)
    X_teste = scaler.transform(X_teste)

    # PCA
    pca = PCA(n_components=n_componentes)
    X_treino_pca = pca.fit_transform(X_treino)
    X_teste_pca = pca.transform(X_teste)

    # Centróides
    centroides = {}
    for genero in set(y_treino):
        Xg = X_treino_pca[np.array(y_treino) == genero]
        centroides[genero] = Xg.mean(axis=0)

    # Classificação
    y_pred = []
    for x in X_teste_pca:
        dist = {g: np.linalg.norm(x - c) for g, c in centroides.items()}
        y_pred.append(min(dist, key=dist.get))

    acc = accuracy_score(y_teste, y_pred)
    return acc, y_pred, y_teste, test_files


# ==========
# Avaliação com matriz de confusão
# ==========
def evaluate(y_test, y_pred, show_plot=True):
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Acurácia: {acc * 100:.2f}%")
    if show_plot:
        labels = sorted(list(set(y_test) | set(y_pred)))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        plt.title("Matriz de Confusão")
        plt.show()

# ==========
# Visualização em 2D do PCA
# ==========
def plot_pca_2d(X_pca, y, title="PCA 2D"):
    pca2d = PCA(n_components=2)
    X_2d = pca2d.fit_transform(X_pca)
    plt.figure(figsize=(8,6))
    for genre in np.unique(y):
        idx = np.array(y) == genre
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=genre, alpha=0.7)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pca_3d(X_pca, y, title="PCA 3D"):
    """
    Plota os dados no espaço tridimensional dos três primeiros componentes principais.
    """
    pca3d = PCA(n_components=3)
    X_3d = pca3d.fit_transform(X_pca)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for genre in np.unique(y):
        idx = np.array(y) == genre
        ax.scatter(X_3d[idx, 0], X_3d[idx, 1], X_3d[idx, 2], label=genre, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    plt.tight_layout()
    plt.show()

# ==========
def gerar_tabela_distancias(X_test_pca, y_test, test_files, y_train, X_train_pca, centroides):
    """
    Gera um DataFrame com distâncias de cada amostra de teste aos centróides.
    """
    dados = []

    for i, x in enumerate(X_test_pca):
        distancias = {g: np.linalg.norm(x - c) for g, c in centroides.items()}
        genero_predito = min(distancias, key=distancias.get)
        dist_min = distancias[genero_predito]

        dados.append({
            'Arquivo': test_files[i],
            'Verdadeiro': y_test[i],
            'Predito': genero_predito,
            **distancias,
            'Mais_proximo': genero_predito,
            'Dist_min': dist_min
        })

    df = pd.DataFrame(dados)
    return df

def gerar_tabela_probabilidades(X_test_pca, y_test, test_files, y_train, X_train_pca, centroides, epsilon=1e-10):
    """
    Gera um DataFrame com probabilidades baseadas nas distâncias aos centróides.
    """
    dados = []
    generos = list(centroides.keys())

    for i, x in enumerate(X_test_pca):
        # Calcula distâncias
        distancias = {g: np.linalg.norm(x - c) for g, c in centroides.items()}
        
        # Inverso das distâncias
        inversos = {g: 1 / (d + epsilon) for g, d in distancias.items()}
        soma_inversos = sum(inversos.values())

        # Probabilidades normalizadas
        probabilidades = {g: inv / soma_inversos for g, inv in inversos.items()}
        genero_predito = max(probabilidades, key=probabilidades.get)

        dados.append({
            'Arquivo': test_files[i],
            'Verdadeiro': y_test[i],
            'Predito': genero_predito,
            **probabilidades,
            'Mais_provavel': genero_predito,
            'Prob_max': probabilidades[genero_predito]
        })

    df = pd.DataFrame(dados)
    return df


# ==========
# Executar
# ==========
if __name__ == "__main__":
    dataset_path = "dataset"  # <- ajuste para o seu caminho real

    # Carregar e processar
    X_train, y_train, X_test, y_test, test_files = load_dataset(dataset_path, n_train=5)

    # Classificação com PCA
    y_pred, X_train_pca, X_test_pca, pca_model = pca_classification(X_train, y_train, X_test, y_test, n_components=10)

    # Avaliação
    evaluate(y_test, y_pred)

    # Visualização PCA 2D
    plot_pca_2d(np.vstack([X_train, X_test]), np.concatenate([y_train, y_test]), title="PCA das músicas")

