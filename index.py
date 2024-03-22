import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Função para carregar e pré-processar os dados do arquivo CSV
def carregar_e_preprocessar(caminho_arquivo):
    # Carrega os dados do arquivo CSV
    df = pd.read_csv(caminho_arquivo)
    # Remove as linhas com quaisquer valores ausentes para garantir a integridade dos dados
    df.dropna(inplace=True)
    return df

# Função para treinar e avaliar modelos de classificação
def treinar_e_avaliar(X_treino, X_teste, y_treino, y_teste):
    # Define um dicionário para armazenar os modelos de classificação
    modelos = {
        "Arvore_Decisao": DecisionTreeClassifier(random_state=42),
        "Naive_Bayes": GaussianNB(),
        "SVM": SVC(random_state=42)
    }
    
    # Itera sobre cada modelo, treina e avalia
    for nome, modelo in modelos.items():
        # Treina o modelo com os dados de treinamento
        modelo.fit(X_treino, y_treino)
        # Realiza previsões com os dados de teste
        previsoes = modelo.predict(X_teste)
        # Calcula a acurácia comparando as previsões com os rótulos verdadeiros
        acuracia = accuracy_score(y_teste, previsoes)
        # Imprime a acurácia do modelo
        print(f"|{nome}| - Acurácia: {acuracia:.4f}")
        
        # Cria a matriz de confusão com os rótulos verdadeiros e as previsões e salva png
        matriz = confusion_matrix(y_teste, previsoes)
        sns.heatmap(matriz, annot=True, fmt='d')
        plt.title(f"Matriz de Confusão do Modelo {nome}")
        plt.ylabel('Verdadeiros')
        plt.xlabel('Previsões')
        
        nome_arquivo = f"matriz_confusao_{nome}.png"
        plt.savefig(nome_arquivo)
        plt.close()

# Define os caminhos dos arquivos de dados para serem processados
caminhos_dados = ['column_2C_weka.csv', 'column_3C_weka.csv']

# Itera sobre cada arquivo de dados
for caminho in caminhos_dados:
    # Carrega e pré-processa os dados do arquivo
    dados = carregar_e_preprocessar(caminho)
    # Separa os dados em características (X) e rótulos (y)
    X = dados.iloc[:, :-1]
    y = dados.iloc[:, -1]
    # Divide os dados em conjuntos de treinamento e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Imprime os resultados para o arquivo de dados atual
    print(f"Resultados arquivo {caminho}:")
    # Chama a função para treinar e avaliar os modelos
    treinar_e_avaliar(X_treino, X_teste, y_treino, y_teste)
    print("\n")
