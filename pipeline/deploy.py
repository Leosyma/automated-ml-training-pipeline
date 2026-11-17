# Projeto 2 - MLOps Nível 1 - Automação do Pipeline de Machine Learning
# Módulo de Deploy do Modelo

# Importa o módulo os para manipulação de caminhos de arquivos
import os

# Importa a biblioteca joblib para carregar o modelo e os objetos necessários para o pré-processamento
import joblib

# Define a função para carregar o modelo, o scaler e as colunas do pré-processamento
def carrega_modelo(model_path, scaler_path, columns_path):

    # Tenta carregar o modelo, o scaler e as colunas salvas
    try:

        # Carrega o modelo do caminho especificado
        model = joblib.load(model_path)

        # Carrega o scaler do caminho especificado
        scaler = joblib.load(scaler_path)

        # Carrega as colunas do caminho especificado
        columns = joblib.load(columns_path)

        # Exibe uma mensagem indicando que os objetos foram carregados com sucesso
        print(f"Modelo, scaler e colunas carregados de {model_path}")

        # Retorna o modelo, o scaler e as colunas carregadas
        return model, scaler, columns

    # Trata possíveis exceções ao carregar os arquivos
    except Exception as e:

        # Exibe uma mensagem de erro detalhando o problema ocorrido
        print(f"Erro ao carregar o modelo ou pré-processamento: {e}")

        # Retorna None para indicar falha no carregamento
        return None, None, None

# Define a função para realizar previsões com o modelo carregado
def faz_previsoes(model, X_new):

    # Faz previsões usando o modelo fornecido
    predictions = model.predict(X_new)
    
    # Retorna as previsões geradas
    return predictions




