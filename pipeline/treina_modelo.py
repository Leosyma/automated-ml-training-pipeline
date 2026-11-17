# Projeto 2 - MLOps Nível 1 - Automação do Pipeline de Machine Learning
# Módulo de Treinamento do Modelo

# Importa o RandomForestClassifier para criação e treinamento do modelo
from sklearn.ensemble import RandomForestClassifier

# Importa a biblioteca joblib para salvar o modelo treinado
import joblib

# Importa o módulo os para manipulação de caminhos de arquivos
import os

# Define a função para treinar o modelo usando os dados de treino
def treina_modelo(X_train, y_train):

    # Exibe uma mensagem indicando o início do treinamento do modelo
    print("Iniciando treinamento do modelo...")

    # Cria uma instância do RandomForestClassifier
    model = RandomForestClassifier()

    # Treina o modelo com os dados de treino fornecidos
    model.fit(X_train, y_train)
    
    # Define o caminho onde o modelo treinado será salvo
    model_path = os.path.join("modelos", "random_forest_model.pkl")

    # Salva o modelo treinado no caminho especificado
    joblib.dump(model, model_path)
    
    # Exibe uma mensagem indicando que o modelo foi salvo com sucesso
    print(f"Modelo salvo em {model_path}")
    
    # Retorna o modelo treinado
    return model
