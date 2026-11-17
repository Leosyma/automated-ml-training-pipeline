# Projeto 2 - MLOps Nível 1 - Automação do Pipeline de Machine Learning
# Módulo de Processamento dos Dados

# Importa a biblioteca pandas para manipulação de dados
import pandas as pd

# Importa StandardScaler para normalização das features
from sklearn.preprocessing import StandardScaler

# Define a função para pré-processar os dados de um DataFrame
def preprocessa_dados(df, target_column, is_train=True, scaler=None, columns=None):

    # Exibe uma mensagem indicando o início do pré-processamento
    print("Iniciando pré-processamento...")
    
    # Verifica e separa a coluna alvo (target) antes de modificar o DataFrame
    if target_column in df.columns:

        # Armazena a coluna alvo em uma variável separada
        y = df[target_column]

        # Remove a coluna alvo do DataFrame para deixar somente as variáveis preditoras
        df = df.drop(columns=[target_column])

    else:

        # Define o target como None se a coluna alvo não existir
        y = None

    # Codificação
    if is_train:

        # Aplica One-Hot Encoding em variáveis categóricas para os dados de treino
        df = pd.get_dummies(df, drop_first=True)

        # Salva as colunas resultantes para referência futura
        columns = df.columns

    else:

        # Aplica One-Hot Encoding para os novos dados
        df = pd.get_dummies(df)
        
        # Adiciona colunas faltantes que estavam presentes no pré-processamento do treino
        missing_cols = set(columns) - set(df.columns)

        # Adiciona as colunas ausentes com valores preenchidos como 0
        for col in missing_cols:
            df[col] = 0
        
        # Garante que a ordem das colunas seja a mesma do treino
        df = df[columns]

    # Normaliza as variáveis numéricas (Padronização)
    if is_train:

        # Cria um objeto StandardScaler e ajusta aos dados de treino
        scaler = StandardScaler()

        # Aplica a normalização aos dados
        X_scaled = scaler.fit_transform(df)

    else:

        # Aplica a normalização aos novos dados usando o scaler treinado
        X_scaled = scaler.transform(df)

    # Exibe uma mensagem indicando a conclusão do pré-processamento
    print("Pré-processamento concluído.")
    
    # Retorna os dados normalizados, a coluna alvo, o scaler e as colunas processadas
    return X_scaled, y, scaler, columns



