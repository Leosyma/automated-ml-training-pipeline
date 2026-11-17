# Projeto 2 - MLOps Nível 1 - Automação do Pipeline de Machine Learning
# Módulo de Limpeza dos Dados

# Define a função para limpar os dados de um DataFrame
def limpa_dados(df):

    # Exibe uma mensagem indicando o início do processo de limpeza de dados
    print("Iniciando limpeza de dados...")

    # Remove duplicatas do DataFrame
    df = df.drop_duplicates()

    # Remove valores nulos do DataFrame
    df = df.dropna()

    # Exibe uma mensagem indicando a conclusão do processo de limpeza
    print("Limpeza de dados concluída.")
    
    # Retorna o DataFrame limpo
    return df
