# Projeto 2 - MLOps Nível 1 - Automação do Pipeline de Machine Learning
# Módulo de Extração dos Dados

# Importa a biblioteca pandas para manipulação de dados
import pandas as pd

# Define a função para extrair os dados de um arquivo CSV
def extrai_dados(file_path):

    # Tenta ler o arquivo CSV no caminho especificado
    try:

        # Carrega os dados do arquivo CSV em um DataFrame do pandas
        data = pd.read_csv(file_path)

        # Exibe uma mensagem indicando sucesso na extração dos dados
        print("Dados extraídos com sucesso!")

        # Retorna o DataFrame com os dados extraídos
        return data

    # Trata possíveis exceções durante a leitura do arquivo
    except Exception as e:

        # Exibe uma mensagem de erro detalhando o problema ocorrido
        print(f"Erro ao extrair dados: {e}")
        
        # Retorna None em caso de erro
        return None
