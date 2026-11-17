# __init__.py
import logging

# Configuração global de logs
logging.basicConfig(level=logging.INFO)

# Importa os módulos
from .dsa_extrai_dados import extrai_dados
from .dsa_limpa_dados import limpa_dados
from .dsa_processa_dados import preprocessa_dados
from .dsa_treina_modelo import treina_modelo
from .dsa_avalia_modelo import avalia_modelo
from .dsa_deploy import carrega_modelo, faz_previsoes

# Estabelece a hierarquia
__all__ = [
    "extrai_dados",
    "limpa_dados",
    "preprocessa_dados",
    "treina_modelo",
    "avalia_modelo",
    "carrega_modelo",
    "faz_previsoes",
]
