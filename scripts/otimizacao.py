import numpy as np
from skopt import gp_minimize
from skopt.space import Real

# 1. Definimos o espaço de busca dos parâmetros do impacto (exemplo de limites)
# Você deve ajustar esses limites com base no arquivo de configuração do desafio!
espaco_de_busca = [
    Real(10.0, 100.0, name='massa_asteroide'),
    Real(11.0, 50.0, name='velocidade_impacto'),
    Real(0.0, 90.0, name='angulo_impacto')
]

# 2. Criamos a função objetivo que avalia quão bom é um cenário gerado
def avaliar_cenario(parametros, modelo_ia_treinado):
    # Convertendo os parâmetros para o formato que a IA entende
    entrada = torch.tensor([parametros], dtype=torch.float32)
    
    # Usando nossa IA treinada para prever o resultado desse cenário
    modelo_ia_treinado.eval()
    with torch.no_grad():
        predicao = modelo_ia_treinado(entrada).numpy()[0]
        
    p80_previsto = predicao[0]
    r95_previsto = predicao[1]
    
    # Calculando a pontuação baseada nas regras do Desafio
    # Regra 1: P80 precisa estar rigidamente entre 96 e 101
    penalidade_p80 = 0
    if not (96 <= p80_previsto <= 101):
        penalidade_p80 = 5000  # Penalidade gigantesca se falhar
        
    # Regra 2: R95 precisa ser menor ou igual a 175
    penalidade_r95 = 0
    if r95_previsto > 175:
        penalidade_r95 = 5000
        
    # Queremos menor energia (massa * velocidade) e menor alcance dos destroços
    energia = parametros[0] * (parametros[1] ** 2)
    score_small_impact = energia + r95_previsto
    
    # Otimizadores buscam MINIMIZAR o valor. Então somamos o score com as penalidades.
    return score_small_impact + penalidade_p80 + penalidade_r95

# Nota: No seu notebook de treinamento, você chamará a função 'gp_minimize' 
# passando essa função 'avaliar_cenario' para encontrar as 20 melhores combinações.