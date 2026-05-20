import torch
import torch.nn as nn

# Criando a estrutura da nossa Rede Neural
class RedeNeuralImpacto(nn.Module):
    def __init__(self, quantidade_entradas, quantidade_saidas):
        super(RedeNeuralImpacto, self).__init__()
        
        # Sequência de camadas "neurais" que vão aprender os padrões complexos
        self.camadas = nn.Sequential(
            nn.Linear(quantidade_entradas, 128), # Primeira camada recebe os parâmetros do asteroide
            nn.GELU(),                          # Função de ativação suave (ótima para física)
            
            nn.Linear(128, 256),                # Camada intermediária profunda
            nn.GELU(),
            
            nn.Linear(256, 128),                # Camada de afunilamento
            nn.GELU(),
            
            nn.Linear(128, quantidade_saidas)   # Saída final: vai prever as métricas P80 e R95
        )
        
    def forward(self, x):
        # Define como os dados "fluem" de ponta a ponta pela rede
        return self.camadas(x)

# Função de Perda Customizada (Aqui adicionamos a física do problema!)
def funcao_perda_com_fisica(predicao, real, dados_entrada):
    # Perda padrão: o quão longe a IA errou o valor real (Mean Squared Error)
    perda_proximidade = nn.functional.mse_loss(predicao, real)
    
    # Restrição física matemática simplificada:
    # Sabendo que energia do impacto está na coluna 0 da entrada e o alcance do destroço (R95) na coluna 1 da saída...
    energia_impacto = dados_entrada[:, 0]
    r95_previsto = predicao[:, 1]
    
    # Penalizar a IA se ela inventar que um asteroide sem energia causou um alcance gigante (violação física)
    violacao_fisica = torch.mean(torch.relu(r95_previsto - (energia_impacto * 2.0 + 30)))
    
    # O erro final une a precisão matemática com o respeito à física
    return perda_proximidade + (0.1 * violacao_fisica)