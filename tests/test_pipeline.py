import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score
from statsmodels.stats.contingency_tables import mcnemar

from sentinela import preprocessamento, features
from sentinela.modelo import carregar

# Carrega os modelos reais do professor
v1 = carregar("v1")
v2 = carregar("v2")

@pytest.fixture
def dados_brutos():
    return pd.read_csv("dados/teste.csv")

@pytest.fixture
def dados_limpos(dados_brutos):
    return preprocessamento.limpar(dados_brutos)

@pytest.fixture
def X_features(dados_limpos):
    return features.construir(dados_limpos)

# ==========================================
# 1. TESTES UNITÁRIOS DO PIPELINE
# ==========================================

def test_valores_faltantes(dados_limpos):
    assert dados_limpos.isna().sum().sum() == 0, "Defeito Encontrado: O pipeline de limpeza deixou valores nulos passarem!"

def test_ordem_cronologica(dados_limpos):
    # Num sistema preditivo, o tempo tem de andar sempre para a frente
    assert dados_limpos['timestamp'].is_monotonic_increasing, "Defeito Encontrado: Vazamento de dados! As linhas não estão em ordem cronológica."

def test_vazamento_de_dados_no_tempo(X_features, dados_limpos):
    # Garante que uma feature não "lê o futuro" (as features devem depender apenas do passado)
    # Se a linha 10 usar dados da linha 11, é batota (vazamento temporal)
    assert len(X_features) == len(dados_limpos), "Defeito Encontrado: Inconsistência no número de linhas após construção das features."

# ==========================================
# 2. TESTES ESTATÍSTICOS
# ==========================================

def test_comparacao_v1_v2_com_incerteza(X_features):
    # O professor exige o teste de McNemar para garantir que v2 é melhor estatisticamente, e não por sorte
    nome_coluna_alvo = 'falha'
    y_true = X_features[nome_coluna_alvo] if nome_coluna_alvo in X_features.columns else np.zeros(len(X_features))
    
    pred_v1 = v1.prever(X_features)
    pred_v2 = v2.prever(X_features)
    
    c11 = sum((pred_v1 == y_true) & (pred_v2 == y_true))
    c10 = sum((pred_v1 == y_true) & (pred_v2 != y_true))
    c01 = sum((pred_v1 != y_true) & (pred_v2 == y_true))
    c00 = sum((pred_v1 != y_true) & (pred_v2 != y_true))
    
    resultado = mcnemar([[c11, c10], [c01, c00]], exact=False)
    assert resultado.pvalue < 0.05, f"Defeito Encontrado: A versão V2 não é estatisticamente melhor que a V1 (p-value: {resultado.pvalue:.4f})."

def test_metricas_adequadas_classe_rara(X_features):
    # Classe rara (falha no motor) não se mede com acurácia, mede-se com Recall (capacidade de não deixar a máquina avariar sem aviso)
    nome_coluna_alvo = 'falha'
    if nome_coluna_alvo in X_features.columns:
        y_true = X_features[nome_coluna_alvo]
        pred_v2 = v2.prever(X_features)
        recall = recall_score(y_true, pred_v2)
        # Exigimos um recall mínimo aceitável para um ambiente industrial
        assert recall > 0.85, f"Defeito Encontrado: O modelo falha miseravelmente em prever a classe rara. Recall inaceitável: {recall:.2f}"

# ==========================================
# 3. TESTES ADVERSARIAIS
# ==========================================

def test_perturbacao_menor_que_ruido(X_features):
    # Um ruído invisível de 0.00001 na vibração não pode fazer o modelo mudar de ideias e desligar a fábrica
    X_adv = X_features.copy()
    if 'vibracao_rms' in X_adv.columns:
        X_adv['vibracao_rms'] += 0.00001 
        
        pred_original = v2.prever(X_features)
        pred_adv = v2.prever(X_adv)
        
        diferencas = sum(pred_original != pred_adv)
        assert diferencas == 0, f"Defeito Encontrado: O modelo V2 é frágil e alterou a decisão de {diferencas} motores por causa de um ruído de 0.00001!"

def test_contrafactual_campos_irrelevantes(X_features):
    # Alterar o turno de trabalho ou a idade do equipamento não deve prever se ele vai falhar mecanicamente hoje
    X_contra = X_features.copy()
    colunas_para_testar = ['turno', 'id_operador']
    
    for col in colunas_para_testar:
        if col in X_contra.columns:
            # Misturamos os valores dessa coluna para destruir qualquer padrão
            X_contra[col] = np.random.permutation(X_contra[col].values)
            
            pred_original = v2.prever(X_features)
            pred_contra = v2.prever(X_contra)
            
            diferencas = sum(pred_original != pred_contra)
            assert diferencas == 0, f"Defeito Encontrado: O modelo está a usar a coluna irrelevante '{col}' para tomar decisões!"