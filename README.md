# Projeto Final - Testes do Sistema Sentinela (Trilha 1 - ML Clássico)

## 📌 Visão Geral
Este repositório contém a suíte de testes desenvolvida para avaliar os modelos preditivos industriais do sistema Sentinela (focado em manutenção preditiva). A suíte contempla testes unitários do pipeline, avaliações estatísticas rigorosas de robustez e cenários adversariais focados na confiabilidade de predições.

## 🚀 Instruções de Execução
Para reproduzir os testes localmente, siga os passos abaixo:

1. Clone este repositório para a sua máquina local.
2. Crie e ative um ambiente virtual.
3. Instale as dependências necessárias:
   - `pip install pytest pandas numpy scikit-learn statsmodels`
4. Execute a suíte de testes:
   - `pytest -v`

## 🐛 Falha Detectada e Documentada
Durante a execução, os testes adversariais detetaram um defeito crítico no modelo candidato (V2).

* **Teste que falhou:** `test_contrafactual_campos_irrelevantes`
* **Descrição do Defeito:** O modelo não é robusto a dados contrafactuais. O algoritmo está a associar o "turno" de trabalho à probabilidade mecânica de falha do motor. Ao embaralhar a coluna `turno` (um fator estritamente logístico e irrelevante para a física da máquina), o modelo V2 alterou indevidamente a sua predição de risco de falha.
* **Evidência do Erro:** `AssertionError: Defeito Encontrado: O modelo está a usar a coluna irrelevante 'turno' para tomar decisões!`

## 📄 Log de Execução
A prova de execução da suíte de testes, contendo todas as validações que passaram e a captura exata do defeito acima, está guardada e disponível no ficheiro `log_execucao.txt` na raiz deste repositório.
