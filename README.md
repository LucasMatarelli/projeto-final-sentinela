# Projeto Final - Testes do Sistema Sentinela (Trilha 1 - ML Clássico)

## 📌 Visão Geral
Este repositório reúne a suíte de testes desenvolvida para avaliar os modelos preditivos industriais do sistema Sentinela, voltado à manutenção preditiva. A suíte inclui testes unitários do pipeline, avaliações estatísticas de robustez e cenários adversariais focados na confiabilidade das predições.

## 🚀 Instruções de Execução
Para reproduzir os testes localmente, siga os passos abaixo:

1. Clone este repositório na sua máquina.
2. Crie e ative um ambiente virtual.
3. Instale as dependências necessárias:
```bash
   pip install pytest pandas numpy scikit-learn statsmodels
```
4. Execute a suíte de testes:
```bash
   pytest -v
```

## 🐛 Falha Detectada e Documentada
Durante a execução, os testes adversariais detectaram um defeito crítico no modelo candidato (V2).

* **Teste que falhou:** `test_contrafactual_campos_irrelevantes`
* **Descrição do defeito:** O modelo não é robusto a dados contrafactuais. Ele associa o `turno` de trabalho à probabilidade mecânica de falha do motor. Ao embaralhar a coluna `turno` (um fator puramente operacional, sem relação com a física da máquina), o modelo V2 alterou indevidamente sua predição de risco de falha.
* **Evidência do erro:** `AssertionError: Defeito Encontrado: O modelo está usando a coluna irrelevante 'turno' para tomar decisões!`

## 📄 Log de Execução
A prova de execução da suíte, com todas as validações aprovadas e a captura exata do defeito acima, está salva no arquivo `log_execucao.txt`, na raiz deste repositório.
