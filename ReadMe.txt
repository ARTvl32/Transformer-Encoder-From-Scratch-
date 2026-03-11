Transformer Encoder — From Scratch

Disciplina: Tópicos em Inteligência Artificial – 2026.1  
Professor: Prof. Dimmy Magalhães  
Instituição: iCEV - Instituto de Ensino Superior  

---

 Descrição

Implementação do Forward Pass completo de um bloco **Encoder do Transformer**, conforme proposto no artigo original *"Attention Is All You Need" (Vaswani et al., 2017).

O projeto implementa do zero, usando apenas `numpy` e `pandas`, todos os componentes matemáticos da arquitetura:

- Tabela de Embeddings e preparação do tensor de entrada `(Batch, Tokens, d_model)`
- Scaled Dot-Product Attention com Softmax estável
- Conexões Residuais (Add) + Layer Normalization
- Position-wise Feed-Forward Network (FFN) com ReLU
- Pilha de N=6 camadas idênticas do Encoder


 Estrutura do Projeto

transformer_encoder/
 README.md                   Este arquivo
├── requirements.txt            Dependências

├── step1_data_prep.py          Passo 1: Vocabulário, Embeddings e tensor de entrada
├── step2_attention.py          Passo 2.1: Scaled Dot-Product Attention
├── step3_add_norm.py           Passo 2.2: Conexão Residual + LayerNorm
├── step4_ffn.py                Passo 2.3: Feed-Forward Network
└── step5_encoder.py            Passo 3: Pilha completa de 6 camadas (main)

---

 Como Rodar

 1. Pré-requisitos

```bash
pip install numpy pandas
```

Ou usando o arquivo de dependências:

```bash
pip install -r requirements.txt
```

 2. Executar o pipeline completo

```bash
python step5_encoder.py
```

Isso executa o Forward Pass completo: frase de entrada → 6 camadas do Encoder → Vetor Z contextualizado.

 3. Executar módulos individuais

Cada passo pode ser testado de forma isolada:

```bash
python step1_data_prep.py    # Testa preparação dos dados
python step2_attention.py    # Testa o mecanismo de atenção
python step3_add_norm.py     # Testa Add & Norm
python step4_ffn.py          # Testa a FFN
```

---

 Parâmetros Arquiteturais

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `d_model` | 64    | Dimensão dos embeddings (paper usa 512) |
| `d_ff`    | 256   | Dimensão interna da FFN (paper usa 2048) |
| `N`       | 6     | Número de camadas do Encoder |
| `eps`     | 1e-6  | Epsilon da LayerNorm |

> **Nota:** O paper original define `d_model = 512`. Este laboratório usa `d_model = 64` para viabilizar execução em CPU, conforme instrução do professor.

---

 Validação de Sanidade

O tensor de entrada entra na Camada 1 com dimensão `(1, T, 64)` e **deve sair da Camada 6 com exatamente a mesma dimensão** `(1, T, 64)`, porém com representações contextualizadas (Vetor Z).

---

Nota de Integridade Acadêmica

O Claude (Anthropic) foi consultado para tirar dúvidas de sintaxe do `numpy` durante o desenvolvimento, conforme permitido pelo Contrato Pedagógico. O código foi escrito e compreendido pelo aluno. Nenhuma classe pronta foi submetida diretamente.

---

 Referência

Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.  
