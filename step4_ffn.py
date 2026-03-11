"""
Passo 2.3: Position-wise Feed-Forward Network (FFN)
=====================================================
Laboratório P1-02 — Transformer Encoder From Scratch
Disciplina: Tópicos em IA — iCEV 2026.1
Prof. Dimmy Magalhães

Equação:
    FFN(x) = max(0, x·W1 + b1) · W2 + b2

Por que o FFN existe se já temos a Atenção?
    - Self-Attention é essencialmente LINEAR após o Softmax
      (produz média ponderada dos Values — operação inter-token)
    - O FFN introduz NÃO-LINEARIDADE via ReLU — operação intra-token
    - Sem o FFN, uma pilha de camadas de atenção seria equivalente
      a UMA ÚNICA transformação linear complexa

Por que expandir 64 → 256 → 64 (ou 512 → 2048 → 512 no paper)?
    - Expansão (×4): "descompacta" a representação em espaço de alta
      dimensão onde padrões indistinguíveis em 64D tornam-se separáveis
    - ReLU: zera seletivamente ~50% das dimensões (codificação esparsa)
    - Compressão: recombina features refinadas de volta a d_model
      para que a conexão residual funcione (exige mesma dimensão)
"""

import numpy as np

# ─────────────────────────────────────────
# Hiperparâmetros
# ─────────────────────────────────────────
D_MODEL = 64
D_FF    = D_MODEL * 4   # 256 (paper usa 2048 = 512 × 4)
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def relu(x):
    """
    Rectified Linear Unit: max(0, x)

    Zera todos os valores negativos, mantendo positivos intactos.
    Implementado com np.maximum para operar elemento a elemento.

    Args:
        x: np.ndarray de qualquer shape

    Returns:
        np.ndarray de mesmo shape com valores negativos zerados
    """
    return np.maximum(0, x)


def init_ffn_weights(d_model, d_ff):
    """
    Inicializa os pesos das duas camadas lineares da FFN.

    W1 ∈ R^(d_model × d_ff)   — expansão
    b1 ∈ R^(d_ff)              — bias da camada 1
    W2 ∈ R^(d_ff × d_model)   — compressão
    b2 ∈ R^(d_model)           — bias da camada 2

    Inicialização de He (sqrt(2/fan_in)) recomendada para redes com ReLU,
    pois metade das unidades são zeradas pela ativação.

    Args:
        d_model: dimensão dos embeddings
        d_ff:    dimensão interna expandida

    Returns:
        tuple (W1, b1, W2, b2)
    """
    W1 = np.random.randn(d_model, d_ff)    * np.sqrt(2.0 / d_model)
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff,    d_model) * np.sqrt(2.0 / d_ff)
    b2 = np.zeros(d_model)

    return W1, b1, W2, b2


def feed_forward_network(x, W1, b1, W2, b2):
    """
    Aplica a Feed-Forward Network position-wise.

    "Position-wise" significa que a MESMA transformação é aplicada
    independentemente a cada token — não há troca de informação
    entre posições aqui (isso é papel da Atenção).

    Pipeline:
        1. Expansão linear:  hidden = x @ W1 + b1  → (B, T, d_ff)
        2. Ativação ReLU:    hidden = max(0, hidden) → sparsidade ~50%
        3. Compressão linear: out   = hidden @ W2 + b2 → (B, T, d_model)

    Args:
        x:  np.ndarray de shape (B, T, d_model)
        W1: np.ndarray de shape (d_model, d_ff)
        b1: np.ndarray de shape (d_ff,)
        W2: np.ndarray de shape (d_ff, d_model)
        b2: np.ndarray de shape (d_model,)

    Returns:
        np.ndarray de shape (B, T, d_model) — mesma shape da entrada
    """
    # ── Passo 1: Transformação linear de expansão ──────────────────────
    # (B, T, d_model) @ (d_model, d_ff) = (B, T, d_ff)
    hidden = x @ W1 + b1

    # ── Passo 2: Ativação ReLU ─────────────────────────────────────────
    # Zera valores negativos → codificação esparsa
    hidden = relu(hidden)

    # ── Passo 3: Transformação linear de compressão ────────────────────
    # (B, T, d_ff) @ (d_ff, d_model) = (B, T, d_model)
    output = hidden @ W2 + b2

    return output


# ─────────────────────────────────────────
# Execução standalone (teste do módulo)
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔷 PASSO 2.3: FEED-FORWARD NETWORK (FFN)\n")

    B, T = 1, 5
    X = np.random.randn(B, T, D_MODEL) * 0.1

    # Inicializar pesos
    W1, b1, W2, b2 = init_ffn_weights(D_MODEL, D_FF)

    print("=" * 55)
    print("DIMENSÕES DOS PESOS DA FFN")
    print("=" * 55)
    print(f"  W1 shape: {W1.shape}   (d_model → d_ff)   EXPANSÃO")
    print(f"  b1 shape: {b1.shape}             (bias camada 1)")
    print(f"  W2 shape: {W2.shape}   (d_ff → d_model)   COMPRESSÃO")
    print(f"  b2 shape: {b2.shape}              (bias camada 2)")
    print(f"\n  Razão de expansão: d_ff/d_model = {D_FF}/{D_MODEL} = {D_FF//D_MODEL}×")

    print("\n" + "=" * 55)
    print("PIPELINE DA FFN")
    print("=" * 55)

    # Mostrar shapes em cada etapa
    hidden_pre_relu  = X @ W1 + b1
    hidden_post_relu = relu(hidden_pre_relu)
    output           = hidden_post_relu @ W2 + b2

    print(f"  Entrada X:           {X.shape}       → (B, T, d_model)")
    print(f"  Após W1 (expansão):  {hidden_pre_relu.shape}    → (B, T, d_ff)")
    print(f"  Após ReLU:           {hidden_post_relu.shape}    → (B, T, d_ff)")
    print(f"  Após W2 (compress.): {output.shape}       → (B, T, d_model)")

    print("\n" + "=" * 55)
    print("ANÁLISE DA ESPARSIDADE DO RELU")
    print("=" * 55)
    sparsidade = (hidden_post_relu == 0).mean() * 100
    print(f"  Neurônios zerados pelo ReLU: {sparsidade:.1f}%")
    print(f"  (esperado: ≈ 50% para distribuição normal)")
    print(f"\n  Por token:")
    for i in range(T):
        sp = (hidden_post_relu[0, i] == 0).mean() * 100
        print(f"    Token {i}: {sp:.1f}% zerado")

    print("\n" + "=" * 55)
    print("PRESERVAÇÃO DIMENSIONAL")
    print("=" * 55)
    output_final = feed_forward_network(X, W1, b1, W2, b2)
    print(f"  Input shape:   {X.shape}")
    print(f"  Output shape:  {output_final.shape}  ← IGUAL à entrada ✓")
    print(f"  (necessário para a conexão residual funcionar)")

    print("\n✅ FFN output shape:", output_final.shape)
    assert output_final.shape == (B, T, D_MODEL), "ERRO: shape incorreto!"
    print("  Shape correto: (B, T, d_model) ✓")
