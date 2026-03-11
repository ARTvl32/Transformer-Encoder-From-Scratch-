"""
Passo 2.1: Scaled Dot-Product Attention
=========================================
Laboratório P1-02 — Transformer Encoder From Scratch
Disciplina: Tópicos em IA — iCEV 2026.1
Prof. Dimmy Magalhães

Equação fundamental:
    Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) * V

Por que o scaling por sqrt(d_k)?
    Quando os elementos de Q e K têm média 0 e variância 1,
    o produto escalar Q·K tem variância d_k. Valores grandes
    saturam o Softmax (gradiente ≈ 0). Dividir por sqrt(d_k)
    restaura a variância para 1, mantendo gradientes saudáveis.
"""

import numpy as np

# ─────────────────────────────────────────
# Hiperparâmetros
# ─────────────────────────────────────────
D_MODEL = 64
D_K = D_MODEL       # dimensão das projeções Q e K
D_V = D_MODEL       # dimensão das projeções V
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def softmax(x, axis=-1):
    """
    Softmax numericamente estável.

    Subtrai o máximo antes de exponenciar para evitar overflow:
        softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    Matematicamente equivalente ao softmax padrão, mas evita
    exp(500) = inf que causaria NaN.

    Args:
        x: np.ndarray de qualquer shape
        axis: eixo ao longo do qual normalizar

    Returns:
        np.ndarray de mesmo shape, com valores somando 1 no eixo dado
    """
    # Subtrai o máximo por estabilidade numérica
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x_shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def init_projection_weights(d_model, d_k, d_v):
    """
    Inicializa as três matrizes de projeção aprendíveis:
        W_Q ∈ R^(d_model × d_k)
        W_K ∈ R^(d_model × d_k)
        W_V ∈ R^(d_model × d_v)

    Em um modelo real, essas matrizes são aprendidas por backpropagation.
    Aqui inicializamos com distribuição normal escalada.

    Args:
        d_model: dimensão dos embeddings de entrada
        d_k: dimensão das projeções Q e K
        d_v: dimensão das projeções V

    Returns:
        tuple (W_Q, W_K, W_V)
    """
    scale = np.sqrt(2.0 / d_model)     # inicialização de He
    W_Q = np.random.randn(d_model, d_k) * scale
    W_K = np.random.randn(d_model, d_k) * scale
    W_V = np.random.randn(d_model, d_v) * scale

    return W_Q, W_K, W_V


def scaled_dot_product_attention(X, W_Q, W_K, W_V):
    """
    Implementa o Scaled Dot-Product Attention completo.

    Pipeline:
        1. Projeção Linear: gera Q, K, V a partir de X
        2. Score bruto: QK^T
        3. Scaling: divide por sqrt(d_k)
        4. Softmax: converte scores em pesos de atenção (somam 1)
        5. Weighted Sum: multiplica pesos pelos Values

    Args:
        X:   np.ndarray de shape (B, T, d_model) — tensor de entrada
        W_Q: np.ndarray de shape (d_model, d_k)
        W_K: np.ndarray de shape (d_model, d_k)
        W_V: np.ndarray de shape (d_model, d_v)

    Returns:
        output:          np.ndarray (B, T, d_v) — representações contextualizadas
        attention_weights: np.ndarray (B, T, T)  — matriz de atenção (para análise)
    """
    d_k = W_K.shape[-1]

    # ── Passo 1: Projeção Linear ───────────────────────────────────────
    # X @ W_Q: (B, T, d_model) @ (d_model, d_k) = (B, T, d_k)
    Q = X @ W_Q   # "O que cada token está procurando?"
    K = X @ W_K   # "O que cada token oferece?"
    V = X @ W_V   # "Que conteúdo cada token carrega?"

    # ── Passo 2: Score bruto via produto escalar ───────────────────────
    # Q @ K^T: (B, T, d_k) @ (B, d_k, T) = (B, T, T)
    # scores[b, i, j] = similaridade entre token i e token j
    scores_raw = Q @ K.transpose(0, 2, 1)

    # ── Passo 3: Scaling por sqrt(d_k) ────────────────────────────────
    # Sem isso, scores com d_k=64 teriam std≈8, saturando o Softmax
    scores_scaled = scores_raw / np.sqrt(d_k)

    # ── Passo 4: Softmax → pesos de atenção ───────────────────────────
    # Converte scores em distribuição de probabilidade (somam 1 por linha)
    attention_weights = softmax(scores_scaled, axis=-1)

    # ── Passo 5: Weighted Sum dos Values ──────────────────────────────
    # (B, T, T) @ (B, T, d_v) = (B, T, d_v)
    output = attention_weights @ V

    return output, attention_weights


# ─────────────────────────────────────────
# Execução standalone (teste do módulo)
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔷 PASSO 2.1: SCALED DOT-PRODUCT ATTENTION\n")

    # Simular tensor de entrada
    B, T = 1, 5
    X = np.random.randn(B, T, D_MODEL) * 0.1

    # Inicializar pesos
    W_Q, W_K, W_V = init_projection_weights(D_MODEL, D_K, D_V)

    print("=" * 50)
    print("DIMENSÕES DAS PROJEÇÕES")
    print("=" * 50)
    print(f"  W_Q: {W_Q.shape}  (d_model × d_k)")
    print(f"  W_K: {W_K.shape}  (d_model × d_k)")
    print(f"  W_V: {W_V.shape}  (d_model × d_v)")

    # Calcular atenção
    output, weights = scaled_dot_product_attention(X, W_Q, W_K, W_V)

    print("\n" + "=" * 50)
    print("PIPELINE DA ATENÇÃO")
    print("=" * 50)
    Q = X @ W_Q
    K = X @ W_K
    scores_raw = Q @ K.transpose(0, 2, 1)
    scores_scaled = scores_raw / np.sqrt(D_K)

    print(f"  X shape:             {X.shape}")
    print(f"  Q shape:             {Q.shape}   → (B, T, d_k)")
    print(f"  K shape:             {K.shape}   → (B, T, d_k)")
    print(f"  Scores raw shape:    {scores_raw.shape}     → (B, T, T)")
    print(f"  Scores scaled shape: {scores_scaled.shape}     → (B, T, T)")
    print(f"  Weights shape:       {weights.shape}     → (B, T, T)")
    print(f"  Output shape:        {output.shape}   → (B, T, d_v)")

    print("\n" + "=" * 50)
    print("VALIDAÇÃO DO SOFTMAX")
    print("=" * 50)
    soma_por_linha = weights[0].sum(axis=-1)
    print(f"  Soma dos pesos por token (deve ser ≈1.0):")
    for i, s in enumerate(soma_por_linha):
        print(f"    Token {i}: {s:.6f}")

    print("\n" + "=" * 50)
    print("EFEITO DO SCALING")
    print("=" * 50)
    print(f"  Std scores SEM scaling: {scores_raw.std():.4f}")
    print(f"  Std scores COM scaling: {scores_scaled.std():.4f}")
    print(f"  sqrt(d_k) = sqrt({D_K}) = {np.sqrt(D_K):.4f}")

    print("\n✅ Attention output shape:", output.shape)
    assert output.shape == (B, T, D_V), "ERRO: shape incorreto!"
    print("  Shape correto: (B, T, d_v) ✓")
