"""
Passo 2.2: Conexões Residuais (Add) + Layer Normalization (Norm)
=================================================================
Laboratório P1-02 — Transformer Encoder From Scratch
Disciplina: Tópicos em IA — iCEV 2026.1
Prof. Dimmy Magalhães

Equação fundamental do bloco:
    Output = LayerNorm(x + Sublayer(x))

Por que o Add (Conexão Residual)?
    ∂Output/∂x = 1 + ∂Sublayer/∂x
    O termo "1" garante que o gradiente nunca zere,
    mesmo com 6 camadas × 2 sub-camadas = 12 transformações em série.

Por que LayerNorm (e não BatchNorm)?
    LayerNorm normaliza ao longo dos d_model features de CADA token
    individualmente (axis=-1). BatchNorm normaliza entre amostras do
    batch (axis=0), o que é problemático para frases de tamanho variável.
"""

import numpy as np

# ─────────────────────────────────────────
# Hiperparâmetros
# ─────────────────────────────────────────
D_MODEL = 64
EPSILON = 1e-6      # evita divisão por zero na normalização
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def add_residual(x, sublayer_output):
    """
    Conexão Residual (Add): soma a entrada original com a saída da sub-camada.

    Operação: x_residual = x + Sublayer(x)

    Isso cria dois caminhos de gradiente durante o backpropagation:
        - Caminho direto (skip): gradiente = 1 → nunca se anula
        - Caminho da sub-camada: gradiente = ∂Sublayer/∂x (pode ser pequeno)

    A soma garante que mesmo que ∂Sublayer/∂x ≈ 0 (Vanishing Gradient),
    o termo 1 do caminho direto mantém o sinal de erro fluindo.

    Args:
        x:                np.ndarray — entrada original
        sublayer_output:  np.ndarray — saída da sub-camada (mesma shape que x)

    Returns:
        np.ndarray — soma residual, mesma shape da entrada
    """
    assert x.shape == sublayer_output.shape, (
        f"ERRO Add: shapes incompatíveis {x.shape} vs {sublayer_output.shape}. "
        f"Ambos devem ser (B, T, d_model) para a soma ser possível."
    )
    return x + sublayer_output


def layer_norm(x, eps=EPSILON):
    """
    Layer Normalization: normaliza cada token individualmente ao longo de d_model.

    Para cada token (cada linha do tensor), calcula:
        μ  = média dos d_model elementos
        σ² = variância dos d_model elementos
        x_norm = (x - μ) / sqrt(σ² + ε)

    Resultado: cada token tem média ≈ 0 e variância ≈ 1,
    independente de quantas camadas já foram processadas.

    Nota: a versão completa inclui parâmetros aprendíveis γ (escala) e β
    (deslocamento), inicializados em 1 e 0 respectivamente. Aqui implementamos
    a versão simplificada sem γ e β para clareza pedagógica.

    Args:
        x:   np.ndarray de shape (B, T, d_model)
        eps: float — constante de estabilidade numérica (padrão: 1e-6)

    Returns:
        np.ndarray normalizado, mesma shape da entrada
    """
    # Média e variância ao longo do último eixo (features de cada token)
    mean = np.mean(x, axis=-1, keepdims=True)   # (B, T, 1)
    var  = np.var(x,  axis=-1, keepdims=True)   # (B, T, 1)

    # Normalização: (x - μ) / sqrt(σ² + ε)
    x_norm = (x - mean) / np.sqrt(var + eps)

    return x_norm


def add_and_norm(x, sublayer_output, eps=EPSILON):
    """
    Combina Add + Norm em uma única operação, conforme o paper:
        Output = LayerNorm(x + Sublayer(x))

    Args:
        x:                np.ndarray (B, T, d_model) — entrada original
        sublayer_output:  np.ndarray (B, T, d_model) — saída da sub-camada
        eps: float — epsilon da LayerNorm

    Returns:
        np.ndarray (B, T, d_model) — tensor normalizado
    """
    x_residual = add_residual(x, sublayer_output)
    return layer_norm(x_residual, eps)


# ─────────────────────────────────────────
# Execução standalone (teste do módulo)
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔷 PASSO 2.2: CONEXÃO RESIDUAL + LAYER NORMALIZATION\n")

    B, T = 1, 5

    # Simular entrada e saída de uma sub-camada
    X = np.random.randn(B, T, D_MODEL) * 0.1
    sublayer_out = np.random.randn(B, T, D_MODEL) * 0.1

    print("=" * 55)
    print("DEMONSTRAÇÃO DO ADD (CONEXÃO RESIDUAL)")
    print("=" * 55)
    x_res = add_residual(X, sublayer_out)
    print(f"  X shape:             {X.shape}")
    print(f"  Sublayer out shape:  {sublayer_out.shape}")
    print(f"  X + Sublayer shape:  {x_res.shape}")
    print(f"\n  Norma de X:               {np.linalg.norm(X):.4f}")
    print(f"  Norma de sublayer_out:    {np.linalg.norm(sublayer_out):.4f}")
    print(f"  Norma da soma residual:   {np.linalg.norm(x_res):.4f}")

    print("\n" + "=" * 55)
    print("DEMONSTRAÇÃO DA LAYER NORM")
    print("=" * 55)

    # Antes da normalização
    print(f"\n  ANTES (token 0):")
    print(f"    Média:     {X[0, 0].mean():.4f}")
    print(f"    Variância: {X[0, 0].var():.4f}")
    print(f"    Std:       {X[0, 0].std():.4f}")

    # Após a normalização
    X_norm = layer_norm(X)
    print(f"\n  DEPOIS da LayerNorm (token 0):")
    print(f"    Média:     {X_norm[0, 0].mean():.6f}  (≈ 0)")
    print(f"    Variância: {X_norm[0, 0].var():.6f}  (≈ 1)")
    print(f"    Std:       {X_norm[0, 0].std():.6f}  (≈ 1)")

    print("\n" + "=" * 55)
    print("OPERAÇÃO COMPLETA: ADD & NORM")
    print("=" * 55)
    output = add_and_norm(X, sublayer_out)
    print(f"  Input shape:   {X.shape}")
    print(f"  Output shape:  {output.shape}  (preservado!)")

    # Verificar normalização em todos os tokens
    medias = output[0].mean(axis=-1)
    variancias = output[0].var(axis=-1)
    print(f"\n  Média por token (todos ≈ 0):")
    for i, m in enumerate(medias):
        print(f"    Token {i}: {m:.6f}")
    print(f"\n  Variância por token (todos ≈ 1):")
    for i, v in enumerate(variancias):
        print(f"    Token {i}: {v:.6f}")

    print("\n" + "=" * 55)
    print("POR QUE LAYERNORM E NÃO BATCHNORM?")
    print("=" * 55)
    print("  BatchNorm  → normaliza entre amostras do batch (axis=0)")
    print("               Problemático: frases têm tamanhos diferentes")
    print("  LayerNorm  → normaliza dentro de cada token (axis=-1)")
    print("               Independente do batch size e do seq length ✓")

    print("\n✅ Add & Norm output shape:", output.shape)
    assert output.shape == (B, T, D_MODEL), "ERRO: shape incorreto!"
    print("  Shape correto: (B, T, d_model) ✓")
