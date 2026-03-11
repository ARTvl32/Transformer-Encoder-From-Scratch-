"""
Passo 3: Encoder Completo — Pilha de N=6 Camadas
==================================================
Laboratório P1-02 — Transformer Encoder From Scratch
Disciplina: Tópicos em IA — iCEV 2026.1
Prof. Dimmy Magalhães

Pipeline de cada camada do Encoder:
    1. X_att   = SelfAttention(X)
    2. X_norm1 = LayerNorm(X + X_att)         ← Add & Norm #1
    3. X_ffn   = FFN(X_norm1)
    4. X_out   = LayerNorm(X_norm1 + X_ffn)   ← Add & Norm #2
    5. X       = X_out  (output vira input da próxima camada)

Validação de Sanidade:
    Tensor entra na Camada 1 com (B, T, d_model)
    e DEVE sair da Camada 6 com (B, T, d_model)
    — mesmas dimensões, vetores contextualizados (Z).
"""

import numpy as np
import pandas as pd

# ── Importar módulos anteriores ────────────────────────────────────────
from step1_data_prep   import build_vocabulary, sentence_to_ids, \
                              build_embedding_table, ids_to_tensor
from step2_attention   import softmax, init_projection_weights, \
                              scaled_dot_product_attention
from step3_add_norm    import add_and_norm, layer_norm
from step4_ffn         import init_ffn_weights, feed_forward_network

# ─────────────────────────────────────────
# Hiperparâmetros Arquiteturais
# ─────────────────────────────────────────
D_MODEL     = 64        # paper usa 512
D_K         = D_MODEL   # dimensão das projeções Q, K
D_V         = D_MODEL   # dimensão das projeções V
D_FF        = D_MODEL * 4   # 256 (paper usa 2048)
N_LAYERS    = 6         # número de camadas do Encoder
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ──────────────────────────────────────────────────────────────────────
# Classe: EncoderLayer
# Representa UMA camada do Encoder com pesos INDEPENDENTES
# ──────────────────────────────────────────────────────────────────────
class EncoderLayer:
    """
    Uma camada do Encoder do Transformer.

    Cada camada contém pesos INDEPENDENTES — não compartilhados
    com outras camadas. Isso permite que camadas inferiores
    especializem em sintaxe local e camadas superiores em semântica global.

    Sub-camadas:
        1. Multi-Head Self-Attention (simplificado: single-head neste lab)
        2. Position-wise Feed-Forward Network
    Cada sub-camada envolvida por Add & Norm.
    """

    def __init__(self, d_model, d_k, d_v, d_ff, layer_id):
        self.layer_id = layer_id

        # Pesos da sub-camada 1: Self-Attention
        self.W_Q, self.W_K, self.W_V = init_projection_weights(d_model, d_k, d_v)

        # Pesos da sub-camada 2: FFN
        self.W1, self.b1, self.W2, self.b2 = init_ffn_weights(d_model, d_ff)

    def forward(self, X):
        """
        Forward pass de uma camada do Encoder.

        Fluxo exato conforme o laboratório:
            1. X_att   = SelfAttention(X)
            2. X_norm1 = LayerNorm(X + X_att)
            3. X_ffn   = FFN(X_norm1)
            4. X_out   = LayerNorm(X_norm1 + X_ffn)

        Args:
            X: np.ndarray de shape (B, T, d_model)

        Returns:
            X_out: np.ndarray de shape (B, T, d_model)
        """
        # ── Sub-camada 1: Self-Attention ──────────────────────────────
        X_att, attn_weights = scaled_dot_product_attention(
            X, self.W_Q, self.W_K, self.W_V
        )

        # Add & Norm #1
        X_norm1 = add_and_norm(X, X_att)

        # ── Sub-camada 2: Feed-Forward Network ────────────────────────
        X_ffn = feed_forward_network(X_norm1, self.W1, self.b1, self.W2, self.b2)

        # Add & Norm #2
        X_out = add_and_norm(X_norm1, X_ffn)

        return X_out


# ──────────────────────────────────────────────────────────────────────
# Classe: TransformerEncoder
# Pilha de N=6 camadas idênticas em estrutura, independentes em pesos
# ──────────────────────────────────────────────────────────────────────
class TransformerEncoder:
    """
    Encoder do Transformer: pilha de N camadas idênticas em estrutura.

    Cada camada tem pesos independentes — o modelo descobre hierarquia:
    - Camadas 1-2: relações sintáticas locais (artigo→substantivo)
    - Camadas 3-4: agrupamentos semânticos (sintagmas)
    - Camadas 5-6: desambiguação contextual global (banco = assento ou banco?)
    """

    def __init__(self, n_layers, d_model, d_k, d_v, d_ff):
        self.n_layers = n_layers
        self.d_model  = d_model

        # Criar N camadas com pesos INDEPENDENTES
        self.layers = [
            EncoderLayer(d_model, d_k, d_v, d_ff, layer_id=i+1)
            for i in range(n_layers)
        ]

    def encode(self, X):
        """
        Passa o tensor X sequencialmente por todas as N camadas.

        A saída de cada camada vira a entrada da próxima.
        A dimensão (B, T, d_model) é preservada ao longo de todo o fluxo.

        Args:
            X: np.ndarray de shape (B, T, d_model)

        Returns:
            Z: np.ndarray de shape (B, T, d_model) — representações contextualizadas
        """
        print(f"\n{'─'*55}")
        print(f"  FORWARD PASS — {self.n_layers} CAMADAS DO ENCODER")
        print(f"{'─'*55}")
        print(f"  Entrada:  shape={X.shape}  norma={np.linalg.norm(X):.4f}")

        current = X
        for layer in self.layers:
            current = layer.forward(current)
            print(f"  Camada {layer.layer_id}:  shape={current.shape}  "
                  f"norma={np.linalg.norm(current):.4f}")

            # Validação de sanidade a cada camada
            assert current.shape == X.shape, (
                f"ERRO: shape mudou na camada {layer.layer_id}! "
                f"Esperado {X.shape}, obtido {current.shape}"
            )

        Z = current
        print(f"\n  Saída Z:  shape={Z.shape}  norma={np.linalg.norm(Z):.4f}")
        return Z


# ──────────────────────────────────────────────────────────────────────
# Pipeline Completo: texto → IDs → Embeddings → Encoder → Z
# ──────────────────────────────────────────────────────────────────────
def run_full_pipeline():
    """
    Executa o pipeline completo do laboratório:
        Texto → IDs → Tensor de Embeddings → 6 Camadas do Encoder → Z
    """
    print("=" * 55)
    print("  TRANSFORMER ENCODER — FROM SCRATCH")
    print("  Laboratório P1-02 | iCEV 2026.1")
    print("=" * 55)

    # ── Passo 1: Preparar dados ────────────────────────────────────────
    print("\n📌 PASSO 1: PREPARAÇÃO DOS DADOS")
    vocab, _ = build_vocabulary()

    frase = ["o", "banco", "bloqueou", "o", "cartao"]
    print(f"\n  Frase de entrada: {' '.join(frase)}")

    ids = sentence_to_ids(frase, vocab)
    embedding_table = build_embedding_table(len(vocab), D_MODEL)
    X = ids_to_tensor(ids, embedding_table)

    print(f"\n  Tensor X shape: {X.shape}  (Batch, Tokens, d_model)")

    # ── Passo 2+3: Construir e executar o Encoder ──────────────────────
    print("\n📌 PASSO 3: ENCODER — PILHA DE 6 CAMADAS")
    encoder = TransformerEncoder(
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        d_k=D_K,
        d_v=D_V,
        d_ff=D_FF
    )

    Z = encoder.encode(X)

    # ── Validação Final ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  VALIDAÇÃO DE SANIDADE FINAL")
    print("=" * 55)
    print(f"  Shape de entrada X: {X.shape}")
    print(f"  Shape de saída  Z:  {Z.shape}")

    assert Z.shape == X.shape, f"ERRO CRÍTICO: shapes diferentes! {Z.shape} ≠ {X.shape}"
    print("\n  ✅ X.shape == Z.shape  →  Dimensão preservada!")
    print("  ✅ Valores alterados   →  Representações contextualizadas!")

    # Verificar que os vetores foram realmente alterados (contextualização)
    diferenca = np.linalg.norm(Z - X)
    print(f"\n  Norma da diferença Z-X: {diferenca:.4f}")
    assert diferenca > 0, "ERRO: Z é igual a X — o Encoder não transformou nada!"
    print("  ✅ Z ≠ X  →  Encoder transformou os embeddings com sucesso!")

    # ── Resumo arquitetural ────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RESUMO ARQUITETURAL")
    print("=" * 55)
    print(f"  d_model:   {D_MODEL}  (paper: 512)")
    print(f"  d_ff:      {D_FF}  (paper: 2048 = 4 × 512)")
    print(f"  N camadas: {N_LAYERS}")
    print(f"  Tokens:    {X.shape[1]}")
    print(f"  Batch:     {X.shape[0]}")
    total_params = (
        N_LAYERS * (
            3 * D_MODEL * D_K +        # W_Q, W_K, W_V
            D_MODEL * D_FF + D_FF +    # W1, b1
            D_FF * D_MODEL + D_MODEL   # W2, b2
        )
    )
    print(f"  Parâmetros aprox: {total_params:,}")

    print("\n" + "=" * 55)
    print("  VETOR Z — PRIMEIROS 8 VALORES DO TOKEN 0")
    print("=" * 55)
    print(f"  Embedding original X[0,0,:8]: {X[0, 0, :8].round(4)}")
    print(f"  Repr. contextual  Z[0,0,:8]: {Z[0, 0, :8].round(4)}")
    print("\n  (Os vetores mudaram — 'banco' agora carrega contexto de")
    print("   'bloqueou' e 'cartao', não é mais um embedding genérico)")

    return Z


# ─────────────────────────────────────────
# Ponto de entrada principal
# ─────────────────────────────────────────
if __name__ == "__main__":
    Z = run_full_pipeline()
    print("\n🏁 Pipeline completo executado com sucesso!")
    print(f"   Vetor Z final shape: {Z.shape}")
