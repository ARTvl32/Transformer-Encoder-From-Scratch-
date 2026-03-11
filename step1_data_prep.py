"""
Passo 1: Preparação dos Dados
==============================
Laboratório P1-02 — Transformer Encoder From Scratch
Disciplina: Tópicos em IA — iCEV 2026.1
Prof. Dimmy Magalhães

Objetivo:
    - Criar vocabulário com pandas
    - Converter frase em IDs
    - Inicializar tabela de embeddings
    - Produzir tensor de entrada (Batch, SequenceLength, d_model)
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────
# Hiperparâmetros
# ─────────────────────────────────────────
D_MODEL = 64        # paper usa 512; usamos 64 para CPU
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def build_vocabulary():
    """
    Cria um vocabulário simples mapeando palavras para IDs inteiros.
    Retorna um DataFrame pandas e um dicionário palavra->ID.
    """
    words = ["<PAD>", "o", "banco", "bloqueou", "cartao",
             "gato", "comeu", "rato", "no", "telhado"]

    vocab = {word: idx for idx, word in enumerate(words)}

    df_vocab = pd.DataFrame(
        list(vocab.items()),
        columns=["palavra", "id"]
    )

    print("=" * 50)
    print("VOCABULÁRIO")
    print("=" * 50)
    print(df_vocab.to_string(index=False))
    print(f"\nTamanho do vocabulário: {len(vocab)} palavras")

    return vocab, df_vocab


def sentence_to_ids(sentence, vocab):
    """
    Converte uma frase (lista de palavras) em lista de IDs inteiros.

    Args:
        sentence: list[str] — lista de palavras
        vocab: dict — mapeamento palavra->ID

    Returns:
        list[int] — lista de IDs correspondentes
    """
    ids = [vocab[word] for word in sentence]

    print("\n" + "=" * 50)
    print("FRASE → IDs")
    print("=" * 50)
    for word, idx in zip(sentence, ids):
        print(f"  '{word}' → {idx}")

    return ids


def build_embedding_table(vocab_size, d_model):
    """
    Inicializa a tabela de embeddings com valores aleatórios.
    Shape: (vocab_size, d_model)

    Em um modelo real, esses pesos são aprendidos durante o treinamento.
    Aqui simulamos com distribuição normal padrão.

    Args:
        vocab_size: int — número de palavras no vocabulário
        d_model: int — dimensão dos vetores de embedding

    Returns:
        np.ndarray de shape (vocab_size, d_model)
    """
    embedding_table = np.random.randn(vocab_size, d_model) * 0.1

    print("\n" + "=" * 50)
    print("TABELA DE EMBEDDINGS")
    print("=" * 50)
    print(f"  Shape: {embedding_table.shape}")
    print(f"  (vocab_size={vocab_size}, d_model={d_model})")
    print(f"  Média: {embedding_table.mean():.4f}")
    print(f"  Std:   {embedding_table.std():.4f}")

    return embedding_table


def ids_to_tensor(ids, embedding_table):
    """
    Converte lista de IDs no tensor de entrada 3D do Encoder.

    1. Lookup: extrai vetores da tabela para cada ID → shape (T, d_model)
    2. Adiciona dimensão batch → shape (1, T, d_model)

    O Transformer exige 3 dimensões:
        - Dim 0 (Batch): permite processar múltiplas frases em paralelo
        - Dim 1 (Sequence): posições dos tokens
        - Dim 2 (d_model): representação vetorial de cada token

    Args:
        ids: list[int] — IDs dos tokens
        embedding_table: np.ndarray de shape (vocab_size, d_model)

    Returns:
        np.ndarray de shape (1, T, d_model)
    """
    # Lookup: (T, d_model)
    X_2d = embedding_table[ids]

    # Adicionar dimensão batch: (1, T, d_model)
    X_3d = np.expand_dims(X_2d, axis=0)

    print("\n" + "=" * 50)
    print("TENSOR DE ENTRADA X")
    print("=" * 50)
    print(f"  Após lookup (2D):         {X_2d.shape}  → (Tokens, d_model)")
    print(f"  Após expand_dims (3D):    {X_3d.shape} → (Batch, Tokens, d_model)")
    print(f"\n  Batch Size:       {X_3d.shape[0]}")
    print(f"  Sequence Length:  {X_3d.shape[1]} tokens")
    print(f"  d_model:          {X_3d.shape[2]}")

    return X_3d


# ─────────────────────────────────────────
# Execução standalone (teste do módulo)
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔷 PASSO 1: PREPARAÇÃO DOS DADOS\n")

    # 1. Construir vocabulário
    vocab, df_vocab = build_vocabulary()

    # 2. Frase de entrada
    frase = ["o", "banco", "bloqueou", "o", "cartao"]
    ids = sentence_to_ids(frase, vocab)

    # 3. Tabela de embeddings
    embedding_table = build_embedding_table(len(vocab), D_MODEL)

    # 4. Tensor de entrada 3D
    X = ids_to_tensor(ids, embedding_table)

    print("\n" + "=" * 50)
    print("✅ VALIDAÇÃO FINAL")
    print("=" * 50)
    print(f"  Tensor X shape: {X.shape}")
    assert X.shape == (1, len(frase), D_MODEL), "ERRO: shape incorreto!"
    print("  Shape correto: (1, T, d_model) ✓")
