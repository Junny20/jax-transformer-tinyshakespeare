import numpy as np


def build_vocab(text: str):
    """Build character-level vocabulary from corpus text.

    Returns (stoi, itos, vocab_size) where stoi maps char→int and itos maps int→char.
    """
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return stoi, itos, len(chars)


def encode(text: str, stoi: dict) -> np.ndarray:
    return np.array([stoi[c] for c in text], dtype=np.int32)


def decode(ids, itos: dict) -> str:
    return ''.join(itos[int(i)] for i in ids)
