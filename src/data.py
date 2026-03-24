"""
Data loading for language model training on WikiText-2.

Demonstrates production DataLoader patterns: tokenization, batching,
sequence chunking, and performance tuning (num_workers, pin_memory).
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path


class LMDataset(Dataset):
    """
    Language modeling dataset: maps raw text into fixed-length
    (input, target) token-id sequences for next-token prediction.
    """

    def __init__(self, token_ids: list[int], seq_len: int = 128):
        self.seq_len = seq_len
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.n_samples = (len(self.token_ids) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.token_ids[start:end]
        y = self.token_ids[start + 1 : end + 1]
        return x, y


def train_tokenizer(
    texts: list[str], vocab_size: int = 8192, save_path: str | None = None
) -> Tokenizer:
    """Train a BPE tokenizer on the provided texts."""
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    if save_path:
        tokenizer.save(save_path)
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def prepare_wikitext2(
    tokenizer: Tokenizer | None = None,
    vocab_size: int = 8192,
    seq_len: int = 128,
    tokenizer_path: str = "tokenizer.json",
) -> tuple[LMDataset, LMDataset, LMDataset, Tokenizer]:
    """
    Download WikiText-2, train tokenizer (if needed), and return
    (train_dataset, val_dataset, test_dataset, tokenizer).
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    train_texts = [t for t in dataset["train"]["text"] if t.strip()]
    val_texts = [t for t in dataset["validation"]["text"] if t.strip()]
    test_texts = [t for t in dataset["test"]["text"] if t.strip()]

    if tokenizer is None:
        tokenizer_file = Path(tokenizer_path)
        if tokenizer_file.exists():
            tokenizer = load_tokenizer(str(tokenizer_file))
        else:
            tokenizer = train_tokenizer(
                train_texts, vocab_size=vocab_size, save_path=str(tokenizer_file)
            )

    def encode_texts(texts: list[str]) -> list[int]:
        all_ids = []
        for text in texts:
            encoded = tokenizer.encode(text)
            all_ids.extend(encoded.ids)
        return all_ids

    train_ids = encode_texts(train_texts)
    val_ids = encode_texts(val_texts)
    test_ids = encode_texts(test_texts)

    return (
        LMDataset(train_ids, seq_len=seq_len),
        LMDataset(val_ids, seq_len=seq_len),
        LMDataset(test_ids, seq_len=seq_len),
        tokenizer,
    )


def create_dataloaders(
    train_dataset: LMDataset,
    val_dataset: LMDataset,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders with production-grade settings.

    Key tuning knobs:
    - num_workers: 0 for CPU-only, 2-4 for GPU training
    - pin_memory: True when using CUDA (enables async H2D transfer)
    - persistent_workers: True to avoid worker respawn overhead between epochs
    """
    common = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **common,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )

    return train_loader, val_loader
