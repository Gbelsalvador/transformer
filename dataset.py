"""
Dataset et collate function pour l'entraînement du Transformer sur des
corpus de traduction parallèles (section 5.1 du papier).

Format attendu : deux fichiers texte alignés ligne à ligne (source / cible),
comme les corpus WMT 2014 EN-DE / EN-FR utilisés dans le papier.
"""

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer_utils import encode_sentence, PAD_ID


class TranslationDataset(Dataset):
    """Charge deux fichiers parallèles (une phrase par ligne) et les tokenise."""

    def __init__(self, src_file, tgt_file, tokenizer, max_len=128):
        with open(src_file, encoding="utf-8") as f:
            src_lines = [line.strip() for line in f if line.strip()]
        with open(tgt_file, encoding="utf-8") as f:
            tgt_lines = [line.strip() for line in f if line.strip()]

        assert len(src_lines) == len(tgt_lines), (
            "Les fichiers source et cible doivent avoir le même nombre de lignes "
            f"({len(src_lines)} vs {len(tgt_lines)})"
        )

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs = list(zip(src_lines, tgt_lines))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        src_ids = encode_sentence(self.tokenizer, src_text)[: self.max_len]
        tgt_ids = encode_sentence(self.tokenizer, tgt_text)[: self.max_len]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def collate_batch(batch):
    """
    Assemble un batch en complétant (<pad>) chaque séquence jusqu'à la
    longueur maximale du batch (cf. "batched together by approximate
    sequence length", section 5.1).
    """
    src_batch, tgt_batch = zip(*batch)

    src_max_len = max(len(s) for s in src_batch)
    tgt_max_len = max(len(t) for t in tgt_batch)

    def pad(seq, max_len):
        return torch.cat(
            [seq, torch.full((max_len - len(seq),), PAD_ID, dtype=torch.long)]
        )

    src_padded = torch.stack([pad(s, src_max_len) for s in src_batch])
    tgt_padded = torch.stack([pad(t, tgt_max_len) for t in tgt_batch])

    return src_padded, tgt_padded


def make_dataloader(src_file, tgt_file, tokenizer, batch_size=32, max_len=128, shuffle=True):
    dataset = TranslationDataset(src_file, tgt_file, tokenizer, max_len)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch
    )


if __name__ == "__main__":
    from tokenizer_utils import load_tokenizer

    tok = load_tokenizer("test_data/tokenizer.json")
    loader = make_dataloader(
        "test_data/train.en", "test_data/train.de", tok, batch_size=4
    )

    for src, tgt in loader:
        print("Batch source :", src.shape, "\n", src)
        print("Batch cible  :", tgt.shape, "\n", tgt)
        break
