"""
Masques pour l'entraînement par batch.

`transformer.py` fournit déjà `subsequent_mask` (masque causal, section
3.2.3), mais un entraînement réel nécessite aussi de masquer les tokens
de padding (<pad>) pour qu'ils n'influencent jamais l'attention. Ce
module combine les deux.
"""

import torch
from transformer import subsequent_mask
from tokenizer_utils import PAD_ID


def make_src_mask(src):
    """
    Masque de padding pour l'encodeur : True là où le token n'est PAS du padding.
    Forme : (batch, 1, src_len) — broadcastable sur les têtes d'attention.
    """
    return (src != PAD_ID).unsqueeze(1)


def make_tgt_mask(tgt):
    """
    Combine le masque de padding et le masque causal pour le décodeur
    (section 3.2.3 : empêcher l'accès aux positions futures, ET ignorer
    les positions de padding).
    Forme : (batch, tgt_len, tgt_len)
    """
    pad_mask = (tgt != PAD_ID).unsqueeze(1)              # (batch, 1, tgt_len)
    causal_mask = subsequent_mask(tgt.size(1)).to(tgt.device)  # (1, tgt_len, tgt_len)
    return pad_mask & causal_mask                          # broadcast -> (batch, tgt_len, tgt_len)


if __name__ == "__main__":
    src = torch.tensor([[1, 5, 6, 2, 0, 0]])   # 2 tokens de padding à la fin
    tgt = torch.tensor([[1, 7, 8, 9, 2, 0]])

    print("Masque source (padding) :")
    print(make_src_mask(src))

    print("\nMasque cible (causal + padding) :")
    print(make_tgt_mask(tgt))
