"""
Fonction de perte avec label smoothing.

D'après la section 5.4 du papier :
    "During training, we employed label smoothing of value eps_ls = 0.1 [36].
     This hurts perplexity, as the model learns to be more unsure, but
     improves accuracy and BLEU score."

On utilise ici `nn.KLDivLoss` avec une distribution cible "adoucie" plutôt
qu'un one-hot strict, comme c'est l'usage standard pour implémenter le
label smoothing (Szegedy et al. 2015, référence [36] du papier).
"""

import torch
import torch.nn as nn

from tokenizer_utils import PAD_ID


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1, pad_id=PAD_ID):
        """
        smoothing = eps_ls = 0.1, comme indiqué section 5.4.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.pad_id = pad_id
        self.criterion = nn.KLDivLoss(reduction="sum")

    def forward(self, logits, target):
        """
        logits : (batch * seq_len, vocab_size) — sorties brutes du générateur
        target : (batch * seq_len,) — indices des tokens cibles
        """
        log_probs = torch.log_softmax(logits, dim=-1)

        true_dist = torch.zeros_like(log_probs)
        # masse répartie uniformément sur les (vocab_size - 2) tokens restants
        # (on exclut le token correct et le token de padding)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_id] = 0

        # ignorer les positions de padding dans la cible
        mask = (target != self.pad_id).unsqueeze(1)
        true_dist = true_dist * mask

        n_tokens = mask.sum().item()
        loss = self.criterion(log_probs, true_dist)
        return loss / max(n_tokens, 1)  # normalisé par nombre de tokens réels


if __name__ == "__main__":
    vocab_size = 50
    criterion = LabelSmoothingLoss(vocab_size, smoothing=0.1)

    logits = torch.randn(6, vocab_size)  # 6 tokens (batch*seq aplati)
    target = torch.tensor([3, 7, 0, 12, 0, 9])  # deux positions de padding (id=0)

    loss = criterion(logits, target)
    print("Loss avec label smoothing :", loss.item())
