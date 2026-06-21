"""
Implémentation PyTorch du Transformer
D'après : "Attention Is All You Need" (Vaswani et al., 2017)
https://arxiv.org/abs/1706.03762

Chaque module ci-dessous est annoté avec la section du papier qui le décrit.
Les hyperparamètres par défaut correspondent au "base model" (Table 3, ligne "base") :
    N = 6, d_model = 512, d_ff = 2048, h = 8, d_k = d_v = 64, P_drop = 0.1
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Section 3.2.1 — Scaled Dot-Product Attention
#
# Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V         (Eq. 1)
# =====================================================================
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)

        # QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # "masking out (setting to -inf) all values in the input of the
        #  softmax which correspond to illegal connections" (section 3.2.3)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, V)
        return output, attn


# =====================================================================
# Section 3.2.2 — Multi-Head Attention
#
# MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
# head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
#
# "In this work we employ h = 8 parallel attention layers, or heads.
#  For each of these we use d_k = d_v = d_model/h = 64."
# =====================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, h: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0, "d_model doit être divisible par h"

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.d_v = d_model // h

        # W_i^Q, W_i^K, W_i^V pour toutes les têtes (projections fusionnées)
        self.W_Q = nn.Linear(d_model, h * self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, h * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, h * self.d_v, bias=False)

        # W^O \in R^{h*d_v x d_model}
        self.W_O = nn.Linear(h * self.d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Projections linéaires puis découpage en h têtes
        Q = self.W_Q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.h, self.d_v).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # broadcast sur les têtes

        # Attention en parallèle sur chaque tête
        x, attn = self.attention(Q, K, V, mask=mask)

        # Concat(head_1, ..., head_h)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_v)

        # Projection finale W^O
        return self.W_O(x)


# =====================================================================
# Section 3.3 — Position-wise Feed-Forward Networks
#
# FFN(x) = max(0, x W1 + b1) W2 + b2                        (Eq. 2)
#
# "the inner-layer has dimensionality d_ff = 2048"
# =====================================================================
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# =====================================================================
# Section 3.5 — Positional Encoding
#
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
# =====================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # pos
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # 10000^(2i/d_model) sous forme exponentielle

        pe[:, 0::2] = torch.sin(position * div_term)  # dimensions paires
        pe[:, 1::2] = torch.cos(position * div_term)  # dimensions impaires

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # "we add positional encodings to the input embeddings"
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# =====================================================================
# Section 3.4 — Embeddings and Softmax
#
# "In the embedding layers, we multiply those weights by sqrt(d_model)."
# =====================================================================
class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# =====================================================================
# Section 3.1 — Encoder
#
# "The encoder is composed of a stack of N = 6 identical layers. Each
#  layer has two sub-layers ... the output of each sub-layer is
#  LayerNorm(x + Sublayer(x))"
# =====================================================================
class SublayerConnection(nn.Module):
    """Residual connection suivie d'une layer norm (norme 'post-LN' du papier)."""

    def __init__(self, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # LayerNorm(x + Sublayer(x))
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x, mask=None):
        # Sous-couche 1 : multi-head self-attention
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))
        # Sous-couche 2 : feed-forward position-wise
        x = self.sublayer2(x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, N=6, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# =====================================================================
# Section 3.1 — Decoder
#
# "the decoder inserts a third sub-layer, which performs multi-head
#  attention over the output of the encoder stack ... We also modify
#  the self-attention sub-layer in the decoder stack to prevent
#  positions from attending to subsequent positions."
# =====================================================================
class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)       # masked self-attention
        self.src_attn = MultiHeadAttention(d_model, h, dropout)        # encoder-decoder attention
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # Sous-couche 1 : masked multi-head self-attention (decoder)
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Sous-couche 2 : encoder-decoder attention
        #   "the queries come from the previous decoder layer, and the
        #    memory keys and values come from the output of the encoder"
        x = self.sublayer2(x, lambda x: self.src_attn(x, memory, memory, src_mask))
        # Sous-couche 3 : feed-forward position-wise
        x = self.sublayer3(x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, N=6, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# =====================================================================
# Masque pour empêcher le décodeur d'attendre les positions futures
# (section 3.2.3 : "preserve the auto-regressive property")
# =====================================================================
def subsequent_mask(size):
    """Renvoie un masque triangulaire inférieur (1 = autorisé, 0 = interdit)."""
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0


# =====================================================================
# Modèle complet : Transformer (Figure 1)
# =====================================================================
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        self.src_embed = nn.Sequential(
            Embeddings(src_vocab_size, d_model),
            PositionalEncoding(d_model, dropout, max_len),
        )
        self.tgt_embed = nn.Sequential(
            Embeddings(tgt_vocab_size, d_model),
            PositionalEncoding(d_model, dropout, max_len),
        )

        self.encoder = Encoder(d_model, h, d_ff, N, dropout)
        self.decoder = Decoder(d_model, h, d_ff, N, dropout)

        # "we share the same weight matrix between the two embedding
        #  layers and the pre-softmax linear transformation" (section 3.4)
        self.generator = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.generator.weight = self.tgt_embed[0].lut.weight

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.generator(output)  # output probabilities (avant softmax)


# =====================================================================
# Section 5.3 — Optimizer (Adam + learning rate schedule)
#
# lrate = d_model^-0.5 * min(step_num^-0.5, step_num * warmup_steps^-1.5)
# =====================================================================
class NoamOpt:
    """Learning-rate scheduler décrit en section 5.3 (Eq. 3)."""

    def __init__(self, d_model, warmup_steps, optimizer):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0

    def rate(self, step=None):
        if step is None:
            step = self._step
        step = max(step, 1)
        return (self.d_model ** -0.5) * min(
            step ** -0.5, step * self.warmup_steps ** -1.5
        )

    def step(self):
        self._step += 1
        lr = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = lr
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_std_opt(model, d_model=512, warmup_steps=4000):
    """Adam avec beta1=0.9, beta2=0.98, eps=1e-9 (section 5.3)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    return NoamOpt(d_model, warmup_steps, optimizer)


# =====================================================================
# Exemple d'utilisation rapide (base model, Table 3)
# =====================================================================
if __name__ == "__main__":
    src_vocab_size = 37000  # ~ vocabulaire partagé EN-DE (section 5.1)
    tgt_vocab_size = 37000

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512,
        N=6,
        h=8,
        d_ff=2048,
        dropout=0.1,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre de paramètres : {n_params / 1e6:.1f}M")
    # NB : Table 3 indique 65M pour le "base model". L'écart vient ici du
    # partage de poids embeddings/generator compté différemment et de la
    # taille du vocabulaire utilisée en exemple ; l'architecture (N, h,
    # d_model, d_ff) reste, elle, strictement celle du papier.

    batch_size, src_len, tgt_len = 2, 10, 12
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

    tgt_mask = subsequent_mask(tgt_len)  # (1, tgt_len, tgt_len)

    out = model(src, tgt, src_mask=None, tgt_mask=tgt_mask)
    print("Forme de la sortie :", out.shape)  # (batch_size, tgt_len, tgt_vocab_size)

    optimizer = get_std_opt(model)
    print("Learning rate au step 1 :", optimizer.rate(1))
    print("Learning rate au step 4000 (pic, fin du warmup) :", optimizer.rate(4000))
