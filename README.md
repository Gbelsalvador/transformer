# Transformer en PyTorch — *Attention Is All You Need*

Implémentation PyTorch du Transformer, fidèle à l'architecture décrite dans l'article fondateur **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (Vaswani et al., 2017).

Chaque module du code est annoté avec la section du papier qu'il implémente, afin de servir à la fois de référence d'apprentissage et de base de travail réutilisable.

## Sommaire

- [Présentation](#présentation)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du code](#structure-du-code)
- [Hyperparamètres](#hyperparamètres)
- [Correspondance avec le papier](#correspondance-avec-le-papier)
- [Limites](#limites)
- [Référence](#référence)
- [Licence](#licence)

## Présentation

Le Transformer est un modèle de transduction de séquences qui abandonne entièrement la récurrence et les convolutions au profit de mécanismes d'attention. Cette implémentation reconstruit son architecture encodeur-décodeur telle que décrite dans le papier original, avec :

- l'attention "Scaled Dot-Product",
- l'attention multi-têtes (*Multi-Head Attention*),
- l'encodage positionnel sinusoïdal,
- les réseaux feed-forward position-wise,
- les connexions résiduelles et la normalisation de couche,
- le scheduler de taux d'apprentissage "Noam" utilisé pour l'entraînement.

## Architecture

Le modèle suit la structure encodeur-décodeur de la Figure 1 du papier :

- **Encodeur** : pile de `N = 6` couches identiques, chacune composée d'une sous-couche de self-attention multi-têtes et d'une sous-couche feed-forward, toutes deux entourées d'une connexion résiduelle suivie d'une normalisation de couche.
- **Décodeur** : pile de `N = 6` couches identiques, avec une sous-couche supplémentaire d'attention encodeur-décodeur, et un masquage causal sur la self-attention pour préserver la propriété auto-régressive.

## Installation

```bash
git clone <https://github.com/Gbelsalvador/transformer.git>
cd <transformer>
pip install torch tokenizers
```

## Utilisation

### Architecture seule

Lancer l'exemple fourni dans `transformer.py` (configuration "base model", Table 3 du papier, poids non entraînés) :

```bash
python transformer.py
```

### Entraînement complet sur un corpus de traduction

Le dépôt inclut un pipeline d'entraînement complet : tokenizer BPE, chargement de données, masques, perte avec label smoothing, boucle d'entraînement et génération (greedy / beam search).

**1. Préparer les données.** Deux fichiers texte alignés ligne à ligne (source et cible), au format des corpus WMT :

```
data/train.en
the cat sat on the mat
i love machine translation
...

data/train.de
die katze saß auf der matte
ich liebe maschinelle übersetzung
...
```

**2. Entraîner le modèle.** Le tokenizer BPE partagé est entraîné automatiquement au premier lancement s'il n'existe pas encore :

```bash
python train.py \
    --src data/train.en \
    --tgt data/train.de \
    --tokenizer_path tokenizer.json \
    --vocab_size 37000 \
    --epochs 20 \
    --save_path checkpoint.pt
```

Pour un test rapide sur petit corpus (modèle réduit) :

```bash
python train.py --src data/train.en --tgt data/train.de \
    --tokenizer_path tokenizer.json --vocab_size 300 \
    --batch_size 4 --epochs 50 --max_len 32 \
    --d_model 64 --d_ff 128 --N 2 --h 4 \
    --save_path checkpoint.pt
```

**3. Traduire avec le modèle entraîné :**

```bash
python translate.py \
    --checkpoint checkpoint.pt \
    --tokenizer_path tokenizer.json \
    --sentence "the cat sat on the mat" \
    --method beam
```

### Utiliser le modèle directement en Python

```python
import torch
from transformer import Transformer, subsequent_mask, get_std_opt

model = Transformer(
    src_vocab_size=37000,
    tgt_vocab_size=37000,
    d_model=512,
    N=6,
    h=8,
    d_ff=2048,
    dropout=0.1,
)

src = torch.randint(0, 37000, (2, 10))   # (batch, longueur source)
tgt = torch.randint(0, 37000, (2, 12))   # (batch, longueur cible)
tgt_mask = subsequent_mask(tgt.size(1))  # masque causal pour le décodeur

logits = model(src, tgt, src_mask=None, tgt_mask=tgt_mask)
# logits.shape == (2, 12, 37000)

# Optimiseur Adam + scheduler de taux d'apprentissage du papier (section 5.3)
optimizer = get_std_opt(model)
```

## Structure du code

| Fichier | Section du papier | Rôle |
|---|---|---|
| `transformer.py` | 3.1 à 3.5, Figure 1 | Architecture du modèle (encodeur, décodeur, attention, etc.) |
| `tokenizer_utils.py` | 5.1 | Tokenizer BPE partagé source/cible |
| `dataset.py` | 5.1 | Chargement et batching des corpus parallèles |
| `masking.py` | 3.2.3 | Masques de padding + masque causal combinés pour l'entraînement |
| `loss.py` | 5.4 | Perte avec label smoothing (ϵ_ls = 0.1) |
| `train.py` | 5.2, 5.3, 5.4 | Boucle d'entraînement complète |
| `translate.py` | 6.1 | Génération : décodage glouton et beam search avec pénalité de longueur |

Détail des modules de `transformer.py` :

| Module | Section du papier | Rôle |
|---|---|---|
| `ScaledDotProductAttention` | 3.2.1 | `Attention(Q,K,V) = softmax(QKᵀ/√d_k)V` |
| `MultiHeadAttention` | 3.2.2 | Projection en `h` têtes, attention en parallèle, concaténation |
| `PositionwiseFeedForward` | 3.3 | `FFN(x) = max(0, xW1+b1)W2+b2` |
| `PositionalEncoding` | 3.5 | Encodage positionnel par sinus/cosinus |
| `Embeddings` | 3.4 | Embeddings mis à l'échelle par `√d_model` |
| `EncoderLayer` / `Encoder` | 3.1 | Pile de `N` couches encodeur |
| `DecoderLayer` / `Decoder` | 3.1 | Pile de `N` couches décodeur, avec masquage |
| `subsequent_mask` | 3.2.3 | Masque triangulaire empêchant l'accès aux positions futures |
| `Transformer` | Figure 1 | Assemblage complet encodeur-décodeur |
| `NoamOpt` / `get_std_opt` | 5.3 | Scheduler de taux d'apprentissage (Équation 3) |

## Hyperparamètres

Les valeurs par défaut correspondent au **"base model"** de la Table 3 du papier :

| Paramètre | Valeur | Description |
|---|---|---|
| `N` | 6 | Nombre de couches encodeur/décodeur |
| `d_model` | 512 | Dimension des représentations |
| `d_ff` | 2048 | Dimension interne du feed-forward |
| `h` | 8 | Nombre de têtes d'attention |
| `d_k`, `d_v` | 64 | Dimension par tête (`d_model / h`) |
| `P_drop` | 0.1 | Taux de dropout |
| `warmup_steps` | 4000 | Étapes de montée du taux d'apprentissage |

Pour reproduire le **"big model"** (Table 3, dernière ligne), utiliser `d_model=1024`, `d_ff=4096`, `h=16`, `dropout=0.3`.

## Correspondance avec le papier

- ✅ Formule d'attention (Équation 1), telle quelle.
- ✅ Multi-head attention avec projections séparées par tête (section 3.2.2).
- ✅ Encodage positionnel sinusoïdal exact (section 3.5).
- ✅ Connexions résiduelles + LayerNorm post-sous-couche : `LayerNorm(x + Sublayer(x))` (section 3.1).
- ✅ Partage des poids entre les deux couches d'embedding et la transformation linéaire pré-softmax (section 3.4).
- ✅ Scheduler de taux d'apprentissage "Noam" (Équation 3, section 5.3).
- ⚠️ Le nombre de paramètres obtenu dans l'exemple (~82M) diffère légèrement des 65M annoncés en Table 3 : l'écart vient de la taille de vocabulaire choisie pour la démonstration, et non de l'architecture, qui reste conforme au papier.

## Limites

- Le beam search (`translate.py`) traite un seul exemple à la fois (`batch_size=1`) ; il n'est pas vectorisé pour plusieurs phrases en parallèle.
- Aucune donnée d'entraînement (WMT 2014 ou autre) n'est fournie : il faut apporter ses propres fichiers texte parallèles.
- Avec un corpus d'entraînement très réduit (quelques dizaines de phrases), le beam search peut converger vers des séquences très courtes (`<sos><eos>`) car le modèle sous-entraîné assigne une probabilité élevée à `<eos>` dès le premier pas — ce n'est pas un bug du code mais un effet attendu du manque de données ; cela disparaît avec un corpus de taille réaliste.
- Pas de support multi-GPU ni de gradient accumulation : pour reproduire fidèlement les temps d'entraînement de la section 5.2 (8 GPU P100), il faut adapter `train.py`.

## Référence

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}
```

Article original : [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

## Licence

Ce code est fourni à des fins pédagogiques et de réimplémentation. Le papier original est publié par Google sous une licence permettant la reproduction de ses figures et tableaux à des fins journalistiques ou académiques avec attribution appropriée.
