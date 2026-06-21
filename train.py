"""
Boucle d'entraînement du Transformer sur une tâche de traduction,
suivant le protocole décrit en section 5 du papier ("Training") :

  - données : corpus parallèle source/cible tokenisé en BPE (5.1)
  - matériel/horaire : ici adapté au CPU/GPU disponible (5.2)
  - optimiseur : Adam + scheduler "Noam" (5.3)
  - régularisation : dropout + label smoothing (5.4)

Usage :
    python train.py \
        --src test_data/train.en \
        --tgt test_data/train.de \
        --epochs 20 \
        --d_model 128 --d_ff 256 --N 2 --h 4   # config réduite pour test rapide

Pour reproduire le "base model" du papier (Table 3), utiliser les valeurs
par défaut : d_model=512, d_ff=2048, N=6, h=8.
"""

import argparse
import time

import torch

from transformer import Transformer, get_std_opt
from tokenizer_utils import train_bpe_tokenizer, load_tokenizer, PAD_ID
from dataset import make_dataloader
from masking import make_src_mask, make_tgt_mask
from loss import LabelSmoothingLoss


def parse_args():
    p = argparse.ArgumentParser(description="Entraînement du Transformer (Vaswani et al., 2017)")
    p.add_argument("--src", required=True, help="Fichier texte source (une phrase par ligne)")
    p.add_argument("--tgt", required=True, help="Fichier texte cible (une phrase par ligne)")
    p.add_argument("--tokenizer_path", default="tokenizer.json")
    p.add_argument("--vocab_size", type=int, default=37000, help="Taille du vocabulaire BPE partagé (5.1)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max_len", type=int, default=128)

    # Hyperparamètres du modèle (par défaut : "base model", Table 3)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--N", type=int, default=6)
    p.add_argument("--h", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=4000)

    p.add_argument("--save_path", default="checkpoint.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_tokens = 0.0, 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        # Decoder input = cible décalée d'une position vers la droite
        # ("the output embeddings are offset by one position", section 3.1)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_input)

        logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_tokens = (tgt_output != PAD_ID).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device : {device}")

    # --- Tokenizer BPE partagé (section 5.1) ---
    try:
        tokenizer = load_tokenizer(args.tokenizer_path)
        print(f"Tokenizer chargé depuis {args.tokenizer_path}")
    except Exception:
        print("Aucun tokenizer existant trouvé, entraînement d'un nouveau tokenizer BPE...")
        tokenizer = train_bpe_tokenizer(
            files=[args.src, args.tgt],
            vocab_size=args.vocab_size,
            save_path=args.tokenizer_path,
        )

    vocab_size = tokenizer.get_vocab_size()

    # --- Données ---
    dataloader = make_dataloader(
        args.src, args.tgt, tokenizer, batch_size=args.batch_size, max_len=args.max_len
    )

    # --- Modèle (vocabulaire partagé source/cible, comme dans le papier) ---
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=args.d_model,
        N=args.N,
        h=args.h,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len + 10,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modèle initialisé : {n_params / 1e6:.2f}M paramètres")

    # --- Loss + optimiseur (sections 5.3 et 5.4) ---
    criterion = LabelSmoothingLoss(vocab_size, smoothing=args.label_smoothing)
    optimizer = get_std_opt(model, d_model=args.d_model, warmup_steps=args.warmup_steps)

    # --- Boucle d'entraînement ---
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        avg_loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        elapsed = time.time() - start
        print(f"Époque {epoch}/{args.epochs} — loss : {avg_loss:.4f} — {elapsed:.1f}s")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "d_model": args.d_model,
            "d_ff": args.d_ff,
            "N": args.N,
            "h": args.h,
            "dropout": args.dropout,
            "max_len": args.max_len + 10,
        },
        args.save_path,
    )
    print(f"Modèle sauvegardé : {args.save_path}")


if __name__ == "__main__":
    main()
