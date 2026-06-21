"""
Génération (traduction) avec un modèle Transformer entraîné.

D'après la section 6.1 du papier :
    "We used beam search with a beam size of 4 and length penalty
     alpha = 0.6 ... We set the maximum output length during inference
     to input length + 50, but terminate early when possible."

Ce script propose deux modes : décodage glouton (greedy, plus simple et
plus rapide) et beam search (fidèle au protocole d'évaluation du papier).
"""

import argparse

import torch

from transformer import Transformer, subsequent_mask
from tokenizer_utils import load_tokenizer, encode_sentence, decode_ids, SOS_ID, EOS_ID, PAD_ID
from masking import make_src_mask


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = Transformer(
        src_vocab_size=ckpt["vocab_size"],
        tgt_vocab_size=ckpt["vocab_size"],
        d_model=ckpt["d_model"],
        N=ckpt["N"],
        h=ckpt["h"],
        d_ff=ckpt["d_ff"],
        dropout=ckpt["dropout"],
        max_len=ckpt.get("max_len", 5000),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def greedy_decode(model, src, src_mask, max_len, device):
    """Décodage glouton : à chaque pas, on choisit le token le plus probable."""
    memory = model.encode(src, src_mask)
    ys = torch.full((src.size(0), 1), SOS_ID, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = subsequent_mask(ys.size(1)).to(device)
        out = model.decode(ys, memory, src_mask, tgt_mask)
        logits = model.generator(out[:, -1])
        next_token = logits.argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)

        if (next_token == EOS_ID).all():
            break

    return ys


@torch.no_grad()
def beam_search_decode(model, src, src_mask, max_len, device, beam_size=4, length_penalty=0.6):
    """
    Beam search avec pénalité de longueur, comme décrit en section 6.1 :
    beam_size=4, alpha=0.6 par défaut (valeurs du papier).
    Ne supporte qu'un seul exemple à la fois (batch_size=1) pour rester simple.
    """
    assert src.size(0) == 1, "beam_search_decode ne supporte qu'un exemple à la fois"

    def normalized_score(seq, score):
        # Pénalité de longueur usuelle : score / ((5+len)/6)^alpha (cf. [38], section 6.1)
        length = seq.size(1)
        penalty = ((5 + length) / 6) ** length_penalty
        return score / penalty

    memory = model.encode(src, src_mask)  # (1, src_len, d_model)

    # Chaque candidat actif du faisceau : (séquence, log-prob cumulée)
    beams = [(torch.tensor([[SOS_ID]], device=device), 0.0)]
    completed = []

    for _ in range(max_len - 1):
        if not beams:
            break

        candidates = []
        for seq, score in beams:
            tgt_mask = subsequent_mask(seq.size(1)).to(device)
            out = model.decode(seq, memory, src_mask, tgt_mask)
            logits = model.generator(out[:, -1])
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

            topk_log_probs, topk_ids = log_probs.topk(beam_size)
            for lp, idx in zip(topk_log_probs, topk_ids):
                new_seq = torch.cat([seq, idx.view(1, 1)], dim=1)
                new_score = score + lp.item()
                if idx.item() == EOS_ID:
                    completed.append((new_seq, new_score))
                else:
                    candidates.append((new_seq, new_score))

        # Ne garder que les `beam_size` meilleurs candidats encore actifs
        candidates.sort(key=lambda item: normalized_score(*item), reverse=True)
        beams = candidates[:beam_size]

        # Arrêt anticipé si on a déjà assez de séquences terminées, et que la
        # meilleure séquence active ne peut plus les dépasser
        if len(completed) >= beam_size and beams:
            best_completed = max(normalized_score(*c) for c in completed)
            best_active = normalized_score(*beams[0])
            if best_completed >= best_active:
                break

    all_candidates = completed if completed else beams
    if not all_candidates:
        # Filet de sécurité : ne devrait pas arriver, mais évite un crash
        return torch.tensor([[SOS_ID, EOS_ID]], device=device)

    best_seq, _ = max(all_candidates, key=lambda item: normalized_score(*item))
    return best_seq


def translate(sentence, model, tokenizer, device, max_len=64, method="greedy", beam_size=4):
    src_ids = torch.tensor([encode_sentence(tokenizer, sentence)], device=device)
    src_mask = make_src_mask(src_ids)

    # Le positional encoding limite la longueur maximale gérable par le modèle
    pe_capacity = model.tgt_embed[1].pe.size(1)
    max_len = min(max_len, pe_capacity)

    if method == "greedy":
        out_ids = greedy_decode(model, src_ids, src_mask, max_len, device)
    else:
        out_ids = beam_search_decode(model, src_ids, src_mask, max_len, device, beam_size=beam_size)

    return decode_ids(tokenizer, out_ids[0].tolist())


def parse_args():
    p = argparse.ArgumentParser(description="Traduction avec un Transformer entraîné")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--sentence", required=True)
    p.add_argument("--method", choices=["greedy", "beam"], default="greedy")
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = load_tokenizer(args.tokenizer_path)
    model = load_model(args.checkpoint, device)

    translation = translate(
        args.sentence, model, tokenizer, device,
        max_len=args.max_len, method=args.method, beam_size=args.beam_size,
    )
    print(f"Source     : {args.sentence}")
    print(f"Traduction : {translation}")


if __name__ == "__main__":
    main()
