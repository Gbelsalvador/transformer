"""
Tokenizer BPE (Byte-Pair Encoding) pour l'entraînement du Transformer.

D'après la section 5.1 du papier :
    "Sentences were encoded using byte-pair encoding [3], which has a
     shared source-target vocabulary of about 37000 tokens."

Ce script entraîne un tokenizer BPE partagé sur les corpus source et cible
combinés (comme dans le papier), puis fournit des fonctions d'encodage et
de décodage utilisées par le reste du pipeline.
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Tokens spéciaux. <pad>=0 est important : le reste du pipeline suppose
# que l'index de padding est 0 (cf. masques et calcul de la loss).
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def train_bpe_tokenizer(files, vocab_size=37000, save_path="tokenizer.json"):
    """
    Entraîne un tokenizer BPE partagé sur une liste de fichiers texte
    (une phrase par ligne). Reproduit l'idée d'un "vocabulaire partagé
    source-cible" du papier (section 5.1).
    """
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )

    tokenizer.train(files, trainer)
    tokenizer.save(save_path)
    print(f"Tokenizer entraîné et sauvegardé : {save_path} "
          f"(vocabulaire réel : {tokenizer.get_vocab_size()} tokens)")
    return tokenizer


def load_tokenizer(path="tokenizer.json"):
    # ``tokenizers`` attend une chaîne de caractères ; sous Windows un objet
    # pathlib.Path ne peut pas être converti automatiquement par l'extension Rust.
    return Tokenizer.from_file(str(path))


def encode_sentence(tokenizer, sentence, add_sos_eos=True):
    """Encode une phrase en liste d'IDs, avec <sos>/<eos> optionnels."""
    ids = tokenizer.encode(sentence).ids
    if add_sos_eos:
        ids = [SOS_ID] + ids + [EOS_ID]
    return ids


def decode_ids(tokenizer, ids, skip_special_tokens=True):
    """Décode une liste d'IDs en texte."""
    return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


if __name__ == "__main__":
    # Exemple : entraîner un tokenizer partagé sur un petit corpus EN-DE
    tok = train_bpe_tokenizer(
        files=["data/train.fr", "data/train.ln"],
        vocab_size=3000,  # petit pour le test, 37000 en conditions réelles
        save_path="test_data/tokenizer.json",
    )

    example = "boza malamu ba ninga na nga ?"
    ids = encode_sentence(tok, example)
    print("Phrase :", example)
    print("IDs    :", ids)
    print("Décodé :", decode_ids(tok, ids))
