import csv

src_path = "data/train.ln"
tgt_path = "data/train.fr"

with open("lingala.csv", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    with open(src_path, "w", encoding="utf-8") as src, open(tgt_path, "w", encoding="utf-8") as tgt:
        for row in reader:
            src.write(row["ex_ln"].strip() + "\n")
            tgt.write(row["ex_fr"].strip() + "\n")