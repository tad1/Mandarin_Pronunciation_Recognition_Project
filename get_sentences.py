import polars as pl
from tones import get_tones




n = 10
df_csv = pl.read_csv("../../../Data/cv-corpus-20.0-2024-12-06/zh-CN/validated.tsv", separator="\t")
df_csv = df_csv.sample(n, with_replacement=False)
df_csv = df_csv.select(["sentence_id", "sentence"])
print(df_csv)

for row in df_csv.iter_rows(named=True):
    sentence_id = row["sentence_id"]
    sentence = row["sentence"]
    print(f"Sentence ID: {sentence_id}")
    print(f"Sentence: {sentence}")
    pinyin, tones = get_tones(sentence)
    print(f"Pinyin: {pinyin}")
    print(f"Tones: {tones}")
