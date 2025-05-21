import os
import polars as pl
import numpy as np

from get_pitch import pitch
from tones import get_tones
import time

def get_dataset(size):
    df_csv = pl.read_csv("../../../Data/cv-corpus-20.0-2024-12-06/zh-CN/validated.tsv", separator="\t")
    df_csv = df_csv.sample(size, with_replacement=False)
    df_csv = df_csv.select(["sentence", "path"])

    tone_sequences = []
    audio = []
    i = 0
    for row in df_csv.iter_rows(named=True):
        if i % 10 == 0:
            print(f"Processing {i} out of {size}")
        sentence = row["sentence"]
        path = row["path"]
        mp3_file = os.path.join("../../../Data/cv-corpus-20.0-2024-12-06/zh-CN/clips/", path)
        
        res = pitch(mp3_file)
        tones = get_tones(sentence)
        
        tone_sequences.append(tones)
        audio.append(res)
        i += 1
    return (audio, tone_sequences)

if __name__ == "__main__":
    import pickle
    dataset = get_dataset(5000)
    # save to picle
    with open('./data/random_5000_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)