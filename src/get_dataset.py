import os
import polars as pl

from get_pitch import pitch
from tones import get_tones
from data.common_voice import get_common_voice_dataframe, AUDIO_PATH

def get_dataset(size, use_minimal_duration=False):
    df_csv = get_common_voice_dataframe()

    if use_minimal_duration:
        print("with sorting")
        df_csv = df_csv.sort(pl.col("duration"))
        df_csv = df_csv.head(size)
    else:
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
        mp3_file = os.path.join(AUDIO_PATH, path)
        
        res = pitch(mp3_file)
        tones = get_tones(sentence)
        
        tone_sequences.append(tones)
        audio.append(res)
        i += 1
    return (audio, tone_sequences)


if __name__ == "__main__":
    import pickle
    dataset = get_dataset(100, False)
    # save to picle
    # with open('./data/random_5000_dataset.pkl', 'wb') as f:
    #     pickle.dump(dataset, f)