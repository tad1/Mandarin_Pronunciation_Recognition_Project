import os
import polars as pl
from typing import Iterator
from get_pitch import pitch
from tones import get_tones
from data.common_voice import VALIDATED_TSV, AUDIO_PATH as CV_AUDIO_PATH

def get_info(sentence_ids) -> Iterator[tuple[str, str, str]]:
    """Get the info for a given sentence_id"""
    df_csv = pl.read_csv(VALIDATED_TSV, separator="\t")
    df_csv = df_csv.filter(pl.col("sentence_id").is_in(sentence_ids))
    df_csv = df_csv.select(["sentence_id", "path", "sentence"])
    return df_csv.iter_rows()

def extract_sequences(data):
    sequences = []
    current_seq = []

    for item in data:
        if item is not None:
            current_seq.append(item)
        else:
            if current_seq:
                sequences.append(current_seq)
                current_seq = []
    if current_seq:
        sequences.append(current_seq)

    return sequences

if __name__ == "__main__":
    res = get_info(["f75cf8748fadf741cc3c6549e3d5c6dbbec72b41662c90542c2c4f997edfb2b7"])
    for row in res:
        print(row)
        sentence_id = row[0]
        path = row[1]
        sentence = row[2]
        pitch_data = pitch(os.path.join(CV_AUDIO_PATH, path))
        tones = get_tones(sentence)
        sequences = extract_sequences(pitch_data)

        print("Sequences:")
        print(len(sequences))
        print("Tones:")
        print(len(tones))
        res = zip(sequences, tones)
        for seq, tone in res:
            print(f"Sequence: {seq}, Tone: {tone}")
        break