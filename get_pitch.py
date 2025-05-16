import polars as pl
import subprocess
import os
import json

def pitch(filename):
    output = subprocess.check_output(["praat", "--run", "ExtractF0.praat", filename])
    pitch = output.decode("utf-8").split("\n")
    
    float_values = []
    for x in pitch:
        x = x.strip()
        if not x:
            continue
        try:
            float_values.append(float(x))
        except ValueError:
            float_values.append(None)
    
    return float_values

if __name__ == "__main__":
    n = 10
    df_csv = pl.read_csv("../../../Data/cv-corpus-20.0-2024-12-06/zh-CN/validated.tsv", separator="\t")
    df_csv = df_csv.sample(n, with_replacement=False)
    df_csv = df_csv.select(["sentence_id", "path"])
    print(df_csv)
    for row in df_csv.iter_rows(named=True):
        sentence_id = row["sentence_id"]
        path = row["path"]
        mp3_file = os.path.join("../../../Data/cv-corpus-20.0-2024-12-06/zh-CN/clips/", path)
        res = pitch(mp3_file)
        json_file = os.path.join("./pitches/", f"{sentence_id}.json")
        with open(json_file, "w") as f:
            json.dump(res, f)



