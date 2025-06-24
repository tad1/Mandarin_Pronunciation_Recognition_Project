import polars as pl
import os
import json
import parselmouth

def pitch(filename):
    snd = parselmouth.Sound(filename)
    pitch_obj = snd.to_pitch(time_step=None, pitch_floor=75, pitch_ceiling=600)
    return [pitch_obj.get_value_in_frame(i) for i in range(pitch_obj.get_number_of_frames())]


# def pitch(filename):
#     output = subprocess.check_output(["praat", "--run", "ExtractF0.praat", filename])
#     pitch = output.decode("utf-8").split("\n")
    
#     float_values = []
#     for x in pitch:
#         x = x.strip()
#         if not x:
#             continue
#         try:
#             float_values.append(float(x))
#         except ValueError:
#             float_values.append(None)
    
#     return float_values

if __name__ == "__main__":
    from data.common_voice import VALIDATED_TSV, AUDIO_PATH
    
    n = 10
    df_csv = pl.read_csv(VALIDATED_TSV, separator="\t")
    df_csv = df_csv.sample(n, with_replacement=False)
    df_csv = df_csv.select(["sentence_id", "path"])
    print(df_csv)
    for row in df_csv.iter_rows(named=True):
        sentence_id = row["sentence_id"]
        path = row["path"]
        mp3_file = os.path.join(AUDIO_PATH, path)
        res = pitch(mp3_file)
        # json_file = os.path.join("./pitches/", f"{sentence_id}.json")
        # with open(json_file, "w") as f:
            # json.dump(res, f)



