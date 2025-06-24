### Allows for use any of Mozilla's Common Voice dataset.

## Note, here's a coupling with the Filesystem
import os
from path import COMMON_VOICE_PATH, RESULT_DIRECTORY

AUDIO_PATH = COMMON_VOICE_PATH + "clips/"
VALIDATED_TSV = COMMON_VOICE_PATH + "validated.tsv"



CACHE_PATH = os.path.join(RESULT_DIRECTORY, "duration_index.parquet")

def get_common_voice_dataframe(useCache=True):
    import polars as pl
    import librosa
    
    df_csv = None
    if useCache:
        if os.path.exists(CACHE_PATH):
            print("Loading duration index from cache")
            df_csv = pl.read_parquet(CACHE_PATH)
        else:
            useCache = False
    
    if not useCache:
        df_csv = pl.read_csv(VALIDATED_TSV, separator="\t")
        
        df_csv = df_csv.with_columns(
            pl.col("sentence").str.strip_chars().str.replace_all(r"\s+", " ")
        )
        durations = [librosa.get_duration(path=os.path.join(AUDIO_PATH, path)) for path in df_csv["path"]]
        df_csv = df_csv.with_columns(pl.Series("duration", durations))
        df_csv = df_csv.sort(pl.col("duration"))
        df_csv.write_parquet(CACHE_PATH)

    df_csv = df_csv.filter(pl.col("down_votes") == 0)
    df_csv = df_csv.filter(pl.col("up_votes") > 2)
    return df_csv

if __name__ == "__main__":
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = get_common_voice_dataframe(useCache=True)

    print(df)
    # Calculate cumulative number of files over duration
    df = df.with_columns(pl.col("duration").cast(float))  # Ensure duration is float
    df_sorted = df.sort("duration")
    df_sorted = df_sorted.with_columns(pl.Series("cumulative_files", range(1, len(df_sorted) + 1)))

    # Plot cumulative number of files over duration
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=df_sorted["duration"], y=df_sorted["cumulative_files"])
    plt.title("Cumulative Number of Files Over Duration")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Cumulative Number of Files")
    plt.grid()
    plt.show()

    # Count number of files shorter than 1s, 2s, and 5s
    num_files_shorter_than_1s = df.filter(pl.col("duration") < 1).shape[0]
    num_files_shorter_than_2s = df.filter(pl.col("duration") < 2).shape[0]
    num_files_shorter_than_5s = df.filter(pl.col("duration") < 5).shape[0]

    print(f"Number of files shorter than 1 second: {num_files_shorter_than_1s}")
    print(f"Number of files shorter than 2 seconds: {num_files_shorter_than_2s}")
    print(f"Number of files shorter than 5 seconds: {num_files_shorter_than_5s}")