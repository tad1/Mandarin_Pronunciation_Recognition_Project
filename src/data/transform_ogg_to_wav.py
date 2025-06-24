import os
import subprocess
from glob import glob
from data.source.pg_experiment import AUDIO_PATH

ogg_files = glob(
    os.path.join(
        AUDIO_PATH, "*", "*", "*.ogg"
    )
)


def oggToWav():
    for ogg_path in ogg_files:
        wav_path = ogg_path.replace(".ogg", ".wav")
        if not os.path.exists(wav_path):
            subprocess.run(
                ["ffmpeg", "-i", ogg_path, wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


if __name__ == "__main__":
    oggToWav()
