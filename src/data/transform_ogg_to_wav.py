import os
import subprocess
from glob import glob

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ogg_files = glob(
    os.path.join(
        BASE_DIR, "..", "..", "..", "pg_dataset", "recordings", "*", "*", "*.ogg"
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
