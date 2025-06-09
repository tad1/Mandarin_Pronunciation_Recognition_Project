import os
import subprocess
from glob import glob

ogg_files = glob("../../../recordings/stageI/*/*.ogg")


def oggToWav():
    for ogg_path in ogg_files:
        wav_path = ogg_path.replace(".ogg", ".wav")
        if not os.path.exists(wav_path):
            subprocess.run(
                ["ffmpeg", "-i", ogg_path, wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
