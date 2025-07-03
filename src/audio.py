# The purpose of this code, is to:
# - provide a source of truth for the sampling rate
# - handle conversion from path to audio samples in desired format
from __future__ import annotations
from typing import Literal, overload, TYPE_CHECKING
from os.path import normpath
from path import fix_path
import re

if TYPE_CHECKING:
    import torch
    import numpy


PROJECT_SAMPLING_RATE = 16000


@overload
def load_audio(path: str, return_type: Literal["numpy"]) -> numpy.ndarray: ...
@overload
def load_audio(path: str, return_type: Literal["torchaudio"]) -> torch.Tensor: ...
# Handle `return_type` hinting
@overload
def load_audio(path: str, return_type: Literal["numpy", "torchaudio"]): ...

def load_audio(path: str, return_type: str):
    path = fix_path(path)  # todo check if it's neseceary
    match return_type:
        case "torchaudio":
            return load_audio_torchaudio(path)
        case "numpy":
            return load_audio_librosa(path)
        case _:
            raise ValueError(f"Unsupported return type: {return_type}")


def load_audio_librosa(path: str) -> numpy.ndarray:
    import librosa
    audio, _ = librosa.load(path, sr=PROJECT_SAMPLING_RATE)
    return audio

def load_audio_torchaudio(path: str) -> torch.Tensor:
    from torchaudio import load
    from torchaudio.functional import resample

    audio, sr = load(path)
    if sr != PROJECT_SAMPLING_RATE:
        audio = resample(audio, sr, PROJECT_SAMPLING_RATE)
    return audio

if __name__ == "__main__":
    from data.source.pg_experiment import AUDIO_PATH, get_pg_experiment_dataset
    import os
    
    print("Loading audio file...")
    df_pronun, _ = get_pg_experiment_dataset()
    files = [os.path.join(AUDIO_PATH, path.replace("/","\\")) for path in df_pronun["rec_path"]]
    file = files[0]
    
    audio = load_audio(file, return_type="torchaudio")
    
    print(file)
    print(audio)
    print(audio.shape)
    
    print("Librosa version:")
    audio = load_audio(file, return_type="numpy")
    print(audio)