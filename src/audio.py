# The purpose of this code, is to:
# - provide a source of truth for the sampling rate
# - handle conversion from path to audio samples in desired format (and ensure it's in mono)
from __future__ import annotations
from typing import Literal, overload, TYPE_CHECKING
from path import fix_path

if TYPE_CHECKING:
    import torch
    import numpy


PROJECT_SAMPLING_RATE = 16000

# this project assumes using mono audio, to prevent any errors from stereo/mono ambiguity `load_audio` MUST return audio in mono format


@overload
def load_audio(path: str, return_type: Literal["numpy"]) -> numpy.ndarray: ...
@overload
def load_audio(path: str, return_type: Literal["torchaudio"]) -> torch.Tensor: ...
# Somehow Pylance gives only typing support for the last overload
# this is a workaround to support type hinting for `return_type`
@overload
def load_audio(path: str, return_type: Literal["numpy", "torchaudio"]): ...

def load_audio(path: str, return_type: str):
    path = fix_path(path)
    match return_type:
        case "torchaudio":
            return load_audio_torchaudio(path)
        case "numpy":
            return load_audio_librosa(path)
        case _:
            raise ValueError(f"Unsupported return type: {return_type}")


def load_audio_librosa(path: str) -> numpy.ndarray:
    from librosa import load
    audio, _ = load(path, sr=PROJECT_SAMPLING_RATE, mono=True)
    return audio

def load_audio_torchaudio(path: str) -> torch.Tensor:
    from torchaudio import load
    from torchaudio.functional import resample

    audio, sr = load(path)
    if sr != PROJECT_SAMPLING_RATE:
        audio = resample(audio, sr, PROJECT_SAMPLING_RATE)
    if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
    return audio.squeeze(0)

if __name__ == "__main__":
    from data.source.pg_experiment import AUDIO_PATH, get_pg_experiment_dataset
    import os
    
    print("Loading audio file...")
    df_pronun, _ = get_pg_experiment_dataset()
    files = [os.path.join(AUDIO_PATH, path.replace("/","\\")) for path in df_pronun["rec_path"]]
    file = files[0]
    
    for i in range(10):
        print(i)
        audio = load_audio(file, return_type="torchaudio")
    
    print(file)
    print(audio)
    print(audio.shape)
    
    print("Librosa version:")
    for i in range(10):
        print(i)
        audio = load_audio(file, return_type="numpy")
    print(audio)
    print(audio.shape)