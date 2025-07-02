# The purpose of this code, is to:
# - provide a source of truth for the sampling rate
# - handle conversion from path to audio samples in desired format
from __future__ import annotations
from typing import Literal, overload, TYPE_CHECKING, Iterable
from os.path import normpath
import re

if TYPE_CHECKING:
    import torch


PROJECT_SAMPLING_RATE = 16000


@overload
def load_audio(path: str, return_type: Literal["torchaudio"]) -> torch.Tensor: ...

def load_audio(path: str, return_type: str):
    path = re.sub(r"/|\\", os.path.sep, path)  # todo check if it's neseceary
    match return_type:
        case "torchaudio":
            return load_audio_torchaudio(path)
        case _:
            raise ValueError(f"Unsupported return type: {return_type}")

@overload
def load_multiple_audio(paths: Iterable[str], return_type: Literal["torchaudio"]) -> Iterable[torch.Tensor]: ...
def load_multiple_audio(paths: Iterable[str], return_type: str) -> Iterable[torch.Tensor]:
    #TODO: handle loading in batch mode
    #also, move it to a single function?
    raise NotImplementedError("not implemented yet")


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
    print(audio.mean(0, keepdim=True).shape)
    print(audio.shape)