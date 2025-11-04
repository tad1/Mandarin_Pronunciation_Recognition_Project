
from typing import Literal
import torchaudio.transforms as T
from silero_vad import collect_chunks, load_silero_vad, get_speech_timestamps
from audio import load_audio, PROJECT_SAMPLING_RATE
import logging
import librosa
import torch

class Sequence():
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, path):
        results = []
        for transform in self.transforms:
            results.append(transform(path))
        return results

class VadAudio():
    def __init__(
        self,
        sample_rate:int=PROJECT_SAMPLING_RATE,
        padding:None|int = None,
    ):
        self.sample_rate = sample_rate
        self.vad = load_silero_vad()
        self.padding = padding
        

    def __call__(self, path):
        waveform = load_audio(path, return_type="torchaudio")
        stamps = get_speech_timestamps(
            waveform, self.vad, sampling_rate=PROJECT_SAMPLING_RATE, min_speech_duration_ms=100
        )
        trimmed_waveform = waveform
        if len(stamps) != 0:
            trimmed_waveform = collect_chunks(stamps, waveform)
        else:
            logging.warning(path, "has no speech segments, using full waveform")
        if self.padding is not None:
            if trimmed_waveform.shape[1] < self.padding:
                padding_size = self.padding - trimmed_waveform.shape[1]
                padded_waveform = T.PadTrim(padding_size)(trimmed_waveform)
                trimmed_waveform = padded_waveform
            elif trimmed_waveform.shape[1] > self.padding:
                trimmed_waveform = T.PadTrim(self.padding)(trimmed_waveform)
        return trimmed_waveform

class Channels():
    def __init__(self, merge_mode:Literal["stack", "cat"], input_mode: Literal["one-to-one", "multiply"], dim=0):
        self.mode = merge_mode
        self.dim = dim
        self.input_mode = input_mode

    def __call__(self, *transforms):
        res = _Channels(*transforms)
        res.dim = self.dim
        res.mode = self.mode
        res.input_mode = self.input_mode
        return res
    
class RMSEnergy():
    def __init__(self, frame_length=1024, hop_length=512):
        self.frame_length = frame_length
        self.hop_length = hop_length

    def __call__(self, path):
        waveform = load_audio(path, return_type="torchaudio").numpy()
        rms = librosa.feature.rms(
            y=waveform,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )
        return torch.tensor(rms, dtype=torch.float32)

class ZeroCrossingRate():
    def __init__(self, frame_length=1024, hop_length=512):
        self.frame_length = frame_length
        self.hop_length = hop_length

    def __call__(self, path):
        waveform = load_audio(path, return_type="torchaudio").numpy()
        zcr = librosa.feature.zero_crossing_rate(
            y=waveform,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )
        return torch.tensor(zcr, dtype=torch.float32)

class FundamentalFrequency():
    def __call__(self, path):
        waveform = load_audio(path, return_type="torchaudio").numpy()
        f0 = librosa.yin(
            y=waveform,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=PROJECT_SAMPLING_RATE
        )
        return torch.tensor(f0, dtype=torch.float32).reshape((1, -1))
    

class _Channels():
    mode: Literal["stack", "cat"]
    input_mode: Literal["one-to-one", "multiply"]
    dim: int = 0
    
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, *inputs):
        if self.input_mode == "one-to-one":
            assert(len(inputs) == len(self.transforms)), "Number of inputs must match number of transforms in one-to-one mode"
            results = [transform(input) for transform, input in zip(self.transforms, inputs)]
        else:
            assert(len(inputs) == 1), "Inputs must be one when using multiply mode"
            results = [transform(input) for transform in self.transforms for input in inputs]
        
        if self.mode == "cat":
            return torch.cat(results, dim=self.dim)
        return torch.stack(results, dim=self.dim)
        
class TorchVadLogMelSpec():
    def __init__(
        self,
        sample_rate=PROJECT_SAMPLING_RATE,
        n_mels=128,
    ):
    
        # Cleaning with VAD
        # filter_mask = pl.Series(
        #     [vad_detected(os.path.join(base_dir, path)) for path in data["rec_path"].to_list()]
        # )
        # self.data = data.filter(filter_mask)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=512,
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        self.vad = load_silero_vad()

        
    def __call__(self, path):
        waveform = load_audio(path, return_type="torchaudio")
        stamps = get_speech_timestamps(
            waveform, self.vad, sampling_rate=PROJECT_SAMPLING_RATE, min_speech_duration_ms=100
        )
        trimmed_waveform = waveform
        if len(stamps) != 0:
            trimmed_waveform = collect_chunks(stamps, waveform)
        else:
            print(path, "has no speech segments, using full waveform")
        
        trimmed_waveform = trimmed_waveform.reshape((1, -1))
        # Log-Mel features
        mel_spec = self.mel_spectrogram(trimmed_waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (
            log_mel_spec.std() + 1e-6
        )
        return log_mel_spec


class TorchVadMFCC():
    def __init__(
        self,
        sample_rate=PROJECT_SAMPLING_RATE,
        n_mfcc=40,
        n_mels=128,
        n_fft=1024,
        delta=0
    ):
    
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.delta=delta
        self.mfcc = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs={
                "n_mels": n_mels,
                "hop_length": 512,
                "n_fft": n_fft,
            },

        )
        self.compute_deltas= T.ComputeDeltas()
        self.vad = load_silero_vad()
        
    def __call__(self, path):
        waveform = load_audio(path, return_type="torchaudio")
        stamps = get_speech_timestamps(
            waveform, self.vad, sampling_rate=PROJECT_SAMPLING_RATE, min_speech_duration_ms=100
        )
        trimmed_waveform = waveform
        if len(stamps) != 0:
            trimmed_waveform = collect_chunks(stamps, waveform)
        else:
            print(path, "has no speech segments, using full waveform")
        
        mfcc_spec = self.mfcc(trimmed_waveform)
        for i in range(self.delta):
            mfcc_spec = self.compute_deltas(mfcc_spec)
        return mfcc_spec
    
if __name__ == "__main__":
    # Example usage
    from data.source.pg_experiment import get_pg_experiment_dataframe
    from audio import load_audio
    pron, _ = get_pg_experiment_dataframe()
    
    transformation = Channels(merge_mode="stack", input_mode="multiply")(
        TorchVadMFCC(delta=1),
        TorchVadMFCC(delta=0),
    )
    
    transformation2 = Channels(merge_mode="stack", input_mode="multiply")(
        ZeroCrossingRate(),
        RMSEnergy(),
    )

    audio_path = pron[0]["rec_path"].to_list()[0]
    trans = transformation(audio_path)
    print(trans.shape)
    delta1, mfcc = trans.unbind()
    print(delta1.shape, mfcc.shape)
    
    trans2 = transformation2(audio_path)
    print(trans2.shape)
    zcr, rms = trans2.unbind()
    print(zcr.shape, rms.shape)
    
    # rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512)
    # print(rms.shape)
    # # Visualize the results
    # times = librosa.times_like(rms, sr=PROJECT_SAMPLING_RATE, hop_length=512)
    # plt.plot(times, rms[0])
    # plt.title("Root Mean Square Energy")
    # plt.xlabel("Time (seconds)")
    # plt.ylabel("RMS Energy")
    # plt.show()
    
    # librosa.display.specshow(delta1.numpy(), sr=PROJECT_SAMPLING_RATE, x_axis='time', y_axis='mel')
    # plt.show()
    # plt.figure()
    # librosa.display.specshow(mfcc.numpy(), sr=PROJECT_SAMPLING_RATE, x_axis='time', y_axis='mel')
    # plt.show()
    