
import torchaudio.transforms as T
from silero_vad import collect_chunks, load_silero_vad, get_speech_timestamps
from audio import load_audio, PROJECT_SAMPLING_RATE

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
    ):
    
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
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
        mfcc_spec = self.mfcc(trimmed_waveform)
        return mfcc_spec