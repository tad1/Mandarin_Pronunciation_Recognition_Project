import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from src.audio import load_audio, PROJECT_SAMPLING_RATE
import polars as pl
import os
import torch.nn.functional as F
from silero_vad import collect_chunks, load_silero_vad, get_speech_timestamps
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def vad_detected(path) -> bool:
    """Check if the audio file has speech segments using VAD."""
    waveform = load_audio(path, return_type="torchaudio")
    vad = load_silero_vad()
    stamps = get_speech_timestamps(
        waveform, vad, sampling_rate=PROJECT_SAMPLING_RATE
    )
    return len(stamps) > 0

class PronunciationDataset(Dataset):
    def __init__(
        self,
        data: pl.DataFrame,
        sample_rate=PROJECT_SAMPLING_RATE,
        n_mels=128,
    ):
    
        # Cleaning with VAD
        # filter_mask = pl.Series(
        #     [vad_detected(os.path.join(base_dir, path)) for path in data["rec_path"].to_list()]
        # )
        # self.data = data.filter(filter_mask)
        self.data = data
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

    def __len__(self):
        return self.data.height

    def __getitem__(self, idx):
        row = self.data.row(idx)
        rec_path = row[3]
        label = row[1]

        waveform = load_audio(rec_path, return_type="torchaudio")
        stamps = get_speech_timestamps(
            waveform, self.vad, sampling_rate=PROJECT_SAMPLING_RATE, min_speech_duration_ms=100
        )
        trimmed_waveform = waveform
        if len(stamps) != 0:
            trimmed_waveform = collect_chunks(stamps, waveform)
        else:
            print(rec_path, "has no speech segments, using full waveform")
        
        trimmed_waveform = trimmed_waveform.reshape((1, -1))
        # Log-Mel features
        mel_spec = self.mel_spectrogram(trimmed_waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (
            log_mel_spec.std() + 1e-6
        )

        return log_mel_spec, torch.tensor(label, dtype=torch.float32)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # adjust input size after pooling
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B,16,H/2,W/2]
        x = self.pool(F.relu(self.conv2(x)))  # [B,32,H/4,W/4]
        x = self.pool(F.relu(self.conv3(x)))  # [B,64,H/8,W/8]
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return torch.sigmoid(x).squeeze(1)

    # Training loop


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for *specs, labels in dataloader:
        specs = (spec.to(device) for spec in specs)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(*specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)

        preds = (outputs >= 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for *specs, labels in dataloader:
            specs = (spec.to(device) for spec in specs)
            labels = labels.to(device)
            outputs = model(*specs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            preds = (outputs >= 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples
    return epoch_loss, epoch_acc
