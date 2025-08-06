import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from audio import load_audio, PROJECT_SAMPLING_RATE
import polars as pl
import os
import torch.nn.functional as F
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class PronunciationDataset(Dataset):
    def __init__(
        self,
        data: pl.DataFrame,
        base_dir,
        sample_rate=PROJECT_SAMPLING_RATE,
        n_mels=128,
    ):
        self.data = data
        self.base_dir = base_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=512,
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def __len__(self):
        return self.data.height

    def __getitem__(self, idx):
        row = self.data.row(idx)
        rec_path = row[3]
        label = row[1]
        full_path = os.path.normpath(os.path.join(self.base_dir, rec_path))

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Audio file not found: {full_path}")

        waveform = load_audio(full_path, return_type="torchaudio").reshape(1, -1)

        # Voice Activity Detection
        trimmed_waveform = torchaudio.functional.vad(
            waveform, sample_rate=self.sample_rate
        )
        if trimmed_waveform.numel() == 0:
            trimmed_waveform = waveform  # fallback to original

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

        # Add batch normalization
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # Increase dropout rates
        self.dropout1 = nn.Dropout2d(0.15)  # For conv layers
        self.dropout2 = nn.Dropout(0.2)  # For fc layers
        self.dropout3 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.dropout1(self.pool(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout1(self.pool(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout1(self.pool(F.relu(self.bn3(self.conv3(x)))))
        x = x.view(x.size(0), -1)
        x = self.dropout2(F.relu(self.fc1(x)))
        x = self.dropout3(x)
        x = self.fc2(x)
        return torch.sigmoid(x).squeeze(1)

    # Training loop


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for specs, labels in dataloader:
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * specs.size(0)

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
        for specs, labels in dataloader:
            specs, labels = specs.to(device), labels.to(device)
            outputs = model(specs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * specs.size(0)

            preds = (outputs >= 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples
    return epoch_loss, epoch_acc
