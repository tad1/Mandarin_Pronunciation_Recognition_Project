import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model


# Dataset
class WavDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.data = dataframe.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filepath = row["filepath"]
        waveform, sample_rate = torchaudio.load(filepath)

        # Mono and resample
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)

        input_values = self.processor(
            waveform.squeeze(), sampling_rate=16000, return_tensors="pt"
        ).input_values[0]

        pron = torch.tensor(row["pron"]).long()
        tone = (
            torch.tensor(row["tone"]).long()
            if row["pron"] == 1
            else torch.tensor(-100).long()
        )

        return {"input_values": input_values, "pron": pron, "tone": tone}


# Collate function to pad batches
def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    pron = torch.stack([item["pron"] for item in batch])
    tone = torch.stack([item["tone"] for item in batch])

    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)

    attention_mask = (input_values != 0).long()

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "pron": pron,
        "tone": tone,
    }


# Model with classification heads
class Wav2VecForPronTone(nn.Module):
    def __init__(self, base_model_name="facebook/wav2vec2-base", num_tones=5):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(base_model_name)
        hidden_size = self.wav2vec.config.hidden_size
        self.pron_classifier = nn.Linear(hidden_size, 2)
        self.tone_classifier = nn.Linear(hidden_size, num_tones)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        pron_logits = self.pron_classifier(pooled)
        tone_logits = self.tone_classifier(pooled)
        return pron_logits, tone_logits


# Training setup
def train(model, dataloader, optimizer, device):
    model.train()
    pron_loss_fn = nn.CrossEntropyLoss()
    tone_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0

    for batch in dataloader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pron_labels = batch["pron"].to(device)
        tone_labels = batch["tone"].to(device)

        optimizer.zero_grad()
        pron_logits, tone_logits = model(input_values, attention_mask=attention_mask)
        pron_loss = pron_loss_fn(pron_logits, pron_labels)
        tone_loss = tone_loss_fn(tone_logits, tone_labels)
        loss = pron_loss + tone_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
