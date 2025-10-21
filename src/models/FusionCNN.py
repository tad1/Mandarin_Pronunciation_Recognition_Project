from torch import nn
import torch

class FusionCNN(nn.Module):
    """
    Simplified model with focus on generalization
    """

    def __init__(self, n_2d_channels, n_1d_channels, num_classes=1, dropout_rate=0.5):
        super(FusionCNN, self).__init__()

        self.features2d = nn.Sequential(
            # Block 1
            nn.Conv2d(n_2d_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            # Global pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.features1d = nn.Sequential(
            # Block 1
            nn.Conv1d(n_1d_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            # Block 2
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            # Block 3
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            # Global pooling
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(96+64+8, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )

    def forward(self, x_2d, x_1d):
        x_2d = self.features2d(x_2d)
        x_2d = x_2d.view(x_2d.size(0), -1)
        x_1d = self.features1d(x_1d)
        x_1d = x_1d.view(x_1d.size(0), -1)
        x = torch.cat((x_2d, x_1d), dim=1)
        x = self.classifier(x)
        return torch.sigmoid(x).squeeze(1)


class ContextFusionCNN(nn.Module):
    """
    Simplified model with focus on generalization
    """

    def __init__(self, n_2d_channels, n_1d_channels, num_classes=1, num_words=30, dropout_rate=0.5):
        super(ContextFusionCNN, self).__init__()

        self.features2d = nn.Sequential(
            # Block 1
            nn.Conv2d(n_2d_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            # Global pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.features1d = nn.Sequential(
            # Block 1
            nn.Conv1d(n_1d_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            # Block 2
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            # Block 3
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            # Global pooling
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.word_embedding = nn.Embedding(num_words,16)

        self.classifier = nn.Sequential(
            nn.Linear(96+64+16, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )

    def forward(self, x_2d, x_1d, x_word):
        x_2d = self.features2d(x_2d)
        x_2d = x_2d.view(x_2d.size(0), -1)
        x_1d = self.features1d(x_1d)
        x_1d = x_1d.view(x_1d.size(0), -1)
        x_word = self.word_embedding(x_word)
        x = torch.cat((x_2d, x_1d, x_word), dim=1)
        x = self.classifier(x)
        return torch.sigmoid(x).squeeze(1)