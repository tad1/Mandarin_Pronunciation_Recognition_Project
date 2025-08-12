import torch.nn as nn


class SimplifiedLightweightCNN(nn.Module):
    """
    Simplified model with focus on generalization
    """

    def __init__(self, input_channels=1, num_classes=1, dropout_rate=0.5):
        super(SimplifiedLightweightCNN, self).__init__()

        # First block - capture fine-grained features
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(5, 3), padding=(2, 1)),  # Asymmetric kernels for freq/time
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),  # Second conv in block
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.1)
        )

        # Second block - mid-level features
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.15)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15)
        )

        # Fourth block - high-level features
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.2)
        )

        # Attention mechanism for important feature focusing
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 128, 1),
            nn.SiLU(),
            nn.Conv2d(128, 256, 1),
            nn.Sigmoid()
        )

        # Global pooling and norm
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(256)

        # Enhanced classifier with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Feature extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights

        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.norm(x)
        x = self.classifier(x)

        return x.squeeze(1)
