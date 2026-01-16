from torch import nn
import torch



class ContextGuesser(nn.Module):
    """
    Simplified model with focus on generalization
    """

    def __init__(self, num_classes=1, num_words=30, dropout_rate=0.5):
        super(ContextGuesser, self).__init__()

        self.word_embedding = nn.Embedding(num_words,16)

        self.classifier = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, num_classes)
        )

    def forward(self, x_word):
        x_word = self.word_embedding(x_word)
        x = self.classifier(x_word)
        return torch.sigmoid(x).squeeze(1)