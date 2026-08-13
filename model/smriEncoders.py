import torch
import torch.nn as nn

class SMRIEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.net(x)  