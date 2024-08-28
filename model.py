import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorizationModel(nn.Module):
    def __init__(self):
        super(ColorizationModel, self).__init__()
        # Encoder: feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 13 * 13, 1024)  # Assuming input image size is 100x100
        self.fc2 = nn.Linear(1024, 4096)  # Increasing dimensionality for decoding
        self.fc3 = nn.Linear(4096, 32 * 25 * 25)  # Final output shape to match decoder input

        # Decoder: upsampling and reconstruction
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        
        # Flatten the output from the encoder
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Reshape for the decoder
        x = x.view(x.size(0), 32, 25, 25)
        
        # Decoder
        x = self.decoder(x)
        return x
