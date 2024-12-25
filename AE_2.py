import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.is_log_mode = False
        
        self.e1 = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.BatchNorm2d(16),  # Add BatchNorm2d after Conv2d
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.e2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),  # Add BatchNorm2d after Conv2d
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
            
        self.e3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 512),
            nn.BatchNorm1d(512),  # Add BatchNorm1d after Linear layer
            nn.LeakyReLU(inplace=True)
        )
        
        self.e4 = nn.Sequential(
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),  # Add BatchNorm1d after Linear layer
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(64, 512),
            nn.BatchNorm1d(512),  # Add BatchNorm1d after Linear layer
            nn.LeakyReLU(inplace=True),

            nn.Linear(512, 32 * 7 * 7),
            nn.LeakyReLU(inplace=True),

            nn.Unflatten(1, (32, 7, 7)),  # Match the shape after last Conv2d in encoder
            
            nn.ConvTranspose2d(32, 16, 3),
            nn.BatchNorm2d(16),  # Add BatchNorm2d after ConvTranspose2d
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.ConvTranspose2d(16, 1, 3),
            nn.Sigmoid()
        )
        
        self.encoder = [self.e1, self.e2, self.e3, self.e4]
        
    def forward(self, x):
        for e in self.encoder:
            x = e(x)
            if self.is_log_mode:
                print(x)
            
        encoded = x
        decoded = self.decoder(encoded)
        
        return decoded, encoded
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def log_mode(self, val):
        assert type(val) == bool
        self.is_log_mode = val
