import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.is_log_mode = False
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.BatchNorm2d(16),  # Add BatchNorm2d after Conv2d
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 256),
            nn.BatchNorm1d(256),  # Add BatchNorm1d after Linear layer
            nn.LeakyReLU(inplace=True),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),  # Add BatchNorm1d after Linear layer
            
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),  # Add BatchNorm1d after Linear layer
            nn.LeakyReLU(inplace=True),

            nn.Linear(256, 16 * 16 * 16),
            nn.LeakyReLU(inplace=True),

            nn.Unflatten(1, (16, 16, 16)),  # Match the shape after last Conv2d in encoder
            
            nn.ConvTranspose2d(16, 1, 3),
            nn.Upsample(scale_factor= 34/18, mode='nearest'),
    
            nn.Sigmoid()
        )
                
    def forward(self, x):
    
        encoded = self.encoder(x)
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
