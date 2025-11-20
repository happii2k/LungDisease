import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, dropout_rate=0.3):
        """
        Optimized CNN architecture for Pneumonia X-ray classification
        Designed for binary classification: Normal vs Pneumonia
        """
        super(Net, self).__init__()
        
        # Block 1: Initial feature extraction
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Block 2: Deeper features
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Block 3: High-level features
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Block 4: Abstract features
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Global Average Pooling (replaces flatten + dense)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate * 1.5),  # Higher dropout before classification
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)  # 2 classes: Normal, Pneumonia
        )
    
    def forward(self, x):
        # Feature extraction
        x = self.conv_block1(x)  # -> [batch, 32, H/2, W/2]
        x = self.conv_block2(x)  # -> [batch, 64, H/4, W/4]
        x = self.conv_block3(x)  # -> [batch, 128, H/8, W/8]
        x = self.conv_block4(x)  # -> [batch, 256, H/16, W/16]
        
        # Global pooling
        x = self.global_avg_pool(x)  # -> [batch, 256, 1, 1]
        x = x.view(x.size(0), -1)    # -> [batch, 256]
        
        # Classification
        x = self.classifier(x)  # -> [batch, 2]
        
        return x  # Return logits (loss function will apply softmax)
