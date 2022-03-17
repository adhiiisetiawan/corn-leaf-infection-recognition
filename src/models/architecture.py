from torch import nn
from torchvision.models import mobilenet_v3_large

class CornLeafClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = mobilenet_v3_large(pretrained=True)
        self.freeze()
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(960, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 2)
        )
        
    def forward(self, x):
        return self.mobilenet(x)
    
    def freeze(self):
        for param in self.mobilenet.parameters():
            param.requires_grad = False
        
    def unfreeze(self):
        for param in self.mobilenet.parameters():
            param.requires_grad = True