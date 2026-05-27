from torchvision.models import resnet18, ResNet18_Weights
from torch import nn

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        for name, param in self.backbone.named_parameters():
            if "layer1" in name or "layer2" in name:
                param.requires_grad = False
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)