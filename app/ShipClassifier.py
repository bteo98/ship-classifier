import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ShipClassifier(nn.Module):
    def __init__(self, ntargets=5):
        super().__init__()
        self.backbone = EfficientNet.from_name('efficientnet-b2')
        self.fc = nn.Sequential(
            nn.LayerNorm(1408*7*7),
            nn.Dropout(0.2),
            nn.Linear(1408*7*7, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2),
            nn.Linear(64, ntargets)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.transforms = T.Compose([ 
            T.Resize((224, 224)),
            T.ToTensor(),
            # standard nomarlization 
            T.Normalize((0.485, 0.456, 0.406), 
                        (0.229, 0.224, 0.225))]
        )
    
    def forward(self, imgs):
        features = self.backbone.extract_features(imgs)
        # flatten
        features = features.reshape(features.size(0), -1)
        return self.fc(features)

    def predict_img(self, img):
        img = self.transforms(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        outputs = self(img)
        pred = self.softmax(outputs)
        prob, clss = pred.max(-1)
        return {
            'class': clss.item(),
            'probability': f'{prob.item():.5f}'
        }