from torchvision.models import mobilenet_v3_small
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
import torch

model = mobilenet_v3_small(weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1)

torch.save(model.state_dict(), "trained_models/MobileNet_V3_Small.pth")