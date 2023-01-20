from torch.utils.mobile_optimizer import optimize_for_mobile
import torch
from torchvision.models import mobilenet_v3_small
from torchvision.models import mobilenet_v3_large
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights 

class FeatureLayer(torch.nn.Module):
    def __init__(self, features, avgpool):
        super().__init__()
        self.features = features
        self.avgpool = avgpool
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

model = mobilenet_v3_small(weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(576, 1024),
    torch.nn.Hardswish(inplace = True),
    torch.nn.Dropout(p = 0.2, inplace = True),
    torch.nn.Linear(1024, 7),
)
model.load_state_dict(torch.load("trained_models/ham10k_trained.pth"))
model.eval()
feature_layer = FeatureLayer(model.features, model.avgpool)
classify_layer = model.classifier

# This doesn't work, need to find way to enable dropout on mobile
for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

X = torch.distributions.uniform.Uniform(-10000, 10000).sample((1, 3, 450, 600))
X_feature = feature_layer(X)

feature_traced_script = torch.jit.trace(feature_layer, X)
classifier_traced_script = torch.jit.trace(classify_layer, X_feature)

#feature_traced_script = optimize_for_mobile(feature_traced_script)
#classifier_traced_script = optimize_for_mobile(classifier_traced_script)
    
feature_traced_script.save("trained_models/ham10k_feature_mobile.pt")
classifier_traced_script.save("trained_models/ham10k_classify_mobile.pt")