from torch.utils.mobile_optimizer import optimize_for_mobile
import torch
from torchvision.models import mobilenet_v3_small
from torchvision.models import mobilenet_v3_large
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights 

model = mobilenet_v3_small(weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model.classifier[-1] = torch.nn.Sequential(
    torch.nn.Linear(1024, 7),
    torch.nn.Softmax()
)
model.load_state_dict(torch.load("trained_models/ham10k_trained.pth"))
model.eval()

# This doesn't work, need to find way to enable dropout on mobile
for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

X = torch.distributions.uniform.Uniform(-10000, 10000).sample((1, 3, 450, 600))
    
traced_script_module = torch.jit.trace(model, X)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    
traced_script_module_optimized.save("trained_models/ham10k_optimized_dropout.pt")