from torch.utils.mobile_optimizer import optimize_for_mobile
import torch
from models import Mobilenet_v3_large

model = Mobilenet_v3_large()
model.conv_4 = torch.nn.Sequential(
    torch.nn.Conv2d(1280, 7, 1),
    torch.nn.Softmax()
)
model.load_state_dict(torch.load("trained_models/ham10k_trained.pth"))
model.eval()
    
X = torch.distributions.uniform.Uniform(-10000, 10000).sample((1, 3, 250, 300))
    
traced_script_module = torch.jit.trace(model, X)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    
traced_script_module_optimized.save("trained_models/ham10k_optimized.pt")