import torch
import torch.nn as nn
import onnx

class SimpleModel(nn.Module):
    def forward(self, x):
        return x * 2

model = SimpleModel()
dummy_input = torch.randn(1, 3, 224, 224)

print("Exporting simple model...")
torch.onnx.export(model, dummy_input, "simple.onnx", verbose=True)
print("Export complete.")
