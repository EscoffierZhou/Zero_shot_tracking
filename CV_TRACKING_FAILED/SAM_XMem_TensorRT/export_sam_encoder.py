import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam

# 1. 配置
checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
output_onnx = "sam_vit_b_encoder.onnx"

# 2. 加载模型
print("Loading SAM model...")
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cpu')
sam.eval()

# 3. 提取 Image Encoder
# SAM 的 encoder 输入是 (1, 3, 1024, 1024)
# 输出是 (1, 256, 64, 64)
class SamEncoder(nn.Module):
    def __init__(self, sam_model: Sam):
        super().__init__()
        self.image_encoder = sam_model.image_encoder

    def forward(self, x):
        return self.image_encoder(x)

encoder = SamEncoder(sam)

# 4. 导出 ONNX
print("Exporting to ONNX...")
dummy_input = torch.randn(1, 3, 1024, 1024)

# 动态轴设置（虽然 SAM 通常是固定尺寸，但为了保险）
dynamic_axes = {
    'input_image': {0: 'batch_size'},
    'image_embeddings': {0: 'batch_size'}
}

torch.onnx.export(
    encoder,
    dummy_input,
    output_onnx,
    export_params=True,
    opset_version=18, # TensorRT 10 喜欢较新的 opset
    do_constant_folding=True,
    input_names=['input_image'],
    output_names=['image_embeddings'],
    dynamic_axes=dynamic_axes
)

print(f"Done! Saved to {output_onnx}")

# 5. 简化 ONNX (可选，推荐)
try:
    import onnx
    from onnxsim import simplify
    print("Simplifying ONNX...")
    onnx_model = onnx.load(output_onnx)
    model_simp, check = simplify(onnx_model)
    if check:
        onnx.save(model_simp, output_onnx)
        print("ONNX simplified successfully.")
    else:
        print("ONNX simplification failed.")
except ImportError:
    print("onnxsim not installed, skipping simplification.")
