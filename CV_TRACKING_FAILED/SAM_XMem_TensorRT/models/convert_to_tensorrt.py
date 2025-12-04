import torch
from torch_tensorrt import compile

def convert_sam_to_trt():
    # 加载 PyTorch 模型
    sam = torch.jit.load("sam_vit_b_01ec64.ts")

    # 编译为 TensorRT
    trt_model = compile(
        sam,
        inputs=[torch.randn((1, 3, 512, 512)).cuda()],  # 示例尺寸
        enabled_precisions={torch.half, torch.float},
        workspace_size=1 << 30,
        truncate_long_and_double=True,
    )

    torch.jit.save(trt_model, "sam_encoder.ts")

if __name__ == "__main__":
    convert_sam_to_trt()