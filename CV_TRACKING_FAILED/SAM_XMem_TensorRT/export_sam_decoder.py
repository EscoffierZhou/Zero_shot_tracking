import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam
import warnings

# 忽略一些无关紧要的警告
warnings.filterwarnings("ignore")

checkpoint = "sam_vit_b_encoder.pth" # 其实还是原来那个文件，只是我们这次只取decoder
model_type = "vit_b"
output_onnx = "sam_vit_b_decoder.onnx"

print("Loading SAM model...")
# 注意：这里我们还是加载整个模型，但只导出 decoder 部分
sam = sam_model_registry[model_type](checkpoint="sam_vit_b_01ec64.pth")
sam.to(device='cpu')
sam.eval()

class SamDecoder(nn.Module):
    def __init__(self, sam_model: Sam):
        super().__init__()
        self.mask_decoder = sam_model.mask_decoder
        self.model = sam_model

    def forward(self, image_embeddings, point_coords, point_labels):
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return low_res_masks, iou_predictions

decoder = SamDecoder(sam)

print("Exporting Decoder to ONNX...")


# 设定最大支持的点数 (20个点足够你点满屏幕了)
MAX_POINTS = 20

# 构造固定维度的输入
embed_dim = 256
embed_size = 64

# 1. 图像特征 (固定)
image_embeddings = torch.randn(1, embed_dim, embed_size, embed_size)

# 2. 点坐标 (Batch=1, Points=20, 2) -> 这里的 20 是写死的
point_coords = torch.zeros(1, MAX_POINTS, 2, dtype=torch.float)
# 随便填点数据防止全是0出问题，实际推理时用真实数据
point_coords[0, :5, :] = torch.randint(0, 1024, (5, 2)).float()

# 3. 点标签 (Batch=1, Points=20)
point_labels = torch.zeros(1, MAX_POINTS, dtype=torch.float)
# 0代表背景/填充点，1代表前景点。这里前5个设为1，后面全是0
point_labels[0, :5] = 1.0

print(f"Exporting Decoder with FIXED shapes (Max Points={MAX_POINTS})...")

torch.onnx.export(
    decoder,
    (image_embeddings, point_coords, point_labels),
    output_onnx,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['image_embeddings', 'point_coords', 'point_labels'],
    output_names=['low_res_masks', 'iou_predictions'],
    # ❌ 彻底删掉 dynamic_axes 参数！
    # dynamic_axes=dynamic_axes,
    dynamo=False
)

print(f"Done! Saved static model to {output_onnx}")