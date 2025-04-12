from PIL import Image
import numpy as np
import torch
import os

# Import FLAIR
from flair import FLAIRModel

# 设置要使用的GPU编号（0-3）
gpu_id = 2  # 这里设置您想要使用的GPU编号
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置预训练权重路径
weights_path = "/home/image/nvme/ZhouZhiLin/zhouzhilin/FoundationModel/FLAIR/flair/modeling/flair_pretrained_weights/flair_resnet.pth"

# Set model with pretrained weights
model = FLAIRModel(from_checkpoint=True, weights_path=weights_path)
model.to(device)  # Move model to GPU

# 打印模型加载后的显存使用情况
print(f"After model loading GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Load image and set target categories 
# (if the repo is not cloned, download the image and change the path!)
# image = np.array(Image.open("./documents/sample_macular_hole.png"))
image = np.array(Image.open("/home/image/nvme/ZhouZhiLin/zhouzhilin/FoundationModel/FLAIR/documents/normal_sample.png"))
text = ["normal", "healthy", "macular edema", "diabetic retinopathy", 
        "glaucoma", "macular hole", "lesion", "lesion in the macula"]

# Forward FLAIR model to compute similarities
with torch.cuda.amp.autocast():  # 使用自动混合精度
    probs, logits = model(image, text)


print("Image-Text similarities:")
print(logits.round(3)) # [[-0.32  -2.782  3.164  4.388  5.919  6.639  6.579 10.478]]
print("Probabilities:")
print(probs.round(3))
