import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp

# ================= 你的绝对路径配置 =================
# 1. 源数据路径 (你解压出来的那个)
source_root = "/projectnb/vipcnns/Boyang_Clutter/Swin_CNN/classification_dataset"

# 2. 输出路径 (清洗后的干净数据放这里)
target_root = "/projectnb/vipcnns/Boyang_Clutter/Swin_CNN/classification_dataset_clean"

# 3. 训练好的 U-Net++ 模型路径
model_path = "/projectnb/vipcnns/Boyang_Clutter/Swin_CNN/unet_plusplus_best.pth"

# 4. 参数配置
IMG_SIZE = 256   # 必须与训练分割模型时一致
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==================================================

def load_model():
    print(f"Loading U-Net++ from: {model_path}")
    # 定义模型结构 (必须和训练时完全一致)
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None, 
        in_channels=3,
        classes=1,
        activation=None
    )
    # 加载权重
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def process_and_save(model, img_path, save_path):
    try:
        # 1. 读取原图
        original_img = Image.open(img_path).convert('RGB')
        w, h = original_img.size
        
        # 2. 预处理
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])
        input_tensor = transform(original_img).unsqueeze(0).to(DEVICE)
        
        # 3. 推理
        with torch.no_grad():
            logits = model(input_tensor)
            mask = (logits.sigmoid() > 0.5).float()
            
        # 4. 后处理 (Mask 还原回原图尺寸)
        mask_pil = transforms.ToPILImage()(mask.squeeze(0).cpu())
        mask_pil = mask_pil.resize((w, h), Image.Resampling.NEAREST)
        mask_tensor = transforms.ToTensor()(mask_pil)
        
        # 原图转 Tensor
        orig_tensor = transforms.ToTensor()(original_img)
        
        # 5. 核心步骤：去背景 (广播相乘)
        clean_tensor = orig_tensor * mask_tensor
        
        # 6. 保存
        clean_img = transforms.ToPILImage()(clean_tensor)
        clean_img.save(save_path)
        return True
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def main():
    if not os.path.exists(source_root):
        print(f"❌ 错误: 找不到源路径 {source_root}")
        return
        
    # 创建目标根目录
    os.makedirs(target_root, exist_ok=True)
    
    # 加载模型
    model = load_model()
    print("✅ 模型加载完毕，开始清洗...")
    
    total_count = 0
    
    # 遍历所有文件夹
    for root, dirs, files in os.walk(source_root):
        # 跳过隐藏文件夹
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
                src_path = os.path.join(root, file)
                
                # 计算相对路径，保持目录结构 (例如 "1/image_01.jpg")
                rel_path = os.path.relpath(src_path, source_root)
                dst_path = os.path.join(target_root, rel_path)
                
                # 确保目标子文件夹存在
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                
                # 处理
                if process_and_save(model, src_path, dst_path):
                    total_count += 1
                    if total_count % 100 == 0:
                        print(f"已清洗 {total_count} 张...")

    print("-" * 40)
    print(f"🎉 全部完成！共清洗 {total_count} 张图片。")
    print(f"干净数据已保存在: {target_root}")
    print("现在，那里的图片应该是【黑底彩色】的了。")

if __name__ == '__main__':
    main()