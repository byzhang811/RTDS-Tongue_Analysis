import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp

# ================= 核心配置区域 =================
# 1. 你的 SCC 绝对路径
IMAGES_DIR = "/projectnb/vipcnns/Boyang_Clutter/Swin_CNN/segmentation_dataset_final/segment_datasets/original"
MASKS_DIR  = "/projectnb/vipcnns/Boyang_Clutter/Swin_CNN/segmentation_dataset_final/segment_datasets/mask"

# 2. 训练参数
MODEL_NAME = "U-Net++ (ResNet34 Backbone)"
SAVE_NAME = "unet_plusplus_best.pth" # 保存的模型文件名
BATCH_SIZE = 16       # 如果爆显存改成 8
LR = 0.0001           # 学习率
EPOCHS = 30           # 30 轮足够收敛
IMG_SIZE = 256        # 统一输入大小
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==============================================

class TongueDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        
        # 过滤隐藏文件并排序，确保图片和Mask一一对应
        self.images = sorted([f for f in os.listdir(img_dir) if not f.startswith('.')])
        self.masks = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])
        
        # 简单检查
        if len(self.images) != len(self.masks):
            print(f"⚠️ 警告: 图片数量 ({len(self.images)}) 与 Mask 数量 ({len(self.masks)}) 不一致！")
            # 截断到较短的长度，防止报错
            min_len = min(len(self.images), len(self.masks))
            self.images = self.images[:min_len]
            self.masks = self.masks[:min_len]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # 1. 读取图片 (RGB)
        image = Image.open(img_path).convert("RGB")
        # 2. 读取 Mask (转灰度)
        mask = Image.open(mask_path).convert("L")

        # 3. 预处理：统一 Resize
        # 使用双线性插值缩放图片，邻近插值缩放 Mask (防止引入虚假像素值)
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST)

        # 4. 转 Numpy 并归一化
        image_np = np.array(image)
        mask_np = np.array(mask)

        # Mask 二值化处理 (确保背景是0，前景是1)
        # 假设白色(255)是前景，或者非黑即前景
        mask_np = (mask_np > 100).astype(np.float32)
        mask_np = np.expand_dims(mask_np, axis=0) # 增加通道维度 [1, H, W]

        # 5. 转 Tensor
        transform = transforms.ToTensor()
        image_tensor = transform(image_np) # 会自动归一化到 [0, 1]
        mask_tensor = torch.from_numpy(mask_np)

        return image_tensor, mask_tensor

def main():
    print(f"🚀 启动 SOTA 分割训练任务...")
    print(f"数据集路径: {IMAGES_DIR}")
    print(f"模型架构: {MODEL_NAME}")

    # --- 1. 准备数据 ---
    full_dataset = TongueDataset(IMAGES_DIR, MASKS_DIR)
    print(f"成功加载数据: {len(full_dataset)} 对")

    # 9:1 划分训练验证集
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- 2. 定义 SOTA 模型 (U-Net++) ---
    # 使用 ResNet34 作为编码器，加载 ImageNet 预训练权重
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )
    model.to(DEVICE)

    # --- 3. 优化器与损失函数 ---
    # DiceLoss + BCE Loss 组合，是分割任务的最佳拍档
    loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # 学习率调整策略 (CosineAnnealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # --- 4. 训练循环 ---
    best_iou = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # 更新学习率
        scheduler.step()

        # --- 验证阶段 ---
        model.eval()
        val_iou_score = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                logits = model(images)
                
                # 计算 IoU (Intersection over Union)
                pred_mask = (logits.sigmoid() > 0.5).long()
                true_mask = masks.long()
                
                # smp 自带的 IoU 计算工具
                tp, fp, fn, tn = smp.metrics.get_stats(pred_mask, true_mask, mode="binary", threshold=0.5)
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                val_iou_score += iou.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_iou = val_iou_score / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_train_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

        # 保存最佳模型
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), SAVE_NAME)
            print(f"  🌟 New Best Model Saved! IoU: {best_iou:.4f}")

    print("-" * 30)
    print(f"✅ 训练完成！最佳 IoU: {best_iou:.4f}")
    print(f"模型已保存为: {os.path.abspath(SAVE_NAME)}")
    print("下一步：使用 clean_data_with_unet.py 加载这个模型来清洗你的分类数据。")

if __name__ == '__main__':
    main()