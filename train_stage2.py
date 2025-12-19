import os
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np

# 引入我们刚才拆分好的模块
from classification_model import RTDS_Stage2_Classifier
from utils import FocalLoss, set_seed

# ================= 配置常量 =================
IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\n--- Epoch {epoch+1}/{num_epochs} ---')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 记录最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # 仅在演示时保存权重，避免占用空间
                torch.save(model.state_dict(), 'best_stage2_model.pth')

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:.4f}')

def main(args):
    # 1. 设置随机种子 (复现性关键)
    set_seed(42)

    # 2. 数据增强与加载
    # 注意：这里我们假设用户已经把数据按 ImageFolder 格式整理好了
    # 结构：root/train/class_names, root/val/class_names
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 隐私保护：如果找不到数据，打印提示信息而不是报错崩溃
    if not os.path.exists(args.data_dir):
        print(f"⚠️  Dataset not found at: {args.data_dir}")
        print("    (This is expected for the GitHub demo code. Real training requires private clinical data.)")
        print("    To run a dummy test, create a folder structure: ./dataset/train/class1 and put 1 image inside.")
        return

    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, 
                                 shuffle=(x=='train'), num_workers=4)
                   for x in ['train', 'val']}

    num_classes = len(image_datasets['train'].classes)
    print(f"✅ Data Loaded. Detected Classes: {num_classes}")

    # 3. 初始化 Stage 2 模型 (Swin-Hybrid)
    print("🚀 Initializing RTDS Stage 2 (Swin-Hybrid) Model...")
    model = RTDS_Stage2_Classifier(num_classes=num_classes)
    model = model.to(DEVICE)

    # 4. 定义 Loss (Focal Loss) 和 优化器
    # 你的论文核心点：Focal Loss 用于解决类别不平衡
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 5. 开始训练
    train_model(model, dataloaders, criterion, optimizer, scheduler, args.epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RTDS Stage 2 Training')
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Path to dataset root')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    main(args)
