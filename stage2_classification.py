import torch
import torch.nn as nn
from torchvision import models
import timm

class SwinFeatureExtractor(nn.Module):
    """
    The Hybrid Head Module: 
    Replaces the standard ResNet Layer4 with a Swin Transformer Block 
    to capture global semantic context.
    """
    def __init__(self, in_channels=256, embed_dim=256, img_res=14, num_classes=7):
        super().__init__()
        
        # 1. 1x1 Conv Projection to align channel dimensions
        self.in_projection = nn.Conv2d(in_channels, embed_dim, kernel_size=1) 
        
        # 2. Swin Transformer Block (using timm implementation)
        self.swin = timm.models.swin_transformer.SwinTransformerBlock(
            dim=embed_dim,
            input_resolution=(img_res, img_res),
            num_heads=8,
            window_size=7,
            shift_size=0,  # Standard window attention (no shift for single block)
            mlp_ratio=4.,
            qkv_bias=True,
            drop_path=0.1
        )
        
        # 3. Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # Input shape from ResNet Layer3: [B, 256, 14, 14]
        B, C, H, W = x.shape
        
        # Align channels
        x = self.in_projection(x) 
        
        # Reshape for Swin Transformer: [B, H*W, C]
        x = x.flatten(2).transpose(1, 2) 
        
        # Apply Self-Attention
        x, _ = self.swin(x)
        
        # Reshape back to spatial tensor: [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        
        # Global Average Pooling & Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.final_fc(x)

class RTDS_Stage2_Classifier(nn.Module):
    """
    Full Stage 2 Model: ResNet-34 Backbone (Layer0-3) + Swin-Hybrid Head
    """
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        
        # Load ImageNet-pretrained ResNet-34
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.resnet34(weights=weights)
        
        # Extract the CNN Trunk (Stem + Layer1 + Layer2 + Layer3)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        
        # Note: We discard layer4 and fc
        
        # Append the Swin-Hybrid Head
        # ResNet34 Layer3 outputs 256 channels with 14x14 resolution (for 224x224 input)
        self.hybrid_head = SwinFeatureExtractor(
            in_channels=256, 
            embed_dim=256, 
            img_res=14, 
            num_classes=num_classes
        )
        
    def forward(self, x):
        # 1. Local Feature Extraction (CNN Trunk)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 2. Global Context Modeling & Classification (Hybrid Head)
        x = self.hybrid_head(x)
        return x