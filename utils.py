import torch
import torch.nn as nn
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none') 

    def forward(self, input, target):
        logpt = -self.ce_loss(input, target) 
        pt = torch.exp(logpt)                
        focal_term = (1 - pt) ** self.gamma
        loss = -self.alpha * focal_term * logpt
        return loss.mean() if self.reduction == 'mean' else loss.sum()