import torch.nn as nn
import torchvision.models as models
import torch
from functools import partial


class MemoryEfficientLoRA(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4):
        super().__init__()
        self.lora_a = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_b = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.normal_(self.lora_a, mean=0, std=0.02)
        
    def forward(self, x):
        return x @ self.lora_a.T @ self.lora_b.T

def initialize_model(model_name, num_classes=10, method=None, **kwargs):
    """优化后的模型初始化"""
    model = None
    input_size = 112
 
    # 应用微调方法
    if method == 'head_only':
        model = apply_head_only(model)
    elif method == 'lora':
        model = apply_lora(model, kwargs.get('rank', 4))  # 默认rank4
    
    return model, input_size

def apply_head_only(model):
    # 先冻结所有
    for param in model.parameters():
        param.requires_grad = False
    
    # 部分解冻最后几层
    if hasattr(model, 'layer4'):  # ResNet
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif hasattr(model, 'features'):  # MobileNet
        for param in model.features[-4:].parameters():
            param.requires_grad = True
    
    # 解冻分类头
    if hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    return model

def apply_lora(model, rank=4):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            k in name for k in ['fc', 'classifier', 'attention']
        ):
            original_weight = module.weight
            module.weight = nn.Parameter(torch.zeros_like(original_weight))
            module.lora = MemoryEfficientLoRA(
                module.in_features, 
                module.out_features, 
                rank
            )
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False
    return model

def get_diff_lr_params(model, lr_dict):
    params = []
    default_lr = 1e-4
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        lr = default_lr
        # 根据层类型分配学习率
        if 'conv1' in name:
            lr = lr_dict.get('conv1', default_lr)
        elif 'layer1' in name:
            lr = lr_dict.get('layer1', default_lr)
        elif 'layer2' in name:
            lr = lr_dict.get('layer2', default_lr*10)
        elif 'layer3' in name or 'layer4' in name:
            lr = lr_dict.get('layer3', default_lr*100)
        elif 'fc' in name or 'classifier' in name:
            lr = lr_dict.get('head', default_lr*1000)
        
        params.append({'params': param, 'lr': lr})
    
    return params
