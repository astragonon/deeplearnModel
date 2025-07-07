import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import get_cifar10_dataloaders
from utils.helpers import initialize_model, get_diff_lr_params
from tqdm import tqdm
import argparse
import os
import time
import psutil

# 设置CPU优化
torch.set_num_threads(4)

def log_resources():
    """记录系统资源使用情况"""
    mem = psutil.virtual_memory()
    print(f"内存使用: {mem.used/1024/1024:.1f}MB/{mem.total/1024/1024:.1f}MB")
    print(f"CPU使用率: {psutil.cpu_percent()}%")

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, method=None, device='cpu'):
    """优化后的训练函数"""
    writer = SummaryWriter(comment=f'_{method}')
    best_acc = 0.0
    training_stats = []
    accumulation_steps = 2  # 梯度累积步数
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        log_resources()
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                optimizer.zero_grad()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            start_time = time.time()
            
            for i, (inputs, labels) in enumerate(tqdm(dataloaders[phase], desc=phase)):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                        
                    if phase == 'train':
                        loss = loss / accumulation_steps
                        loss.backward()
                        
                        if (i+1) % accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                
                running_loss += loss.item() * inputs.size(0) * (accumulation_steps if phase == 'train' else 1)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_time = time.time() - start_time
            
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {epoch_time:.2f}s')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), f'best_{method}_model.pth')
            
            training_stats.append({
                'epoch': epoch,
                'phase': phase,
                'loss': epoch_loss,
                'accuracy': epoch_acc.item(),
                'time': epoch_time
            })
    
    writer.close()
    return model, training_stats

def main():
    parser = argparse.ArgumentParser(description='高效微调训练脚本')
    parser.add_argument('--model', type=str, default='resnet18', 
                       choices=['resnet18', 'mobilenetv3', 'efficientnet_b0'],
                       help='模型名称')
    parser.add_argument('--method', type=str, required=True,
                       choices=['head_only', 'lora', 'diff_lr'],
                       help='微调方法')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练epoch数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批量大小')
    parser.add_argument('--rank', type=int, default=4,
                       help='LoRA秩(仅lora方法)')
    args = parser.parse_args()
    
    device = torch.device("cpu")
    print(f"使用设备: {device}")
    print(f"配置: {args}")
    
    # 数据加载
    train_loader, test_loader, classes = get_cifar10_dataloaders(args.batch_size)
    dataloaders = {'train': train_loader, 'val': test_loader}
    
    # 模型初始化
    model, _ = initialize_model(
        args.model, 
        num_classes=len(classes), 
        method=args.method,
        reduction=args.reduction,
        rank=args.rank
    )
    model = model.to(device)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 优化器设置
    if args.method == 'diff_lr':
        lr_dict = {
            'conv1': 1e-5,
            'layer1': 1e-5,
            'layer2': 1e-4,
            'layer3': 1e-3,
            'head': 1e-2
        }
        params = get_diff_lr_params(model, lr_dict)
        optimizer = optim.AdamW(params)
    else:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    # 训练模型
    model, stats = train_model(
        model, 
        dataloaders, 
        criterion, 
        optimizer, 
        num_epochs=args.epochs,
        method=args.method,
        device=device
    )
    
    # 保存结果
    torch.save(model.state_dict(), f'final_{args.method}_model.pth')
    torch.save(stats, f'training_stats_{args.method}.pt')

if __name__ == '__main__':
    main()