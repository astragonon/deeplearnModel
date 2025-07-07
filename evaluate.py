import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from utils.helpers import initialize_model
import argparse
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model_path, model_name, method):
    device = torch.device("cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model, _ = initialize_model(model_name, method=method)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    test_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_set = CIFAR10(root='./datasets', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # 高效评估
    correct = 0
    total = 0
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    
    with torch.inference_mode():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for t, p in zip(labels, predicted):
                confusion_matrix[t, p] += 1
    
    # 打印结果
    print(f'\n测试准确率: {100 * correct / total:.2f}%')
    plot_confusion_matrix(confusion_matrix, classes, method)

def plot_confusion_matrix(cm, classes, method):
    """优化后的混淆矩阵绘制"""
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix ({method})')
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=8)
    plt.yticks(tick_marks, classes, fontsize=8)
    
    threshold = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > threshold:
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white", fontsize=6)
    
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{method}.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--model_name', type=str, required=True,
                       help='模型名称')
    parser.add_argument('--method', type=str, required=True,
                       choices=['head_only', 'lora', 'diff_lr'],
                       help='微调方法')
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.model_name, args.method)