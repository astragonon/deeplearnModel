import torch
import matplotlib.pyplot as plt

methods = ['lora', 'diff_lr', 'head_only']
plt.figure(figsize=(12, 5))

# 绘制准确率曲线
plt.subplot(1, 2, 1)
for method in methods:
    stats = torch.load(f'training_stats_{method}.pt')
    val_acc = [x['accuracy'] for x in stats if x['phase']=='val']
    plt.plot(val_acc, label=method)
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.legend()

# 绘制损失曲线
plt.subplot(1, 2, 2)
for method in methods:
    stats = torch.load(f'training_stats_{method}.pt')
    train_loss = [x['loss'] for x in stats if x['phase']=='train']
    plt.plot(train_loss, label=method)
plt.title('Training Loss') 
plt.xlabel('Epoch')

plt.tight_layout()
plt.savefig('comparison.png')
plt.show()