# 机器学习与深度学习大作业 - 模型微调技术探索

## 实验环境

本实验使用以下环境和依赖：

- 操作系统:Windows 11
- Python版本: 3.9
- 依赖库:
  - `torch`
  - `torchvision`
  - `transformers`
  - `matplotlib`
  - `numpy`
  - `pandas`

## 数据集下载

使用cifar-10数据集
规模：类别数10，训练样本50000，测试样本10000，图像尺寸24*24

## 运行方式

### 1. 克隆代码仓库

首先，克隆此项目的 Git 仓库：

```bash
git clone https://github.com/astragonon/deeplearnModel.git
```

### 2. 训练模型

运行以下命令开始训练模型：

```bash
# 1. 仅分类头微调
python train.py --model resnet18 --method head_only --epochs 30 --batch_size 128

# 2. LoRA微调
python train.py --model resnet18 --method lora --rank 8 --epochs 30

# 3. 差分学习率微调
python train.py --model resnet18 --method diff_lr --epochs 30
```

### 3. 测试模型

模型训练完成后，运行以下命令进行评估：

```bash
python evaluate.py --model_path best_diff_lr_model.pth --model_name resnet18 --method diff_lr

python evaluate.py --model_path best_head_only_model.pth --model_name resnet18 --method head_only

```

### 4. 可视化结果

若要可视化训练过程中的损失和准确率，运行：

```bash
python visualize.py
```

## 实验结果

以下是实验结果：

### 图像分类实验

| 微调方法       | 准确率 (%)  | 训练时间 (小时) |
| -------------- | ----------- | --------------- |
| 仅分类头微调   |  90.74      | 9               |
| LoRA微调       |  91.02      | 12              |
| 差分学习率微调 |  92.94      | 14              |

实验结果表明
（1）差分学习率的准确率最高，为92.94%，但消耗参数和训练时间最长，训练代价较高。
（2）Head_only方法准确率为90.74%，训练时间最低。
（3）LoRA的效率最高，相比于head_only方法，以较少的参数和训练时间为代价，换取了00.28%的准确率提升。
