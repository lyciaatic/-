# 手动实现深度神经网络

这个项目使用纯NumPy实现了一个深度学习框架，包括从基本组件到复杂模型的所有内容，用于MNIST手写数字识别任务。

## 项目结构

- `neural_network.py`: 包含基本网络组件（层、激活函数、优化器等）的实现
- `models.py`: 包含不同模型架构的实现（MLP、CNN、ResNet）
- `train.py`: 训练和测试的主程序

## 内容
1. MLP 通过列表layers定义各层神经元数量，其中有输入层、隐藏层和输出层
   ```bash
   class MLP(Model):
    """多层感知机模型"""
    def __init__(self, input_dim=784, hidden_dims=[256, 128], num_classes=10, use_dropout=True, dropout_rate=0.5):
        super().__init__()
        
        # 输入层 -> 第一个隐藏层
        self.layers.append(Linear(input_dim, hidden_dims[0], weight_decay=1e-4))
        self.layers.append(ReLU())
        if use_dropout:
            self.layers.append(Dropout(dropout_rate))
        
        # 隐藏层
        for i in range(len(hidden_dims) - 1):
            self.layers.append(Linear(hidden_dims[i], hidden_dims[i+1], weight_decay=1e-4))
            self.layers.append(ReLU())
            if use_dropout:
                self.layers.append(Dropout(dropout_rate))
        
        # 输出层
        self.layers.append(Linear(hidden_dims[-1], num_classes))
   ```
  -激活函数：隐藏层使用`LeakyReLU`，输出层使用`Softmax`将对数输出转换为类别概率
```bash
class LeakyReLU(Layer):
    """LeakyReLU激活函数"""
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.input = None
        self.optimizable = False
    
    def forward(self, inputs):
        self.input = inputs
        return np.maximum(self.alpha * inputs, inputs)
    
    def backward(self, grad):
        return grad * np.where(self.input > 0, 1, self.alpha)
```   
  -dropout正则化，训练过程中随机将部分神经元输出置0，防止过拟合，通过`dropout`参数控制失活概率
  ```bash
class Dropout(Layer):
    """Dropout层，用于防止过拟合"""
    def __init__(self, drop_rate=0.5):
        super().__init__()
        self.drop_rate = drop_rate
        self.mask = None
        self.training = True
        self.optimizable = False
```
2.卷积神经网络CNN：集成自定义的`Conv2D`层，后续连接MLP。模型包括：
  -Conv2D层：通过可学习的卷积核计算特征图，支持参数包括输入通道、输出通道、步长、填充。
  ```bash
class Conv2D(Layer):
    """二维卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, weight_decay=0, weight_decay_lambda=1e-4):
        super().__init__()
```
## 训练过程
1. 完成第二项优化算法：动量随机梯度下降，实现更新规则![image](https://github.com/user-attachments/assets/df191664-70c6-4cf8-9c34-ceacb1d5475b)固定动量系数$`\beta`$<sub>t</sub>=0.9，加速收敛减少振荡。训练中调整学习率$`\alpha`$<sub>t</sub>，平衡初始收敛速度和最终优化精度。
2. 完成第三项正则化：L2正则化。在损失函数中添加权重惩罚项以抑制过拟合。![image](https://github.com/user-attachments/assets/659e2d6c-fc39-4015-9bea-cbd3caa34565)为正则化强度，控制权重衰减程度。
3. 完成第四项损失函数：交叉熵损失。与输出层Softmax结合，计算预测概率和真实值的差异![image](https://github.com/user-attachments/assets/90a63cfe-3065-4ed2-84f6-eaf7cec9a02c)，其中$`y`$<sub>i,c</sub>为独热编码，$`p`$<sub>i,c</sub>为类别c的预测概率。

## 模型配置

 

## 特点

- 纯NumPy实现，不依赖任何深度学习框架
- 支持多种模型架构：
  - 多层感知机 (MLP)
  - 卷积神经网络 (CNN)
  - 残差网络 (ResNet)
- 包含多种层类型：
  - 线性层 (Linear)
  - 卷积层 (Conv2D)
  - 最大池化层 (MaxPool2D)
  - 批归一化层 (BatchNorm2D)
  - Dropout正则化
- 多种优化器：
  - SGD
  - 带动量的SGD
  - Adam
- 学习率调度
- 数据增强
- 模型保存和加载
- 权重可视化

## 使用方法

1. 确保数据集位于`./dataset/MNIST/`目录下
2. 运行训练脚本：

```bash
python train.py
```

3. 按照提示选择模型类型、是否使用数据增强和优化器类型
4. 训练完成后，将显示训练历史图表并在测试集上评估模型性能
5. 模型保存在`./saved_models/`目录下
6. 可视化图表保存在`./figs/`目录下

## 实验记录

- MLP模型: 测试集准确率约97%
- CNN模型: 测试集准确率约99%
- ResNet模型: 测试集准确率约99%

## 项目要求满足对应

本项目满足了以下要求：

1. 自己实现神经网络的核心功能：
   - 手动实现了前向传播和反向传播
   - 实现了关键层（线性层、卷积层等）
   - 实现了多种激活函数

2. 实现了多种复杂结构：
   - 多层感知机
   - 卷积神经网络
   - 带残差连接的网络

3. 实现了多种优化技术：
   - 权重衰减（L2正则化）
   - Dropout正则化
   - 批归一化

4. 支持多种训练策略：
   - 学习率调度
   - 不同的优化器
   - 数据增强 
