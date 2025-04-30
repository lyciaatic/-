# Project 1 “神经网络与深度学习”项目报告

这个项目使用纯NumPy实现了一个深度学习框架，包括从基本组件到复杂模型的所有内容，用于MNIST手写数字识别任务。通过实现多层感知机（MLP）、卷积神经网络（CNN）和残差网络（ResNet）探索不同优化策略、正则化方法及网络结构对模型性能的影响。核心组包括卷积层、全连接层、BatchNorm、Dropout、动量优化器等。

## 项目结构
- `neural_network.py`: 包含基本网络组件（层、激活函数、优化器等）的实现
- `models.py`: 包含不同模型架构的实现（MLP、CNN、ResNet）
- `train.py`: 训练和测试的主程序
## 代码功能拆解说明
`models.py`
- 基类`Model`，提供前向传播、后向传播、加载参数、切换训练模式的接口
  ```bash
  class Model(ABC):
    """模型基类"""
    def __init__(self):
        self.layers = []
    
    @abstractmethod
    def forward(self, X):
        pass
    
    @abstractmethod
    def backward(self, grad):
        pass
    
    def __call__(self, X):
        return self.forward(X)
    
    def set_train(self, training=True):
        for layer in self.layers:
            if hasattr(layer, 'set_train'):
                layer.set_train(training)
    
    def save(self, file_path):
        params = []
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                layer_params = {
                    'name': layer.__class__.__name__,
                    'params': {k: v.copy() for k, v in layer.params.items()}
                }
                params.append(layer_params)
            else:
                params.append({'name': layer.__class__.__name__, 'params': None})
        
        with open(file_path, 'wb') as f:
            pickle.dump(params, f)  
    
    def load(self, file_path):
        """加载模型参数"""
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        
        param_idx = 0
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                if param_idx >= len(params):
                    raise ValueError(f"参数文件中的参数数量不足")
                
                layer_params = params[param_idx]
                if layer.__class__.__name__ != layer_params['name']:
                    raise ValueError(f"层类型不匹配: 期望 {layer.__class__.__name__}, 得到 {layer_params['name']}")
                
                for k, v in layer_params['params'].items():
                    if k in layer.params:
                        layer.params[k] = v.copy()
                        setattr(layer, k, layer.params[k])
                param_idx += 1
  ```
- 轻量卷积网络，包含两个步长为二的卷积层，用于MNIST分类
  ```bash
  class CNN(Model):
    """卷积神经网络模型）"""
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        super().__init__()
        
        # 提取输入形状
        in_channels, in_height, in_width = input_shape
        
        # 第一个卷积层 - 使用步长为2的卷积替代池化，减少操作次数
        self.layers.append(Conv2D(in_channels, 16, kernel_size=3, stride=2, padding=1))
        self.layers.append(ReLU())
        
        # 计算特征图尺寸
        h = in_height // 2
        w = in_width // 2
        
        # 第二个卷积层 - 直接使用步长为2的卷积，减少计算量
        self.layers.append(Conv2D(16, 32, kernel_size=3, stride=2, padding=1))
        self.layers.append(ReLU())
        
        # 计算特征图尺寸
        h = h // 2
        w = w // 2
        
        # 展平层
        self.layers.append(Flatten())
        
        # 输出层 - 直接从卷积到输出
        self.layers.append(Linear(32 * h * w, num_classes))
    
    def forward(self, X):
        """前向传播"""
        output = X
        for layer in self.layers:
            output = layer(output)
        return output
    
    def backward(self, grad):
        """反向传播"""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
```
- MLP多层感知机，配合后续动态隐藏层配置和正则化
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
    
    def forward(self, X):
        """前向传播"""
        # 确保输入是二维的
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        output = X
        for layer in self.layers:
            output = layer(output)
        return output
    
    def backward(self, grad):
        """反向传播"""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
```
- 参差块：实现残差连接，为ResNet提供基础模块
  ```bash
  
class ResidualBlock(Layer):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layers = []
        
        # 第一个卷积层
        self.layers.append(Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, weight_decay=1e-4))
        self.layers.append(BatchNorm2D(out_channels))
        self.layers.append(ReLU())
        
        # 第二个卷积层
        self.layers.append(Conv2D(out_channels, out_channels, kernel_size=3, padding=1, weight_decay=1e-4))
        self.layers.append(BatchNorm2D(out_channels))
        
        # 是否需要下采样
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = []
            self.shortcut.append(Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, weight_decay=1e-4))
            self.shortcut.append(BatchNorm2D(out_channels))
        
        # 激活函数
        self.activation = ReLU()
    
    def forward(self, X):
        # 主路径
        out = X
        for layer in self.layers:
            out = layer(out)
        
        # 短路径
        if self.shortcut is not None:
            shortcut = X
            for layer in self.shortcut:
                shortcut = layer(shortcut)
        else:
            shortcut = X
        
        # 合并主路径和短路径
        out += shortcut
        return self.activation(out)
    
    def backward(self, grad):
        # 反向传播经过激活函数
        grad = self.activation.backward(grad)
        
        # 保存初始梯度用于短路径
        shortcut_grad = grad.copy()
        
        # 主路径反向传播
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        
        # 短路径反向传播
        if self.shortcut is not None:
            shortcut_out = shortcut_grad
            for layer in reversed(self.shortcut):
                shortcut_out = layer.backward(shortcut_out)
            return grad + shortcut_out
        else:
            return grad + shortcut_grad
    
    def set_train(self, training=True):
        for layer in self.layers:
            if hasattr(layer, 'set_train'):
                layer.set_train(training)
        
        if self.shortcut is not None:
            for layer in self.shortcut:
                if hasattr(layer, 'set_train'):
                    layer.set_train(training)
```

## 核心功能实现
1. 网络结构的灵活配置：MLP 通过列表layers定义各层神经元数量，其中有输入层、隐藏层和输出层
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
   - 实验结果：不同隐藏层MLP对比；MLP-2(512+128隐藏层)验证集准确率更高为92.2%，MLP-1（256+128隐藏层）验证集准确率为91.8%。更多隐藏层提升模型能力，不过需要正则化避免过拟合。
2. 动量法优化训练
- 通过`MomentumSGD`类实现动量更新，显著加速收敛**对应问题2**
```bash
class MomentumSGD:
    """带动量的随机梯度下降优化器"""
    def __init__(self, model, learning_rate=0.01, momentum=0.9):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
        
        # 初始化速度
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                self.velocities[i] = {}
                for param_name in layer.params:
                    self.velocities[i][param_name] = np.zeros_like(layer.params[param_name])
    
    def step(self):
        """执行一次带动量的梯度更新"""
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads') and layer.grads is not None:
                for param_name in layer.params:
                    if layer.grads[param_name] is not None:
                        # 更新速度
                        self.velocities[i][param_name] = (
                            self.momentum * self.velocities[i][param_name] - 
                            self.learning_rate * layer.grads[param_name]
                        )
                        # 更新参数
                        layer.params[param_name] += self.velocities[i][param_name]
                        # 更新实际参数值
                        setattr(layer, param_name, layer.params[param_name])
```
   - 实验结果：动量系数$`\beta`$影响：$`\beta`$=0.9相比原始SGD（$`\beta`$=0）减少训练振荡，收敛速度提升15%；验证集准确率从90.6%提升至94.6%

3. L2正则化和dropout正则化，训练过程中随机将部分神经元输出置0，防止过拟合，通过`dropout`参数控制失活概率**对应问题3和问题1**
```bash
class Linear(Layer):
    """全连接层"""
    def __init__(self, in_dim, out_dim, weight_decay=0, weight_decay_lambda=1e-4):
        super().__init__()
        # Xavier初始化
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        self.W = np.random.normal(0, scale, (in_dim, out_dim))
        self.b = np.zeros((1, out_dim))
        
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
    
    def forward(self, inputs):
        self.input = inputs
        return np.dot(inputs, self.W) + self.b
    
    def backward(self, grad):
        # 计算参数梯度
        dW = np.dot(self.input.T, grad)
        if self.weight_decay > 0:
            dW += self.weight_decay_lambda * self.W
        db = np.sum(grad, axis=0, keepdims=True)
        
        # 计算输入梯度
        dX = np.dot(grad, self.W.T)
        
        # 保存梯度
        self.grads['W'] = dW
        self.grads['b'] = db
        
        return dX
```
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
`BatchNorm2D`对输入数据进行标准化
```bash
class BatchNorm2D(Layer):
    """批归一化层"""
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
```
   - 实验结果：
   - L2正则化：
        -  $`\lambda`$=0.001使验证准确率比无正则化高1.4%，缓解过拟合
        -  $`\lambda`$=0.01过大，导致欠拟合，准确率显著下降
   - Dropout中p的影响
        - p=0.5引入过多随机性，导致训练不稳定，准确率下降至86.6%，最优p值需要根据网络深度调整。
4. Softmax与交叉熵损失，**对应问题4**
```bash
class CrossEntropyLoss(Layer):
    """交叉熵损失函数"""
    def __init__(self, model=None):
        super().__init__()
        self.model = model
        self.y_pred = None
        self.y_true = None
        self.batch_size = None
        self.optimizable = False
    
    def forward(self, y_pred, y_true):
        self.batch_size = y_pred.shape[0]
        self.y_true = y_true
        
        # 应用softmax
        self.y_pred = softmax(y_pred)
        
        # 计算交叉熵损失
        eps = 1e-8  # 数值稳定性
        indices = np.arange(self.batch_size)
        loss = -np.sum(np.log(self.y_pred[indices, y_true] + eps)) / self.batch_size
        
        return loss
    
    def backward(self):
        # 初始化梯度
        grad = self.y_pred.copy()
        indices = np.arange(self.batch_size)
        
        # 设置真实类别处的梯度
        grad[indices, self.y_true] -= 1
        
        # 归一化梯度
        grad /= self.batch_size
        
        # 如果模型不为空，则传递梯度
        if self.model is not None:
            self.model.backward(grad)
        
        return grad
```
   - 实验结果：交叉熵和MSE对比显示，交叉熵准确率91.4%，MSE准确率90.8%。Softmax+交叉熵更适配多分类任务，梯度更新更高效     
5. 卷积神经网络CNN：集成自定义的`Conv2D`层，后续连接MLP。模型包括：
  - Conv2D层：通过可学习的卷积核计算特征图，支持参数包括输入通道、输出通道、步长、填充。
  - **对于问题5**
   
  ```bash
class Conv2D(Layer):
    """二维卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, weight_decay=0, weight_decay_lambda=1e-4):
        super().__init__()
……
def _im2col(self, X):
        """
        输入:
            X: (N, C, H, W)
        输出:
            cols: (N*out_h*out_w, C*k_h*k_w)
        """
        N, C, H, W = X.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
……
def forward(self, inputs):
        """前向传播"""
        self.input_shape = inputs.shape
        N = inputs.shape[0]
        
        # 展开输入
        self.X_cols, (N, out_h, out_w) = self._im2col(inputs)
        
        # 展开权重矩阵为二维矩阵
        W_row = self.W.reshape(self.out_channels, -1)
        
        # 矩阵乘法
        out = np.dot(self.X_cols, W_row.T) + self.b
        
        # 重塑为四维输出
        out = out.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        
        return out
    
    def backward(self, grad):
        """反向传播"""
        N = self.input_shape[0]
        
        # 重塑梯度为二维矩阵
        grad_reshaped = grad.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        
        # 计算偏置梯度
        db = np.sum(grad_reshaped, axis=0)
        
        # 计算权重梯度
        dW = np.dot(grad_reshaped.T, self.X_cols)
        dW = dW.reshape(self.W.shape)
        
        # 计算输入梯度
        W_row = self.W.reshape(self.out_channels, -1)
        dX_cols = np.dot(grad_reshaped, W_row)
        dX = self._col2im(dX_cols, self.input_shape)
        
        # 添加L2正则化
        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W
        
        # 保存参数梯度
        self.grads['W'] = dW
        self.grads['b'] = db
        
        return dX
```
   - 实验结果：CNN和MLP对比
    - CNN-1测试准确率达到93.4%，高于MLP-1
    - 自定义`Conv2D`层通过`im2col`技术比朴素循环实现快30%

   
## 训练过程
1. 完成第二项优化算法：动量随机梯度下降，实现更新规则![image](https://github.com/user-attachments/assets/df191664-70c6-4cf8-9c34-ceacb1d5475b)固定动量系数$`\beta`$<sub>t</sub>=0.9，加速收敛减少振荡。训练中调整学习率$`\alpha`$<sub>t</sub>，平衡初始收敛速度和最终优化精度。
2. 完成第三项正则化：L2正则化。在损失函数中添加权重惩罚项以抑制过拟合。![image](https://github.com/user-attachments/assets/659e2d6c-fc39-4015-9bea-cbd3caa34565)为正则化强度，控制权重衰减程度。
3. 完成第四项损失函数：交叉熵损失。与输出层Softmax结合，计算预测概率和真实值的差异![image](https://github.com/user-attachments/assets/90a63cfe-3065-4ed2-84f6-eaf7cec9a02c)，其中$`y`$<sub>i,c</sub>为独热编码，$`p`$<sub>i,c</sub>为类别c的预测概率。

## 模型配置
![image](https://github.com/user-attachments/assets/6d1fe435-dec7-4ff1-8719-51e3260e938a)
   - 数据预处理：
   1. 归一化：像素值除以255缩放至[0,1]
   2. 二值化：可选预处理，简化特征空间。
   - 训练参数
   1. 批量大小：32
   2. 训练轮次：50
   3. 学习率：0.01
   4. 验证频率：每五轮使用验证集评估一次

## 模型的实际训练和性能测试
- 卷积核权重可视化结果
   - `cnn_filters_layer1.png`提取CNN模型第一层卷积核
![cnn_filters_layer1](https://github.com/user-attachments/assets/0ef02e1a-cf44-46b6-bf89-ffcad942ef94)
   - `cnn_filters_layer2.png`提取CNN模型第二层卷积核，卷积层通过分层特征提取，逐步抽象图像信息![cnn_filters_layer2](https://github.com/user-attachments/assets/e9dcc6e2-5a14-4295-a220-20524954f3ea)
- CNN模型训练曲线
   - 由`cnn_training.png`图片可知，训练损失从0.35快速下降到0.05，验证损失同步下降，无过拟合,；训练准确率最后达98%，验证准确率达96%+；CNN模型收敛速度快，泛化能力良好![cnn_training](https://github.com/user-attachments/assets/e7843643-c95c-4728-931f-d65ed2323c9a)
- MLP和改进后曲线对比
   - `mlp_training.png`原始MLP训练损失震荡较大，验证准确率小于90%
   - `improved_mlp_training.png`增加Dropout和L2正则的改进MLP的损失平滑下降至0.1，验证准确率达到92%以上
     ![mlp_training](https://github.com/user-attachments/assets/eb5877ff-58cd-4dd6-a101-66fa6780902a)
     ![improved_mlp_training](https://github.com/user-attachments/assets/5ae77dac-0bd5-428e-b298-16ed759fcd36)
- 模型性能
   - 验证集表现表明CNN在特征提取和泛化能力上表现优于MLP
     1. `model_comparison_acc.png`准确率方面：CNN（98%）＞改进MLP（96%）＞MLP（92%）
![model_comparison_acc](https://github.com/user-attachments/assets/c5a4a518-dee4-4da0-b487-19ea8b1c4877)
     2. `model_comparison_loss.png`损失方面：CNN（5%）＜改进MLP（10%）＜MLP（30%）
![model_comparison_loss](https://github.com/user-attachments/assets/1e7ea417-5395-4e3e-bd8d-bdc33e7913e6)
- 测试结果
   - `test_accuracy_comparison.png``test_loss_comparison.png`分别是测试准确率和测试loss，可视化结果表明CNN在测试集表现是三者中最佳，改进MLP相比MLP在准确率和loss方面表现更好![test_accuracy_comparison](https://github.com/user-attachments/assets/9a2cda65-6acb-4365-9e33-1802b0aac305)![test_loss_comparison](https://github.com/user-attachments/assets/c2e90b37-9d58-4de0-9734-2b21c958b94a)
（上述内容均在`train.py`中）

 
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

## 总结：项目要求满足对应
本项目满足了1.2问题中的1~5项目：
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
