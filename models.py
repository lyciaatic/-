from neural_network import *
import pickle
import numpy as np

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


class ResNet(Model):
    """残差网络模型"""
    def __init__(self, input_shape=(1, 28, 28), num_classes=10, block_counts=[2, 2, 2]):
        super().__init__()
        in_channels, height, width = input_shape
        
        # 初始卷积层
        self.layers.append(Conv2D(in_channels, 64, kernel_size=3, stride=1, padding=1, weight_decay=1e-4))
        self.layers.append(BatchNorm2D(64))
        self.layers.append(ReLU())
        
        # 残差块
        channels = 64
        for i, block_count in enumerate(block_counts):
            for j in range(block_count):
                # 只有每组的第一个块可能改变尺寸
                stride = 2 if j == 0 and i > 0 else 1
                # 添加残差块
                self.layers.append(ResidualBlock(channels, channels * (2 if j == 0 and i > 0 else 1), stride))
                # 更新通道数
                channels = channels * (2 if j == 0 and i > 0 else 1)
        
        # 全局平均池化
        # 计算最终特征图尺寸
        for i in range(len(block_counts) - 1):
            height //= 2
            width //= 2
        
        # 展平层
        self.layers.append(Flatten())
        
        # 全连接层
        self.layers.append(Linear(channels * height * width, num_classes))
    
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