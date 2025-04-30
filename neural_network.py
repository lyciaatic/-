import numpy as np
from abc import ABC, abstractmethod
import pickle
import os

# ==================== 基本层类 ====================
class Layer(ABC):
    """所有层的基类"""
    def __init__(self):
        self.optimizable = True
    
    @abstractmethod
    def forward(self, inputs):
        pass
    
    @abstractmethod
    def backward(self, grad):
        pass
    
    def __call__(self, inputs):
        return self.forward(inputs)

# ==================== 核心层 ====================
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

class Conv2D(Layer):
    """二维卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, weight_decay=0, weight_decay_lambda=1e-4):
        super().__init__()
        
        # 支持单数或者元组形式的kernel_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(stride, int):
            stride = (stride, stride)
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 初始化权重
        self.W = np.random.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]) * 0.01
        self.b = np.zeros(out_channels)
        
        # 优化器参数
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        
        # L2正则化
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        
        # 中间变量
        self.X_cols = None
        self.input_shape = None
    
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
        
        # 计算输出大小
        out_h = (H + 2 * p_h - k_h) // s_h + 1
        out_w = (W + 2 * p_w - k_w) // s_w + 1
        
        # 添加padding
        if p_h > 0 or p_w > 0:
            X_pad = np.pad(X, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)), 'constant')
        else:
            X_pad = X
        
        # 初始化输出矩阵
        cols = np.zeros((N, C, k_h, k_w, out_h, out_w))
        
        # 填充输出矩阵
        for h in range(k_h):
            h_max = h + s_h * out_h
            for w in range(k_w):
                w_max = w + s_w * out_w
                cols[:, :, h, w, :, :] = X_pad[:, :, h:h_max:s_h, w:w_max:s_w]
        
        # 重新排列输出维度
        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        
        return cols, (N, out_h, out_w)
    
    def _col2im(self, dcols, output_shape):
        """将展开的列变回图像形式
        输入:
            dcols: (N*out_h*out_w, C*k_h*k_w)
            output_shape: 原始图像形状 (N, C, H, W)
        输出:
            dX: (N, C, H, W)
        """
        N, C, H, W = output_shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        
        # 计算输出大小
        out_h = (H + 2 * p_h - k_h) // s_h + 1
        out_w = (W + 2 * p_w - k_w) // s_w + 1
        
        # 重塑为原始形状
        dcols = dcols.reshape(N, out_h, out_w, C, k_h, k_w).transpose(0, 3, 4, 5, 1, 2)
        
        # 初始化梯度
        dX_pad = np.zeros((N, C, H + 2 * p_h, W + 2 * p_w))
        
        # 将梯度重新填充到原始位置
        for h in range(k_h):
            h_max = h + s_h * out_h
            for w in range(k_w):
                w_max = w + s_w * out_w
                dX_pad[:, :, h:h_max:s_h, w:w_max:s_w] += dcols[:, :, h, w, :, :]
        
        # 移除padding
        if p_h > 0 or p_w > 0:
            dX = dX_pad[:, :, p_h:-p_h, p_w:-p_w]
        else:
            dX = dX_pad
        
        return dX
    
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

class MaxPool2D(Layer):
    """最大池化层"""
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        self.input = None
        self.input_pad = None
        self.max_indices = None
        self.optimizable = False
    
    def forward(self, inputs):
        batch_size, channels, in_h, in_w = inputs.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        
        # 计算输出尺寸
        out_h = (in_h + 2 * ph - kh) // sh + 1
        out_w = (in_w + 2 * pw - kw) // sw + 1
        
        # 输入padding
        self.input = inputs
        if ph > 0 or pw > 0:
            self.input_pad = np.pad(inputs, ((0, 0), (0, 0), (ph, ph), (pw, pw)), 'constant', constant_values=-np.inf)
        else:
            self.input_pad = inputs
        
        # 初始化输出和最大值索引
        out = np.zeros((batch_size, channels, out_h, out_w))
        self.max_indices = np.zeros((batch_size, channels, out_h, out_w, 2), dtype=int)
        
        # 执行池化
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    i_start = i * sh
                    for j in range(out_w):
                        j_start = j * sw
                        # 提取当前位置的patch
                        patch = self.input_pad[b, c, i_start:i_start+kh, j_start:j_start+kw]
                        # 找到最大值及其索引
                        idx = np.unravel_index(np.argmax(patch), patch.shape)
                        out[b, c, i, j] = patch[idx]
                        # 记录最大值的相对位置
                        self.max_indices[b, c, i, j] = np.array(idx)
        
        return out
    
    def backward(self, grad):
        batch_size, channels, in_h, in_w = self.input.shape
        _, _, out_h, out_w = grad.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        
        # 初始化输入梯度
        dX_pad = np.zeros_like(self.input_pad)
        
        # 计算梯度
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    i_start = i * sh
                    for j in range(out_w):
                        j_start = j * sw
                        # 获取最大值的位置
                        idx = tuple(self.max_indices[b, c, i, j])
                        # 将梯度传递给最大值位置
                        dX_pad[b, c, i_start+idx[0], j_start+idx[1]] += grad[b, c, i, j]
        
        # 如果有padding，需要去除padding部分
        if ph > 0 or pw > 0:
            dX = dX_pad[:, :, ph:-ph, pw:-pw]
        else:
            dX = dX_pad
        
        return dX

class BatchNorm2D(Layer):
    """批归一化层"""
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        # 可训练参数: 缩放和偏移
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))
        
        # 运行时统计量
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))
        
        # 批次统计量
        self.batch_mean = None
        self.batch_var = None
        self.normalized = None
        self.input = None
        
        self.params = {'gamma': self.gamma, 'beta': self.beta}
        self.grads = {'gamma': None, 'beta': None}
        
        self.training = True
    
    def forward(self, inputs):
        self.input = inputs
        
        if self.training:
            # 计算批次统计量
            axes = (0, 2, 3)  # 在batch、height和width维度上计算均值和方差
            self.batch_mean = np.mean(inputs, axis=axes, keepdims=True)
            self.batch_var = np.var(inputs, axis=axes, keepdims=True)
            
            # 更新运行时统计量
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
            
            # 归一化
            self.normalized = (inputs - self.batch_mean) / np.sqrt(self.batch_var + self.eps)
            # 缩放和偏移
            return self.gamma * self.normalized + self.beta
        else:
            # 测试时使用运行时统计量
            normalized = (inputs - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * normalized + self.beta
    
    def backward(self, grad):
        # 批次大小
        batch_size = grad.shape[0] * grad.shape[2] * grad.shape[3]
        
        # 计算beta的梯度
        dbeta = np.sum(grad, axis=(0, 2, 3), keepdims=True)
        
        # 计算gamma的梯度
        dgamma = np.sum(grad * self.normalized, axis=(0, 2, 3), keepdims=True)
        
        # 计算关于归一化输入的梯度
        dnorm = grad * self.gamma
        
        # 计算关于方差的梯度
        dvar = np.sum(dnorm * (self.input - self.batch_mean) * (-0.5) * np.power(self.batch_var + self.eps, -1.5), 
                       axis=(0, 2, 3), keepdims=True)
        
        # 计算关于均值的梯度
        dmean = np.sum(dnorm * (-1) / np.sqrt(self.batch_var + self.eps), axis=(0, 2, 3), keepdims=True) + \
                dvar * np.sum(-2 * (self.input - self.batch_mean), axis=(0, 2, 3), keepdims=True) / batch_size
        
        # 计算关于输入的梯度
        dx = dnorm / np.sqrt(self.batch_var + self.eps) + \
             dvar * 2 * (self.input - self.batch_mean) / batch_size + \
             dmean / batch_size
        
        # 保存参数梯度
        self.grads['gamma'] = dgamma
        self.grads['beta'] = dbeta
        
        return dx
    
    def set_train(self, training=True):
        self.training = training

class Flatten(Layer):
    """展平层，将多维输入展平为二维"""
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.optimizable = False
    
    def forward(self, inputs):
        self.input_shape = inputs.shape
        batch_size = inputs.shape[0]
        return inputs.reshape(batch_size, -1)
    
    def backward(self, grad):
        return grad.reshape(self.input_shape)

class Dropout(Layer):
    """Dropout层，用于防止过拟合"""
    def __init__(self, drop_rate=0.5):
        super().__init__()
        self.drop_rate = drop_rate
        self.mask = None
        self.training = True
        self.optimizable = False
    
    def forward(self, inputs):
        if self.training:
            # 生成二项分布掩码
            self.mask = np.random.binomial(1, 1 - self.drop_rate, inputs.shape) / (1 - self.drop_rate)
            return inputs * self.mask
        else:
            return inputs
    
    def backward(self, grad):
        if self.training:
            return grad * self.mask
        else:
            return grad
    
    def set_train(self, training=True):
        self.training = training

# ==================== 激活函数 ====================
class ReLU(Layer):
    """ReLU激活函数"""
    def __init__(self):
        super().__init__()
        self.input = None
        self.optimizable = False
    
    def forward(self, inputs):
        self.input = inputs
        return np.maximum(0, inputs)
    
    def backward(self, grad):
        return grad * (self.input > 0)

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

class Sigmoid(Layer):
    """Sigmoid激活函数"""
    def __init__(self):
        super().__init__()
        self.output = None
        self.optimizable = False
    
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output
    
    def backward(self, grad):
        return grad * self.output * (1 - self.output)

# ==================== 辅助函数 ====================
def softmax(x):
    """Softmax函数"""
    x_max = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ==================== 损失函数 ====================
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

# ==================== 优化器 ====================
class SGD:
    """随机梯度下降优化器"""
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate
    
    def step(self):
        """执行一次梯度更新"""
        for layer in self.model.layers:
            if hasattr(layer, 'params') and hasattr(layer, 'grads') and layer.grads is not None:
                for param_name in layer.params:
                    if layer.grads[param_name] is not None:
                        layer.params[param_name] -= self.learning_rate * layer.grads[param_name]
                        # 更新实际参数值
                        setattr(layer, param_name, layer.params[param_name])

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

class Adam:
    """Adam优化器"""
    def __init__(self, model, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.model = model
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
        
        # 初始化动量和二阶矩估计
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                self.m[i] = {}
                self.v[i] = {}
                for param_name in layer.params:
                    self.m[i][param_name] = np.zeros_like(layer.params[param_name])
                    self.v[i][param_name] = np.zeros_like(layer.params[param_name])
    
    def step(self):
        """执行一次Adam更新"""
        self.t += 1
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads') and layer.grads is not None:
                for param_name in layer.params:
                    if layer.grads[param_name] is not None:
                        grad = layer.grads[param_name]
                        
                        # 更新一阶矩估计
                        self.m[i][param_name] = self.beta1 * self.m[i][param_name] + (1 - self.beta1) * grad
                        # 更新二阶矩估计
                        self.v[i][param_name] = self.beta2 * self.v[i][param_name] + (1 - self.beta2) * (grad ** 2)
                        
                        # 计算偏差修正
                        m_corrected = self.m[i][param_name] / (1 - self.beta1 ** self.t)
                        v_corrected = self.v[i][param_name] / (1 - self.beta2 ** self.t)
                        
                        # 更新参数
                        layer.params[param_name] -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
                        # 更新实际参数值
                        setattr(layer, param_name, layer.params[param_name])

# ==================== 学习率调度器 ====================
class StepLR:
    """步长学习率调度器"""
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = optimizer.learning_rate
        self.step_count = 0
    
    def step(self):
        """更新学习率"""
        self.step_count += 1
        if self.step_count % self.step_size == 0:
            self.optimizer.learning_rate *= self.gamma

class ExponentialLR:
    """指数学习率调度器"""
    def __init__(self, optimizer, gamma=0.95):
        self.optimizer = optimizer
        self.gamma = gamma
        self.base_lr = optimizer.learning_rate
    
    def step(self):
        """更新学习率"""
        self.optimizer.learning_rate *= self.gamma 