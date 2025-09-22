
import numpy as np
import scipy.io as io  # 读取MAT格式脑电数据
from scipy import signal  # 信号处理（陷波滤波、STFT时频分析）
import matplotlib.pyplot as plt  # 结果可视化与注意力热力图
from collections import Counter  # 统计标签类别分布

# sklearn工具：数据预处理、交叉验证、评估指标
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# TensorFlow/Keras：深度学习模型构建（CBAM注意力+CNN）
import tensorflow as tf
from tensorflow.keras import layers, backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# -------------------------- 3. CBAM注意力模块定义（通道+空间注意力串行执行）--------------------------
class ChannelAttention(layers.Layer):
    """
    通道注意力模块：通过全局平均/最大池化捕捉通道间重要性差异，增强关键通道特征
    原理：对每个通道的全局信息建模，生成通道权重，抑制无关通道干扰
    """
    def __init__(self, filters, ratio=16):
        super(ChannelAttention, self).__init__()
        self.filters = filters  # 输入特征图的通道数
        self.ratio = ratio      # 通道压缩比（减少参数量，避免过拟合）
        
        # 全局池化层：捕捉每个通道的全局信息（两种池化互补）
        self.avg_pool = GlobalAveragePooling2D()  # 全局平均池化（强调整体趋势）
        self.max_pool = GlobalMaxPooling2D()      # 全局最大池化（突出局部峰值）
        
        # 全连接层：压缩通道维度→激活→恢复通道维度（生成通道权重）
        self.dense1 = layers.Dense(filters // ratio, activation='relu')  # 压缩通道
        self.dense2 = layers.Dense(filters, activation='sigmoid')        # 输出0-1权重

    def call(self, inputs):
        # 1. 全局池化：(batch, H, W, C) → (batch, C)
        avg_pool_out = self.avg_pool(inputs)
        max_pool_out = self.max_pool(inputs)
        
        # 2. 维度调整：(batch, C) → (batch, 1, 1, C)（匹配输入特征图维度，便于后续乘法）
        avg_pool_out = layers.Reshape((1, 1, self.filters))(avg_pool_out)
        max_pool_out = layers.Reshape((1, 1, self.filters))(max_pool_out)
        
        # 3. 生成通道权重：全连接层计算+两种池化结果融合
        avg_weight = self.dense2(self.dense1(avg_pool_out))
        max_weight = self.dense2(self.dense1(max_pool_out))
        channel_weight = layers.Add()([avg_weight, max_weight])  # 权重融合（平均+最大）
        
        # 4. 特征加权：输入特征图 × 通道权重（增强关键通道）
        return layers.Multiply()([inputs, channel_weight])


class SpatialAttention(layers.Layer):
    """
    空间注意力模块：通过通道维度的平均/最大池化捕捉空间（时频域）重要区域
    原理：对每个空间位置的跨通道信息建模，生成空间权重，聚焦有效时频区域
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size  # 卷积核大小（7×7兼顾感受野与计算效率）
        
        # 卷积层：输入2个通道（平均+最大池化），输出1个空间注意力图（0-1权重）
        self.conv = layers.Conv2D(1, kernel_size, padding='same', activation='sigmoid')

    def call(self, inputs):
        # 1. 通道维度池化：(batch, H, W, C) → (batch, H, W, 1)（压缩通道，保留空间信息）
        avg_pool_out = layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(inputs)  # 通道平均
        max_pool_out = layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(inputs)    # 通道最大
        
        # 2. 特征融合：2个通道拼接 → (batch, H, W, 2)
        concat_out = layers.Concatenate()([avg_pool_out, max_pool_out])
        
        # 3. 生成空间权重：卷积层提取空间特征→sigmoid激活（0-1权重）
        spatial_weight = self.conv(concat_out)
        
        # 4. 特征加权：输入特征图 × 空间权重（聚焦有效时频区域）
        return layers.Multiply()([inputs, spatial_weight])


class CBAM(layers.Layer):
    """
    CBAM注意力模块：通道注意力 → 空间注意力（串行执行，双重聚焦关键特征）
    优势：先筛选重要通道，再聚焦通道内的有效空间区域，减少冗余信息干扰
    """
    def __init__(self, filters, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(filters, ratio)  # 通道注意力（先执行）
        self.spatial_att = SpatialAttention(kernel_size)     # 空间注意力（后执行）

    def call(self, inputs):
        # 串行执行：通道注意力→空间注意力
        x = self.channel_att(inputs)
        x = self.spatial_att(x)
        return x


# -------------------------- 4. CNN+CBAM模型构建函数--------------------------
def build_cbam_cnn_model(input_shape):
    """
    构建基于CBAM注意力的CNN分类模型（适配脑电STFT时频特征）
    参数：
        input_shape: 输入特征维度 (频率维度, 时间维度, 通道数)
    返回：
        编译后的Keras模型
    """
    # 1. 输入层（匹配STFT时频特征维度：频率×时间×通道）
    inputs = Input(shape=input_shape)
    
    # 2. 卷积层1：提取局部时频特征（3×4卷积核，兼顾频率与时间维度的局部关联）
    x = layers.Conv2D(64, (3, 4), activation='relu', padding='same')(inputs)
    x = CBAM(64)(x)  # 加入CBAM注意力（通道数=卷积输出通道数64）
    
    # 3. 卷积层2+最大池化：进一步提取高级特征+降维（减少计算量）
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)  # 2×2池化，频率和时间维度各降为1/2
    
    # 4. 全连接层：将时频特征映射为分类向量
    x = layers.Flatten()(x)  # 展平特征图（batch × (H*W*C)）
    x = layers.Dense(64, activation='relu')(x)  # 64维全连接层（减少参数量）
    
    # 5. 输出层：二分类概率（softmax激活，输出0/1类概率）
    outputs = layers.Dense(2, activation='softmax')(x)
    
    # 6. 模型编译（优化器+损失函数+评估指标）
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=1e-4)  # 学习率1e-4（避免梯度爆炸，适配小样本）
    model.compile(
        loss='sparse_categorical_crossentropy',  # 稀疏交叉熵（标签为整数，无需One-Hot）
        optimizer=optimizer,
        metrics=['accuracy']  # 训练中监控准确率
    )
    return model
