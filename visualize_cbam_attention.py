
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




# -------------------------- 5. 注意力可视化工具函数--------------------------
def visualize_cbam_attention(model, sample_image):
    """
    可视化CBAM模块的通道注意力权重与空间注意力热力图
    参数：
        model: 训练好的CNN+CBAM模型
        sample_image: 单个测试样本特征（维度：频率×时间×通道）
    返回：
        spatial_weights: 空间注意力权重图（维度：1×频率×时间×1）
    """
    # 提取模型中的CBAM层（需确认CBAM在模型中的索引，此处为第2层）
    cbam_layer = model.layers[2]
    assert isinstance(cbam_layer, CBAM), "模型第2层不是CBAM模块，请检查层索引！"
    
    # -------------------------- 5.1 通道注意力权重可视化 --------------------------
    # 构建子模型：输入→通道注意力输出
    channel_att_model = Model(inputs=model.input, outputs=cbam_layer.channel_att(model.layers[1].output))
    channel_att_out = channel_att_model.predict(np.expand_dims(sample_image, axis=0), verbose=0)
    
    # 计算通道权重（全局平均+最大池化融合）
    avg_pool = np.mean(channel_att_out, axis=(1, 2))  # (1, 64)，64为通道数
    max_pool = np.max(channel_att_out, axis=(1, 2))    # (1, 64)
    dense2_weights = cbam_layer.channel_att.dense2.get_weights()[0]  # 全连接层权重（64/16 × 64）
    channel_weights = (dense2_weights @ (avg_pool + max_pool).T).squeeze()  # (64,)，最终通道权重
    
    # 绘制通道注意力权重条形图
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(channel_weights)), channel_weights, alpha=0.7, color='skyblue')
    plt.title("CBAM Channel Attention Weights")
    plt.xlabel("Feature Channel Index (64 Channels)")
    plt.ylabel("Attention Weight (0-1)")
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    # -------------------------- 5.2 空间注意力热力图可视化 --------------------------
    # 构建子模型：输入→空间注意力输出（需先经过通道注意力）
    spatial_att_model = Model(
        inputs=model.input,
        outputs=cbam_layer.spatial_att(cbam_layer.channel_att(model.layers[1].output))
    )
    spatial_weights = spatial_att_model.predict(np.expand_dims(sample_image, axis=0), verbose=0)
    
    # 绘制空间注意力热力图（叠加在原始时频特征上）
    plt.figure(figsize=(8, 6))
    # 原始时频特征（取第0通道作为背景）
    background = sample_image[:, :, 0]
    plt.imshow(background, cmap='viridis', aspect='auto', origin='lower')
    # 空间注意力热力图（透明度0.5，红色为高权重区域）
    plt.imshow(spatial_weights[0, :, :, 0], cmap='jet', alpha=0.5, aspect='auto', origin='lower')
    plt.title("CBAM Spatial Attention Heatmap (Red = High Importance)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    # 坐标轴刻度（匹配实际时间/频率范围）
    plt.xticks(np.linspace(0, sample_image.shape[1]-1, 4), ['0', '1', '2', '3'])
    plt.yticks(np.linspace(0, sample_image.shape[0]-1, 6), ['0', '50', '100', '150', '200', '250'])
    plt.colorbar(label="Spatial Attention Weight")
    plt.show()
    
    return spatial_weights


def visualize_conv_vs_cbam(model, sample_image):
    """
    对比可视化卷积层输出与CBAM处理后的输出（直观展示注意力效果）
    参数：
        model: 训练好的CNN+CBAM模型
        sample_image: 单个测试样本特征（维度：频率×时间×通道）
    返回：
        conv_out: 卷积层输出特征图
        cbam_out: CBAM处理后输出特征图
    """
    # 提取卷积层输出（模型第1层：Conv2D）
    conv_model = Model(inputs=model.input, outputs=model.layers[1].output)
    conv_out = conv_model.predict(np.expand_dims(sample_image, axis=0), verbose=0)  # (1, H, W, 64)
    
    # 提取CBAM输出（模型第2层）
    cbam_model = Model(inputs=model.input, outputs=model.layers[2].output)
    cbam_out = cbam_model.predict(np.expand_dims(sample_image, axis=0), verbose=0)  # (1, H, W, 64)
    
    # 绘制对比图（取第0通道特征，便于直观对比）
    plt.figure(figsize=(14, 6))
    # 卷积层输出
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(conv_out[0, :, :, 0], cmap='viridis', aspect='auto', origin='lower')
    plt.title("Conv2D Output (Before CBAM) - Channel 0")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.xticks(np.linspace(0, conv_out.shape[2]-1, 4), ['0', '1', '2', '3'])
    plt.yticks(np.linspace(0, conv_out.shape[1]-1, 6), ['0', '50', '100', '150', '200', '250'])
    plt.colorbar(im1, fraction=0.046, pad=0.04, label="Feature Value")
    
    # CBAM输出
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(cbam_out[0, :, :, 0], cmap='viridis', aspect='auto', origin='lower')
    plt.title("CBAM Output (After Attention) - Channel 0")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.xticks(np.linspace(0, cbam_out.shape[2]-1, 4), ['0', '1', '2', '3'])
    plt.yticks(np.linspace(0, cbam_out.shape[1]-1, 6), ['0', '50', '100', '150', '200', '250'])
    plt.colorbar(im2, fraction=0.046, pad=0.04, label="Feature Value")
    
    plt.tight_layout()
    plt.show()
    
    return conv_out, cbam_out