
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




# -------------------------- 2. 数据加载工具函数（减少重复代码，提升可维护性）--------------------------
def load_single_mat_data(filename, total_pic, total_mov, total_pic_valence, total_pic_arousal, total_mov_valence, total_mov_arousal):
    """
    加载单个MAT文件的脑电特征与情绪标签，并合并到全局存储列表
    参数：
        filename: MAT文件名称（无需后缀，默认读取.mat文件）
        total_pic: 图片刺激脑电特征列表（每个元素维度：15336×174，时间点×通道）
        total_mov: 视频刺激脑电特征列表（每个元素维度：174×55719，通道×时间点）
        total_pic_valence: 图片刺激的Valence标签列表
        total_pic_arousal: 图片刺激的Arousal标签列表
        total_mov_valence: 视频刺激的Valence标签列表
        total_mov_arousal: 视频刺激的Arousal标签列表
    返回：
        更新后的6个全局存储列表
    """
    # 读取MAT文件（MATLAB导出的cell数据需通过索引[0]提取numpy数组）
    mat_content = io.loadmat(filename)
    
    # 提取图片刺激相关数据
    new_cell_pic = mat_content['new_cell_pic']  # 图片脑电特征（cell类型）
    new_Valence_pic = mat_content['new_Valence_pic']  # 图片Valence标签
    new_Arousal_pic = mat_content['new_Arousal_pic']  # 图片Arousal标签
    
    # 提取视频刺激相关数据
    new_cell_mov = mat_content['new_cell_mov']  # 视频脑电特征（cell类型）
    new_Valence_mov = mat_content['new_Valence_mov']  # 视频Valence标签
    new_Arousal_mov = mat_content['new_Arousal_mov']  # 视频Arousal标签
    
    # 合并图片刺激数据（cell转numpy数组，按样本维度追加）
    for i in range(len(new_cell_pic)):
        total_pic.append(new_cell_pic[i][0])  # cell元素需通过[0]提取数组
    for i in range(len(new_Valence_pic)):
        total_pic_valence.append(new_Valence_pic[i])
    for i in range(len(new_Arousal_pic)):
        total_pic_arousal.append(new_Arousal_pic[i])
    
    # 合并视频刺激数据
    for i in range(len(new_cell_mov)):
        total_mov.append(new_cell_mov[i][0])
    for i in range(len(new_Valence_mov)):
        total_mov_valence.append(new_Valence_mov[i])
    for i in range(len(new_Arousal_mov)):
        total_mov_arousal.append(new_Arousal_mov[i])
    
    return total_pic, total_mov, total_pic_valence, total_pic_arousal, total_mov_valence, total_mov_arousal
