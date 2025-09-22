
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



# -------------------------- 6. 主流程：数据加载→预处理→训练→评估→可视化 --------------------------
if __name__ == "__main__":
    # -------------------------- 6.1 初始化数据存储列表 --------------------------
    total_pic = []  # 图片刺激脑电特征（15336×174）
    total_mov = []  # 视频刺激脑电特征（174×55719）
    total_pic_valence = []  # 图片Valence标签
    total_pic_arousal = []  # 图片Arousal标签
    total_mov_valence = []  # 视频Valence标签
    total_mov_arousal = []  # 视频Arousal标签

    # -------------------------- 6.2 加载3轮实验的MAT数据 --------------------------
    mat_files = ['10_yang_1', '10_yang_2', '10_yang_3']  # 3次实验的MAT文件名
    for file in mat_files:
        total_pic, total_mov, total_pic_valence, total_pic_arousal, total_mov_valence, total_mov_arousal = load_single_mat_data(
            file, total_pic, total_mov, total_pic_valence, total_pic_arousal, total_mov_valence, total_mov_arousal
        )
    print(f"图片刺激样本数：{len(total_pic)}，视频刺激样本数：{len(total_mov)}")

    # -------------------------- 6.3 标签预处理：Valence标签二分类（0=负性，1=正性） --------------------------
    # 合并图片+视频标签，取第3列有效标签（MAT文件中标签存储格式）
    pic_mov_valence = np.array(total_pic_valence + total_mov_valence)[:, 2].reshape(-1, 1)
    
    # Z-Score标准化标签（均值=0，标准差=1，便于后续二分类划分）
    scaler = StandardScaler()
    pic_mov_valence_scaler = scaler.fit_transform(pic_mov_valence)
    
    # 标签二分类（以0为分界：<=0→负性0，>0→正性1）
    pic_mov_valence_binary = np.where(pic_mov_valence_scaler <= 0, 0, 1)
    
    # 统计标签类别分布（检查数据平衡性）
    unique_labels, label_counts = np.unique(pic_mov_valence_binary, return_counts=True)
    print(f"Valence标签分布：类别{unique_labels}，数量{label_counts}，占比{label_counts/label_counts.sum():.2f}")

    # -------------------------- 6.4 脑电信号预处理：滤波+STFT时频特征提取 --------------------------
    # 关键参数设置
    fs = 512  # 脑电采样率（Hz）
    time_slice = slice(4222, 6722)  # 截取有效时间片段（2500个时间点，约4.88秒）
    freq_slice = slice(2, 294)  # STFT后截取2-150Hz频率范围（脑电有效频段）
    pic_num = len(total_pic)  # 图片刺激样本数
    channel_num = total_pic[0].shape[1]  # 脑电通道数（174）
    
    # 初始化STFT特征数组（维度：样本数×频率维度×时间维度×通道数）
    stft_feature = np.zeros((pic_num, freq_slice.stop - freq_slice.start, 176, channel_num))
    
    # 逐样本、逐通道处理脑电信号
    for sample_idx in range(pic_num):  # 遍历每个图片样本
        for ch_idx in range(channel_num):  # 遍历每个脑电通道
            # 1. 截取当前样本-通道的有效时间片段
            raw_eeg = total_pic[sample_idx][time_slice, ch_idx]
            
            # 2. 陷波滤波：去除50Hz工频干扰及其谐波（100/150/200/250Hz）
            for freq in [50, 100, 150, 200, 250]:
                # 归一化截止频率（Nyquist频率=fs/2=256Hz）
                cutoff = [freq-0.2, freq+0.2] / 256
                b, a = signal.butter(4, cutoff, 'bandstop')  # 4阶Butterworth带阻滤波
                raw_eeg = signal.filtfilt(b, a, raw_eeg)  # 零相位滤波（避免信号偏移）
            
            # 3. STFT时频分析（提取时频特征）
            f, t, Zxx = signal.stft(
                raw_eeg, fs=fs,
                window=signal.get_window('hamming', 1000),  # Hamming窗（减少频谱泄漏）
                nperseg=1000, noverlap=800  # 窗口大小1000，重叠800（时间分辨率更高）
            )
            
            # 4. 特征处理：幅度谱→对数变换（压缩动态范围）→截取有效频段
            amp_spectrum = np.abs(Zxx)  # 复频谱取绝对值→幅度谱
            log_spectrum = 10 * np.log10(amp_spectrum)  # 对数变换（dB单位，增强区分度）
            filtered_spectrum = log_spectrum[freq_slice, :]  # 截取2-150Hz频段
            
            # 5. 保存当前样本-通道的STFT特征
            stft_feature[sample_idx, :, :, ch_idx] = filtered_spectrum
    print(f"STFT特征维度：{stft_feature.shape}（样本数×频率×时间×通道）")

    # -------------------------- 6.5 模型训练与评估（5折分层交叉验证） --------------------------
    # 准备训练数据（仅用图片样本，Valence二分类标签）
    X = stft_feature  # 输入特征：STFT时频特征
    y = pic_mov_valence_binary[0:pic_num]  # 标签：图片样本的Valence二分类标签

    # 交叉验证设置（分层KFold：保持训练/测试集类别分布一致）
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # 早停机制（避免过拟合：验证集损失5轮无改善则停止，恢复最优权重）
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, restore_best_weights=True)

    # 初始化评估指标存储数组
    acc_scores = []  # 准确率
    f1_scores = []   # F1分数（平衡类别不平衡）
    auc_scores = []  # AUC值（二分类区分能力）
    total_y_test = []  # 所有测试集标签（后续可用于混淆矩阵）
    total_y_pred = []  # 所有测试集预测结果

    # 执行5折交叉验证
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== 第{fold+1}折交叉验证 ===")
        # 划分训练集/测试集
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 构建模型（每折重新初始化，避免参数泄露）
        input_shape = (X.shape[1], X.shape[2], X.shape[3])  # (频率维度, 时间维度, 通道数)
        model = build_cbam_cnn_model(input_shape)

        # 训练模型
        model.fit(
            X_train, y_train,
            epochs=10,  # 最大训练轮次（早停机制会提前终止）
            batch_size=8,  # 批次大小（根据GPU内存调整，默认8）
            validation_split=0.2,  # 训练集划分20%为验证集
            callbacks=[earlystop],
            verbose=1
        )

        # 模型评估（测试集）
        # 1. 预测概率→类别（取概率最大的类别）
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # 2. 计算评估指标
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        auc = roc_auc_score(y_test, y_pred_prob[:, 1])  # AUC用正类（1）的预测概率

        # 3. 存储指标
        acc_scores.append(acc)
        f1_scores.append(f1)
        auc_scores.append(auc)
        total_y_test.extend(y_test.flatten())
        total_y_pred.extend(y_pred)

        # 打印当前折指标
        print(f"第{fold+1}折：准确率={acc:.4f}，F1={f1:.4f}，AUC={auc:.4f}")

    # 输出5折交叉验证平均结果
    print(f"\n=== 5折交叉验证平均结果 ===")
    print(f"平均准确率：{np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    print(f"平均F1分数：{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"平均AUC值：{np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

    # 打印模型结构
    print("\n=== 模型结构 ===")
    model.summary()

    # -------------------------- 6.6 CBAM注意力可视化（选择前5个测试样本） --------------------------
    print("\n=== CBAM注意力可视化（前5个测试样本） ===")
    for sample_idx in range(min(5, len(X_test))):
        print(f"\n--- 测试样本{sample_idx+1}（标签：{y_test[sample_idx][0]}，0=负性，1=正性） ---")
        sample_image = X_test[sample_idx]  # 单个测试样本的STFT特征

        # 1. 可视化原始STFT特征（第0通道）
        plt.figure(figsize=(8, 6))
        plt.imshow(sample_image[:, :, 0], cmap='viridis', aspect='auto', origin='lower')
        plt.title(f"Sample {sample_idx+1} - Raw STFT Feature (Channel 0)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.xticks(np.linspace(0, sample_image.shape[1]-1, 4), ['0', '1', '2', '3'])
        plt.yticks(np.linspace(0, sample_image.shape[0]-1, 6), ['0', '50', '100', '150', '200', '250'])
        plt.colorbar(label="Log Amplitude (dB)")
        plt.show()

        # 2. 可视化CBAM注意力（通道权重+空间热力图）
        spatial_weights = visualize_cbam_attention(model, sample_image)

        # 3. 对比卷积层与CBAM输出（直观展示注意力效果）
        conv_out, cbam_out = visualize_conv_vs_cbam(model, sample_image)