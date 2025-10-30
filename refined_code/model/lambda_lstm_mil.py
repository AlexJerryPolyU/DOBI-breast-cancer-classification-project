"""
lambda_lstm_mil.py - Part of fNIR Base Model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attention import se_block  # 假设 se_block 已定义在 .Attention 模块中
from lambda_networks import LambdaLayer  # 假设已正确导入

# TemporalMILInstanceLevel 类（已提供）
class TemporalMILInstanceLevel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, lstm_out):
        attention_scores = self.attention(lstm_out)  # (batch, 23, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, 23, 1)
        bag_rep = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size)
        return bag_rep

# TemporalMILBagLevel 类（已提供）
class TemporalMILBagLevel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, lstm_out):
        batch_size, seq_len, _ = lstm_out.shape
        all_pooled = []
        all_scores = []
        for win_size in range(2, seq_len + 1):
            for start in range(0, seq_len - win_size + 1):
                window = lstm_out[:, start:start + win_size, :]
                pooled = F.adaptive_max_pool1d(window.transpose(1, 2), 1).squeeze(-1)
                all_pooled.append(pooled)
                score = self.attention(pooled)
                all_scores.append(score)
        all_pooled = torch.stack(all_pooled, dim=1)
        all_scores = torch.stack(all_scores, dim=1)
        attention_weights = torch.softmax(all_scores, dim=1)
        bag_rep = torch.sum(attention_weights * all_pooled, dim=1)
        return bag_rep

# TemporalMILBaginstanceLevel 类（已提供）
class TemporalMILBaginstanceLevel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.instance_attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.window_attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, lstm_out):
        batch_size, seq_len, _ = lstm_out.shape
        instance_scores = self.instance_attention(lstm_out)
        instance_weights = torch.softmax(instance_scores, dim=1)
        instance_bag_rep = torch.sum(instance_weights * lstm_out, dim=1)
        all_pooled = []
        all_scores = []
        for win_size in range(2, seq_len + 1):
            for start in range(0, seq_len - win_size + 1):
                window = lstm_out[:, start:start + win_size, :]
                pooled = F.adaptive_max_pool1d(window.transpose(1, 2), 1).squeeze(-1)
                all_pooled.append(pooled)
                score = self.window_attention(pooled)
                all_scores.append(score)
        all_pooled = torch.stack(all_pooled, dim=1)
        all_scores = torch.stack(all_scores, dim=1)
        window_weights = torch.softmax(all_scores, dim=1)
        window_bag_rep = torch.sum(window_weights * all_pooled, dim=1)
        combined_bag_rep = (instance_bag_rep + window_bag_rep) / 2
        return combined_bag_rep

# BaginstanceLevel_including_instance 类（已提供）
class BaginstanceLevel_including_instance(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.instance_attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.window_attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, lstm_out):
        batch_size, seq_len, _ = lstm_out.shape
        instance_scores = self.instance_attention(lstm_out)
        instance_weights = torch.softmax(instance_scores, dim=1)
        instance_bag_rep = torch.sum(instance_weights * lstm_out, dim=1)
        all_pooled = []
        all_scores = []
        for win_size in range(1, seq_len + 1):
            for start in range(0, seq_len - win_size + 1):
                window = lstm_out[:, start:start + win_size, :]
                pooled = F.adaptive_max_pool1d(window.transpose(1, 2), 1).squeeze(-1)
                all_pooled.append(pooled)
                score = self.window_attention(pooled)
                all_scores.append(score)
        all_pooled = torch.stack(all_pooled, dim=1)
        all_scores = torch.stack(all_scores, dim=1)
        window_weights = torch.softmax(all_scores, dim=1)
        window_bag_rep = torch.sum(window_weights * all_pooled, dim=1)
        combined_bag_rep = (instance_bag_rep + window_bag_rep) / 2
        return combined_bag_rep

class CNNLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, cnn_input_channels,
                 use_attention=False, use_conv1to3=False, mil_type=0):
        """
        Initialize the CNNLSTMNet with DenseNet121, LSTM, and configurable MIL attention.

        Parameters:
        - input_size (int): Feature dimension for LSTM input (e.g., 128)
        - hidden_size (int): Number of features in the LSTM hidden state
        - num_layers (int): Number of LSTM layers
        - output_size (int): Number of output classes
        - cnn_input_channels (int): Number of input channels for the images
        - use_attention (bool): Whether to use SE block in CNN (default: False)
        - use_conv1to3 (bool): Whether to convert 1 channel to 3 channels (default: False)
        - mil_type (int): Type of MIL attention (0-3, default: 0 for TemporalMILInstanceLevel)
        """
        super(CNNLSTMNet, self).__init__()

        # **Load pre-trained DenseNet**
        self.DenseNet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.in_features = self.DenseNet.classifier.in_features  # 1024

        # **输入通道数处理**
        self.cnn_input_channels = cnn_input_channels
        self.use_conv1to3 = use_conv1to3

        if cnn_input_channels == 1:
            if use_conv1to3:
                self.convert_to_rgb = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
            else:
                # 修改 DenseNet 的第一个卷积层以接受单通道输入
                self.DenseNet.features[0] = nn.Conv2d(
                    1, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
        elif cnn_input_channels == 3:
            self.convert_to_rgb = None
        else:
            self.convert_to_rgb = None
            print(f"Adjusting DenseNet first conv layer for {cnn_input_channels} input channel(s)")
            self.DenseNet.features[0] = nn.Conv2d(
                cnn_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # **定义 DenseNet 的特征提取部分**
        layers = [
            self.DenseNet.features[:-2],  # 前半部分特征提取
            self.DenseNet.features[-2:],  # 后半部分特征提取
        ]

        # **控制 CNN 中的注意力机制（SE block）**
        # 根据 use_attention 添加注意力机制（这里保留 LambdaLayer）
        if use_attention:
            layers.append(
                LambdaLayer(
                    dim=1024,      # 输入通道数
                    dim_out=1024,  # 输出通道数
                    r=23,          # 感受野大小
                    dim_k=16,      # 键维度
                    heads=4,       # 多头数量
                    dim_u=1        # 内部深度维度
                )
            )
        layers.extend([
            nn.AdaptiveAvgPool2d(1),  # (batch_size * seq_len, 1024, 1, 1)
            nn.Flatten(),  # (batch_size * seq_len, 1024)
            nn.Linear(self.in_features, out_features=input_size, bias=True)  # (batch_size * seq_len, input_size)
        ])

        self.DenseNetAtt = nn.Sequential(*layers)

        # **LSTM 层**
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)

        # **配置 MIL 注意机制**
        mil_dict = {
            0: TemporalMILInstanceLevel(hidden_size),
            1: TemporalMILBagLevel(hidden_size),
            2: TemporalMILBaginstanceLevel(hidden_size),
            3: BaginstanceLevel_including_instance(hidden_size)
        }
        self.mil = mil_dict.get(mil_type, None)
        if self.mil is None:
            raise ValueError(f"Unsupported mil_type: {mil_type}. Choose from {list(mil_dict.keys())}.")

        # **全连接层进行分类**
        self.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        """
        Forward pass of the CNNLSTMNet.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, channels, height, width)

        Returns:
        - out (torch.Tensor): Output tensor of shape (batch_size, output_size)
        """
        # **提取维度**
        batch_size, seq_len, channels, height, width = x.size()

        # **验证输入通道数**
        if channels != self.cnn_input_channels:
            raise ValueError(f"Expected input channels {self.cnn_input_channels}, but got {channels}")

        # **CNN 特征提取**
        cnn_features = []
        for i in range(seq_len):
            frame = x[:, i, :, :, :]  # Shape: (batch_size, channels, height, width)
            if self.cnn_input_channels == 1 and self.use_conv1to3:
                frame = self.convert_to_rgb(frame)  # 单通道转三通道
            out = self.DenseNetAtt(frame)  # Shape: (batch_size, input_size)
            out = out.view(batch_size, -1)  # 展平为向量
            cnn_features.append(out)

        # **将 CNN 特征堆叠**，形状为 (batch_size, seq_len, input_size)
        cnn_features = torch.stack(cnn_features, dim=1)

        # **LSTM 处理时序特征**
        lstm_out, _ = self.lstm(cnn_features)  # Shape: (batch_size, seq_len, hidden_size)

        # **应用 MIL 注意机制**
        lstm_out = self.mil(lstm_out)  # Shape: (batch_size, hidden_size)

        # **通过全连接层进行分类**
        out = self.fc(lstm_out)  # Shape: (batch_size, output_size)

        return out