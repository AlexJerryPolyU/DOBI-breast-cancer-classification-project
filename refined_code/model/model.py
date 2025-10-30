"""
model.py - Part of fNIR Base Model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attention import se_block  # 假设 se_block 已正确定义

class CNNLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 cnn_input_channels, use_attention=False, use_conv1to3=False):
        super(CNNLSTMNet, self).__init__()
        # Load pre-trained DenseNet
        self.DenseNet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.in_features = self.DenseNet.classifier.in_features

        # 输入通道数处理
        self.cnn_input_channels = cnn_input_channels

        self.use_conv1to3 = use_conv1to3

        # 根据输入通道数调整输入层
        if cnn_input_channels == 1:
            if use_conv1to3:
                self.convert_to_rgb = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
            else:
                # 修改 DenseNet 的第一个卷积层以接受单通道输入
                self.DenseNet.features[0] = nn.Conv2d(
                    1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif cnn_input_channels == 3:
            pass
        else:
            print(f"Adjusting DenseNet first conv layer for {cnn_input_channels} input channel(s)")
            self.DenseNet.features[0] = nn.Conv2d(
                cnn_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # 定义 DenseNetAtt
        layers = [
            self.DenseNet.features[:-2],  # 提取 DenseNet 前半部分（到 512 通道）
            self.DenseNet.features[-2:],  # DenseNet 后半部分（到 1024 通道）
        ]

        # 是否加入 se_block
        if use_attention:
            layers.append(se_block(1024))  # 在 1024 通道处添加 SE 块

        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.in_features, out_features=input_size, bias=True)
        ])

        self.DenseNetAtt = nn.Sequential(*layers)

        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        # x 的形状为 (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.size()


        # # 验证输入通道数
        # if channels != self.cnn_input_channels:
        #     raise ValueError(f"Expected input channels {self.cnn_input_channels}, but got {channels}")

        # CNN 提取特征
        cnn_features = []

        for i in range(seq_len):
            frame = x[:, i, :, :, :]  # 提取单帧
            if self.cnn_input_channels == 1 and self.use_conv1to3:
                frame = self.convert_to_rgb(frame)  # 单通道转三通道
            out = self.DenseNetAtt(frame)
            out = out.view(batch_size, -1)  # 展平为向量
            cnn_features.append(out)

        # 将 CNN 特征堆叠，形状为 (batch_size, seq_len, input_size)
        cnn_features = torch.stack(cnn_features, dim=1)

        # LSTM 处理时序特征
        lstm_out, _ = self.lstm(cnn_features)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步

        # 全连接层分类
        out = self.fc(lstm_out)
        return out

"""
 __________________________CNN+Transformer_________________________________
 
"""

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 确保位置编码形状与输入一致
        x = x + self.pe[:x.size(1), :].unsqueeze(0).to(x.device)
        return x


# 定义 CNN + Transformer 网络结构
class CNNTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, cnn_input_channels,
                 nhead=8, dim_feedforward=2048, use_attention=False, use_conv1to3=False):
        """
        Initialize the CNNTransformer with DenseNet121 and Transformer Encoder.

        Parameters:
        - input_size (int): Feature dimension for Transformer input (e.g., 128)
        - hidden_size (int): Unused in this model, kept for compatibility
        - num_layers (int): Number of Transformer encoder layers
        - output_size (int): Number of output classes
        - cnn_input_channels (int): Number of input channels for the images
        - nhead (int): Number of heads in multi-head attention (default: 8)
        - dim_feedforward (int): Dimension of the feedforward network in Transformer (default: 2048)
        - use_attention (bool): Whether to use attention mechanism (e.g., SE block) (default: False)
        - use_conv1to3 (bool): Whether to convert 1 channel to 3 channels (default: False)
        """
        super(CNNTransformer, self).__init__()

        # **Load pre-trained DenseNet**
        self.DenseNet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.in_features = self.DenseNet.classifier.in_features  # 1024

        # **控制 1 转 3 卷积**
        self.use_conv1to3 = use_conv1to3
        if self.use_conv1to3:
            self.convert_to_rgb = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.convert_to_rgb = None
            # 如果不使用 1 转 3，则直接修改 DenseNet 的第一层以匹配输入通道数
            self.DenseNet.features[0] = nn.Conv2d(
                cnn_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # **定义 DenseNet 的特征提取部分**
        layers = [
            self.DenseNet.features[:-2],  # 前半部分特征提取
            self.DenseNet.features[-2:],  # 后半部分特征提取
        ]

        # **控制注意力机制**
        self.use_attention = use_attention
        if self.use_attention:
            layers.append(se_block(1024))  # 在 1024 通道处添加 SE 块

        layers.extend([
            nn.AdaptiveAvgPool2d(1),  # (batch_size * seq_len, 1024, 1, 1)
            nn.Flatten(),  # (batch_size * seq_len, 1024)
            nn.Linear(self.in_features, out_features=128, bias=True)  # (batch_size * seq_len, 128)
        ])

        self.DenseNetAtt = nn.Sequential(*layers)

        # **定义 Transformer Encoder**
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,  # Transformer 的输入维度
            nhead=nhead,  # 多头注意力头数
            dim_feedforward=dim_feedforward,  # Feedforward 网络维度
            dropout=0.5  # Dropout 比例
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # **定义位置编码**
        self.positional_encoding = PositionalEncoding(d_model=input_size)

        # **全连接层进行分类**
        self.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(input_size, output_size),
        )

    def forward(self, x):
        """
        Forward pass of the CNNTransformer.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, channels, height, width)

        Returns:
        - out (torch.Tensor): Output tensor of shape (batch_size, output_size)
        """
        # **提取维度**
        batch_size, seq_len, channels, height, width = x.size()

        # **检查输入通道兼容性**
        if self.use_conv1to3 and channels != 1:
            raise ValueError("use_conv1to3=True requires input channels to be 1")

        # **CNN 提取特征**
        cnn_features = []
        for i in range(seq_len):
            frame = x[:, i, :, :, :]  # Shape: (batch_size, channels, height, width)
            if self.use_conv1to3 and self.convert_to_rgb is not None:
                frame = self.convert_to_rgb(frame)  # 单通道转三通道
            out = self.DenseNetAtt(frame)  # Shape: (batch_size, 128)
            out = out.view(batch_size, -1)  # 展平为向量
            cnn_features.append(out)

        # **将 CNN 特征堆叠**，形状为 (batch_size, seq_len, feature_dim)
        cnn_features = torch.stack(cnn_features, dim=1)

        # **调整为 Transformer 输入的形状** (seq_len, batch_size, feature_dim)
        cnn_features = cnn_features.permute(1, 0, 2)

        # **添加位置编码**
        cnn_features = self.positional_encoding(cnn_features)

        # **Transformer 处理时序特征**
        transformer_out = self.transformer(cnn_features)

        # **取 Transformer 最后一个时间步的输出** (batch_size, feature_dim)
        transformer_out = transformer_out[-1, :, :]

        # **通过全连接层进行分类**
        out = self.fc(transformer_out)

        return out


"""
 __________________________CNN+GRU_________________________________

"""
class CNNGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, cnn_input_channels,
                 use_attention=False, use_conv1to3=False):
        """
        Initialize the CNNGRU with DenseNet121 and GRU.

        Parameters:
        - input_size (int): Feature dimension for GRU input (e.g., 128)
        - hidden_size (int): Number of features in the GRU hidden state
        - num_layers (int): Number of GRU layers
        - output_size (int): Number of output classes
        - cnn_input_channels (int): Number of input channels for the images
        - use_attention (bool): Whether to use attention mechanism (e.g., SE block) (default: False)
        - use_conv1to3 (bool): Whether to convert 1 channel to 3 channels (default: False)
        """
        super(CNNGRU, self).__init__()

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
            raise ValueError("cnn_input_channels must be 1 or 3")

        # **定义 DenseNet 的特征提取部分**
        layers = [
            self.DenseNet.features[:-2],  # 前半部分特征提取
            self.DenseNet.features[-2:],  # 后半部分特征提取
        ]

        # **控制注意力机制**
        self.use_attention = use_attention
        if self.use_attention:
            layers.append(se_block(1024))  # 在 1024 通道处添加 SE 块

        layers.extend([
            nn.AdaptiveAvgPool2d(1),  # (batch_size * seq_len, 1024, 1, 1)
            nn.Flatten(),  # (batch_size * seq_len, 1024)
            nn.Linear(self.in_features, out_features=input_size, bias=True)  # (batch_size * seq_len, input_size)
        ])

        self.DenseNetAtt = nn.Sequential(*layers)

        # **替换 LSTM 为 GRU**
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)

        # **全连接层进行分类**
        self.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        """
        Forward pass of the CNNGRU.

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

        # **GRU 处理时序特征**
        gru_out, _ = self.gru(cnn_features)  # Shape: (batch_size, seq_len, hidden_size)
        gru_out = gru_out[:, -1, :]  # 取最后一个时间步的输出，形状为 (batch_size, hidden_size)

        # **通过全连接层进行分类**
        out = self.fc(gru_out)  # Shape: (batch_size, output_size)

        return out