"""
LNN.py - Part of fNIR Base Model.
"""

import torch
import torch.nn as nn
from .Attention import se_block
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP, NCP  # 假设这些类已定义在 ncp 模块中


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNNLNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, cnn_input_channels,
                 use_attention=False, use_conv1to3=False, use_cfc=True):
        """
        Initialize the CNNLNNNet with DenseNet121 and configurable RNN (CfC or LTC).

        Parameters:
        - input_size (int): Feature dimension for RNN input (e.g., 128)
        - hidden_size (int): Number of features in the RNN hidden state
        - num_layers (int): Unused in CfC/LTC, kept for compatibility
        - output_size (int): Number of output classes
        - cnn_input_channels (int): Number of input channels for the images
        - use_attention (bool): Whether to use SE block in CNN (default: False)
        - use_conv1to3 (bool): Whether to convert 1 channel to 3 channels (default: False)
        - use_cfc (bool): Whether to use CfC (True) or LTC (False) as RNN (default: True)
        """
        super(CNNLNNNet, self).__init__()
        self.hidden_size = hidden_size

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

        # **控制 CNN 中的注意力机制（SE block）**
        self.use_attention = use_attention
        if self.use_attention:
            layers.append(se_block(1024))  # 在 1024 通道处添加 SE 块

        layers.extend([
            nn.AdaptiveAvgPool2d(1),  # (batch_size * seq_len, 1024, 1, 1)
            nn.Flatten(),  # (batch_size * seq_len, 1024)
            nn.Linear(self.in_features, input_size, bias=True)  # (batch_size * seq_len, input_size)
        ])

        self.DenseNetAtt = nn.Sequential(*layers)

        # **配置 RNN 层（CfC 或 LTC）**
        wiring = AutoNCP(hidden_size, input_size)  # 调整 wiring 参数顺序以匹配输入和隐藏大小
        self.use_cfc = use_cfc
        if self.use_cfc:
            self.rnn = CfC(input_size, wiring)
        else:
            self.rnn = LTC(input_size, wiring)

        # **全连接层进行分类**
        self.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        """
        Forward pass of the CNNLNNNet.

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

        # **RNN 处理时序特征**
        h0 = torch.zeros(batch_size, self.hidden_size).to(cnn_features.device)  # 初始隐藏状态
        out, _ = self.rnn(cnn_features, h0)  # Shape: (batch_size, seq_len, hidden_size)
        out = out[:, -1, :]  # 取最后一个时间步，形状为 (batch_size, hidden_size)

        # **通过全连接层进行分类**
        out = self.fc(out)  # Shape: (batch_size, output_size)

        return out