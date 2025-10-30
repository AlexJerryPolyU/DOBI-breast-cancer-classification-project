"""
lambda_net.py - Part of fNIR Base Model.
"""

import torch
import torch.nn as nn
from lambda_networks import LambdaLayer  # 假设已正确导入

class CNNLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, cnn_input_channels,
                 use_attention=False, use_conv1to3=False):
        super(CNNLSTMNet, self).__init__()
        # Load pre-trained DenseNet
        self.DenseNet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.in_features = self.DenseNet.classifier.in_features

        # 输入通道数处理
        self.cnn_input_channels = cnn_input_channels
        self.use_conv1to3 = use_conv1to3

        # 通道处理逻辑
        if cnn_input_channels == 1:
            if use_conv1to3:
                self.convert_to_rgb = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
            else:
                print(f"Adjusting DenseNet first conv layer for {cnn_input_channels} input channel(s)")
                self.DenseNet.features[0] = nn.Conv2d(
                    1, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
        elif cnn_input_channels == 3:
            self.convert_to_rgb = None
        else:
            print(f"Adjusting DenseNet first conv layer for {cnn_input_channels} input channel(s)")
            self.DenseNet.features[0] = nn.Conv2d(
                cnn_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # 定义 DenseNetAtt
        layers = [
            self.DenseNet.features[:-2],  # 到 512 通道，7x7
            self.DenseNet.features[-2:],  # 到 1024 通道
        ]

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

        # 验证输入通道数（可选，建议保留）
        if channels != self.cnn_input_channels:
            raise ValueError(f"Expected input channels {self.cnn_input_channels}, but got {channels}")

        # CNN 特征提取
        cnn_features = []
        for i in range(seq_len):
            frame = x[:, i, :, :, :]
            if self.cnn_input_channels == 1 and self.use_conv1to3:
                frame = self.convert_to_rgb(frame)  # 单通道转三通道
            out = self.DenseNetAtt(frame)
            out = out.view(batch_size, -1)
            cnn_features.append(out)

        # 堆叠 CNN 特征
        cnn_features = torch.stack(cnn_features, dim=1)  # (batch_size, seq_len, input_size)

        # LSTM 处理时序特征
        lstm_out, _ = self.lstm(cnn_features)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步

        # 全连接层分类
        out = self.fc(lstm_out)
        return out