"""
cnn_mil.py - Part of fNIR Base Model.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attention import se_block  # 假设 se_block 已正确定义


# mil直接与网络结合 (Instance 方式)

class CNNLSTMNetMILins(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, cnn_input_channels,
                 use_attention=False, use_conv1to3=False):
        super(CNNLSTMNetMILins, self).__init__()
        # **输入通道数处理**
        self.cnn_input_channels = cnn_input_channels
        self.use_conv1to3 = use_conv1to3

        # DenseNet用于特征提取
        self.DenseNet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.in_features = self.DenseNet.classifier.in_features

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

        # CNN特征提取部分
        layers = [
            self.DenseNet.features[:-2],  # 提取特征，直到倒数第二层
            self.DenseNet.features[-2:],  # 剩余层
        ]
        if use_attention:
            layers.append(se_block(1024))  # 可选添加 SE block
        layers.extend([
            nn.AdaptiveAvgPool2d(1),      # 自适应平均池化
            nn.Flatten(),                 # 展平
            nn.Linear(self.in_features, input_size, bias=True)  # 输出到LSTM的输入维度
        ])
        self.DenseNetAtt = nn.Sequential(*layers)

        # LSTM用于单独处理每个时间序列
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)

        # MIL注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),  # 将LSTM输出投影到128维
            nn.ReLU(),                    # 激活函数
            nn.Linear(128, 1)             # 输出每个实例的注意力分数
        )

        # 最终分类层
        self.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # 输入x的形状为 (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.size()

        # 验证输入通道数
        if channels != self.cnn_input_channels:
            raise ValueError(f"Expected input channels {self.cnn_input_channels}, but got {channels}")

        # 合并CNN特征提取和LSTM处理
        instances = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]  # 提取单个时间序列（图片）
            if self.cnn_input_channels == 1 and self.use_conv1to3:
                frame = self.convert_to_rgb(frame)  # 单通道转三通道
            cnn_out = self.DenseNetAtt(frame)  # 通过CNN提取特征
            cnn_out = cnn_out.view(batch_size, -1)  # 展平为向量，形状为 (batch_size, input_size)
            seq_input = cnn_out.unsqueeze(1)  # (batch_size, 1, input_size)
            lstm_out, _ = self.lstm(seq_input)  # 通过LSTM处理，(batch_size, 1, hidden_size)
            instances.append(lstm_out[:, -1, :])  # 取最后一个时间步输出，(batch_size, hidden_size)
        instances = torch.stack(instances, dim=1)  # (batch_size, seq_len, hidden_size)

        # MIL注意力机制
        attention_scores = self.attention(instances)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # 使用softmax归一化，(batch_size, seq_len, 1)

        # 加权聚合实例
        mil_output = torch.sum(attention_weights * instances, dim=1)  # (batch_size, hidden_size)

        # 分类
        out = self.fc(mil_output)  # (batch_size, output_size)

        return out  # 返回分类结果  修改的主要点：


# Bag 方式
class CNNLSTMWithMILbag(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, cnn_input_channels,
                 use_attention=False, use_conv1to3=False):
        super(CNNLSTMWithMILbag, self).__init__()

        # 输入通道数处理
        self.cnn_input_channels = cnn_input_channels
        self.use_conv1to3 = use_conv1to3

        # CNN 部分：提取每帧特征
        self.DenseNet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.in_features = self.DenseNet.classifier.in_features

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

        # CNN特征提取部分
        layers = [
            self.DenseNet.features[:-2],
            self.DenseNet.features[-2:],
        ]
        if use_attention:
            layers.append(se_block(1024))
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.in_features, input_size, bias=True)
        ])
        self.DenseNetAtt = nn.Sequential(*layers)

        # 特征注意力机制（可选）
        self.feature_attention = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)

        # 子序列内部的注意力机制
        self.internal_attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 子序列级别的注意力机制
        self.subsegment_attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 窗口大小级别的注意力机制
        self.winsize_attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 分类器
        self.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        if channels != self.cnn_input_channels:
            raise ValueError(f"Expected input channels {self.cnn_input_channels}, but got {channels}")

        # CNN 提取特征
        cnn_features = []
        for i in range(seq_len):
            frame = x[:, i, :, :, :]
            if self.cnn_input_channels == 1 and self.use_conv1to3:
                frame = self.convert_to_rgb(frame)
            out = self.DenseNetAtt(frame)
            out = out.view(batch_size, -1)
            cnn_features.append(out)
        cnn_features = torch.stack(cnn_features, dim=1)

        # 为每个 win_size 生成表示
        all_winsize_reps = []
        for win_size in range(2, seq_len + 1):
            subsegment_reps = []
            for start in range(0, seq_len - win_size + 1):
                segment = cnn_features[:, start:start + win_size, :]
                lstm_out, _ = self.lstm(segment)
                internal_scores = self.internal_attention(lstm_out)
                internal_weights = torch.softmax(internal_scores, dim=1)
                subsegment_rep = torch.sum(internal_weights * lstm_out, dim=1)
                subsegment_reps.append(subsegment_rep)
            subsegment_reps = torch.stack(subsegment_reps, dim=1)
            subsegment_scores = self.subsegment_attention(subsegment_reps)
            subsegment_weights = torch.softmax(subsegment_scores, dim=1)
            winsize_rep = torch.sum(subsegment_weights * subsegment_reps, dim=1)
            all_winsize_reps.append(winsize_rep)

        all_winsize_reps = torch.stack(all_winsize_reps, dim=1)
        bag_rep = torch.sum(all_winsize_reps, dim=1)
        out = self.fc(bag_rep)
        return out



# bag ins
class CNNLSTMWithMILbagins(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, cnn_input_channels,
                 use_attention=False, use_conv1to3=False):
        super(CNNLSTMWithMILbagins, self).__init__()

        # 输入通道数处理
        self.cnn_input_channels = cnn_input_channels
        self.use_conv1to3 = use_conv1to3

        # CNN 部分：提取每帧特征
        self.DenseNet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.in_features = self.DenseNet.classifier.in_features

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

        # CNN特征提取部分
        layers = [
            self.DenseNet.features[:-2],
            self.DenseNet.features[-2:],
        ]
        if use_attention:
            layers.append(se_block(1024))
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.in_features, input_size, bias=True)
        ])
        self.DenseNetAtt = nn.Sequential(*layers)

        # 特征注意力机制（可选）
        self.feature_attention = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)

        # 子序列内部的注意力机制
        self.internal_attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 子序列级别的注意力机制
        self.subsegment_attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 窗口大小级别的注意力机制
        self.winsize_attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 分类器
        self.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        if channels != self.cnn_input_channels:
            raise ValueError(f"Expected input channels {self.cnn_input_channels}, but got {channels}")

        # CNN 提取特征
        cnn_features = []
        for i in range(seq_len):
            frame = x[:, i, :, :, :]
            if self.cnn_input_channels == 1 and self.use_conv1to3:
                frame = self.convert_to_rgb(frame)
            out = self.DenseNetAtt(frame)
            out = out.view(batch_size, -1)
            cnn_features.append(out)
        cnn_features = torch.stack(cnn_features, dim=1)

        # 为每个 win_size 生成表示
        all_winsize_reps = []
        for win_size in range(1, seq_len + 1):
            subsegment_reps = []
            for start in range(0, seq_len - win_size + 1):
                segment = cnn_features[:, start:start + win_size, :]
                lstm_out, _ = self.lstm(segment)
                internal_scores = self.internal_attention(lstm_out)
                internal_weights = torch.softmax(internal_scores, dim=1)
                subsegment_rep = torch.sum(internal_weights * lstm_out, dim=1)
                subsegment_reps.append(subsegment_rep)
            subsegment_reps = torch.stack(subsegment_reps, dim=1)
            subsegment_scores = self.subsegment_attention(subsegment_reps)
            subsegment_weights = torch.softmax(subsegment_scores, dim=1)
            winsize_rep = torch.sum(subsegment_weights * subsegment_reps, dim=1)
            all_winsize_reps.append(winsize_rep)

        all_winsize_reps = torch.stack(all_winsize_reps, dim=1)
        bag_rep = torch.sum(all_winsize_reps, dim=1)
        out = self.fc(bag_rep)
        return out


# 总的 CNNLSTMNet 类
# class CNNLSTMNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size, cnn_input_channels=1,
#                  mil_type=0, use_attention=False, use_conv1to3=False):
#         super(CNNLSTMNet, self).__init__()
#
#         # 根据 mil_type 选择对应的模型
#         print(input_size, hidden_size, num_layers, output_size, cnn_input_channels,
#                  mil_type, use_attention, use_conv1to3)
#         if mil_type == 0:
#             self.model = CNNLSTMNetWithMIL(
#                 input_size=input_size,
#                 hidden_size=hidden_size,
#                 num_layers=num_layers,
#                 output_size=output_size,
#                 cnn_input_channels=cnn_input_channels,
#                 use_attention=use_attention,
#                 use_conv1to3=use_conv1to3
#             )
#         elif mil_type == 1:
#             self.model = CNNLSTMWithMIL(
#                 input_size=input_size,
#                 hidden_size=hidden_size,
#                 num_layers=num_layers,
#                 output_size=output_size,
#                 cnn_input_channels=cnn_input_channels,
#                 use_attention=use_attention,
#                 use_conv1to3=use_conv1to3
#             )
#         elif mil_type == 2:
#             self.model = CNNLSTMWithMILbagins(
#                 input_size=input_size,
#                 hidden_size=hidden_size,
#                 num_layers=num_layers,
#                 output_size=output_size,
#                 cnn_input_channels=cnn_input_channels,
#                 use_attention=use_attention,
#                 use_conv1to3=use_conv1to3
#             )
#         else:
#             raise ValueError("mil_type must be 0 (instance) or 1 (bag)")
#
#     def forward(self, x):
#         # 直接调用选择的模型的前向传播
#         return self.model(x)