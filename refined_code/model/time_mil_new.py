"""
time_mil_new.py - Part of fNIR Base Model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Wavelet Functions (Full Implementations)
def mexican_hat_wavelet(size, scale, shift, device='cpu'):
    """
    Mexican Hat (Ricker) wavelet: Second derivative of a Gaussian.
    Args:
        size (tuple): (feature_dim, kernel_size)
        scale (tensor): Scale parameter (feature_dim,)
        shift (tensor): Shift parameter (feature_dim,)
        device (str): Device for computation
    Returns:
        tensor: Wavelet kernel of shape (feature_dim, kernel_size)
    """
    x = torch.linspace(-(size[1] - 1) / 2, (size[1] - 1) / 2, size[1], device=device)
    x = x.reshape(1, -1).repeat(size[0], 1) - shift.unsqueeze(1)  # Broadcasting shift
    C = 2 / (3 ** 0.5 * torch.pi ** 0.25)  # Normalization constant
    x_scaled = x / scale.unsqueeze(1)
    wavelet = C * (1 - x_scaled ** 2) * torch.exp(-x_scaled ** 2 / 2)
    wavelet = wavelet / (torch.abs(scale) ** 0.5).unsqueeze(1)  # Scale normalization
    return wavelet


def haar_wavelet(size, scale, shift, device='cpu'):
    """
    Haar wavelet: Simple step function for abrupt changes.
    """
    x = torch.linspace(-(size[1] - 1) / 2, (size[1] - 1) / 2, size[1], device=device)
    x = x.reshape(1, -1).repeat(size[0], 1) - shift.unsqueeze(1)
    x_scaled = x / scale.unsqueeze(1)
    wavelet = torch.where(x_scaled < 0, torch.ones_like(x_scaled), -torch.ones_like(x_scaled))
    wavelet = wavelet / (torch.abs(scale) ** 0.5).unsqueeze(1)
    return wavelet


def morlet_wavelet(size, scale, shift, device='cpu'):
    """
    Morlet wavelet: Gaussian-windowed sinusoid for oscillatory patterns.
    """
    x = torch.linspace(-(size[1] - 1) / 2, (size[1] - 1) / 2, size[1], device=device)
    x = x.reshape(1, -1).repeat(size[0], 1) - shift.unsqueeze(1)
    x_scaled = x / scale.unsqueeze(1)
    C = torch.pi ** (-0.25)  # Normalization constant
    wavelet = C * torch.exp(-x_scaled ** 2 / 2) * torch.cos(5 * x_scaled)  # Real part only
    wavelet = wavelet / (torch.abs(scale) ** 0.5).unsqueeze(1)
    return wavelet


def gabor_wavelet(size, scale, shift, device='cpu'):
    """
    Gabor wavelet: Sinusoidal wave with Gaussian envelope.
    """
    x = torch.linspace(-(size[1] - 1) / 2, (size[1] - 1) / 2, size[1], device=device)
    x = x.reshape(1, -1).repeat(size[0], 1) - shift.unsqueeze(1)
    x_scaled = x / scale.unsqueeze(1)
    wavelet = torch.exp(-x_scaled ** 2 / 2) * torch.cos(2 * torch.pi * x_scaled)
    wavelet = wavelet / (torch.abs(scale) ** 0.5).unsqueeze(1)
    return wavelet



class WaveletEncoding(nn.Module):
    def __init__(self, dim=128, wavelet_kernel_size=19, wavelet_type='mexican_hat',
                 num_wavelets=3, multi_scale=1, use_fourier=False, num_recursive_layers=2):
        """
        Initialize the recursive WaveletEncoding module.

        Args:
            dim (int): Feature dimension of the input.
            wavelet_kernel_size (int): Size of the wavelet kernel.
            wavelet_type (str): Type of wavelet ('mexican_hat', 'haar', 'morlet', 'gabor').
            num_wavelets (int): Number of wavelets per scale.
            multi_scale (int): Number of scales per layer.
            use_fourier (bool): Whether to include Fourier encoding.
            num_layers (int): Number of recursive layers (e.g., 2 or 3).
        """
        super(WaveletEncoding, self).__init__()
        self.dim = dim
        self.wavelet_kernel_size = wavelet_kernel_size
        self.wavelet_type = wavelet_type
        self.num_wavelets = num_wavelets
        self.multi_scale = multi_scale
        self.use_fourier = use_fourier
        self.num_layers = num_recursive_layers  # Depth of recursion (2 or 3)

        # Projection layer: Adjust input size based on num_layers and multi_scale
        # Each layer adds multi_scale wavelet outputs, plus the final low-pass
        self.proj = nn.Linear(dim * (self.num_layers * multi_scale + 1), dim)

        # Wavelet function dictionary
        self.wavelet_funcs = {
            'mexican_hat': mexican_hat_wavelet,
            'haar': haar_wavelet,
            'morlet': morlet_wavelet,
            'gabor': gabor_wavelet
        }

        # Learnable wavelet parameters (shared across layers)
        self.wavelets = nn.ParameterList([
            nn.Parameter(torch.randn(2, dim, 1) * 0.1) for _ in range(self.num_wavelets)
        ])

        # Learnable low-pass filter scale (shared across layers)
        self.low_pass_scale = nn.Parameter(torch.randn(dim, 1) * 0.1)

        # Fourier projection layer (if enabled)
        if self.use_fourier:
            self.fft_proj = nn.LazyLinear(out_features=dim)

    def gaussian_low_pass(self, size, scale, device='cpu'):


        """Generate a Gaussian low-pass filter kernel."""
        x = torch.linspace(-(size[1] - 1) / 2, (size[1] - 1) / 2, size[1], device=device)
        x = x.reshape(1, -1).repeat(size[0], 1)
        gaussian = torch.exp(- (x / scale.unsqueeze(1)) ** 2 / 2)
        gaussian = gaussian / gaussian.sum(dim=-1, keepdim=True)  # Normalize to sum to 1
        return gaussian

    def forward(self, x, current_layer=0):
        """
        Forward pass with recursive decomposition.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, feature_dim).
            current_layer (int): Current recursion depth (internal tracking).

        Returns:
            Tensor: Encoded output of shape (batch_size, seq_len, dim) if current_layer == 0,
                    or list of intermediate outputs otherwise.
        """
        batch_size, seq_len, feature_dim = x.size()
        assert seq_len <= 23, "Sequence length exceeds maximum time-step of 23"

        # Select wavelet function
        wavelet_func = self.wavelet_funcs.get(self.wavelet_type, mexican_hat_wavelet)

        # Transpose for convolution: (batch, feature_dim, seq_len)
        x_trans = x.transpose(1, 2)

        # Multi-scale wavelet encoding
        wavelet_outputs = []
        for k in range(self.multi_scale):
            scale_multiplier = 2 ** k  # Geometric progression: 1, 2, 4, ...
            wavelet_kernel_sum = 0
            for i in range(self.num_wavelets):
                scale = self.wavelets[i][0, :, 0] * scale_multiplier
                shift = self.wavelets[i][1, :, 0]
                kernel = wavelet_func((self.dim, self.wavelet_kernel_size), scale, shift, x.device)
                wavelet_kernel_sum = wavelet_kernel_sum + kernel

            scale_pos = F.conv1d(x_trans, wavelet_kernel_sum.unsqueeze(1), groups=self.dim, padding='same')
            wavelet_outputs.append(scale_pos.transpose(1, 2))  # (batch, seq_len, feature_dim)

        # Low-pass filter encoding
        low_pass_kernel = self.gaussian_low_pass((self.dim, self.wavelet_kernel_size), self.low_pass_scale, x.device)
        low_pass_pos = F.conv1d(x_trans, low_pass_kernel.unsqueeze(1), groups=self.dim, padding='same')
        low_pass_pos = low_pass_pos.transpose(1, 2)  # (batch, seq_len, feature_dim)

        # Recursively process the low-pass output if not at the final layer
        if current_layer < self.num_layers - 1:
            recursive_outputs = self.forward(low_pass_pos, current_layer + 1)
        else:
            recursive_outputs = []

        # Combine current wavelet outputs and recursive outputs
        all_outputs = wavelet_outputs + recursive_outputs

        # Include the low-pass output only at the final layer
        if current_layer == self.num_layers - 1:
            all_outputs.append(low_pass_pos)

        # Concatenate all outputs along the feature dimension
        pos = torch.cat(all_outputs, dim=-1)  # (batch, seq_len, dim * (multi_scale * num_layers + 1))

        # Project and finalize only at the top level
        if current_layer == 0:
            pos = self.proj(pos)

            # Optional Fourier encoding at the top level
            if self.use_fourier:
                fft_out = torch.fft.rfft(x, dim=1).abs()
                fft_out = F.interpolate(fft_out, size=seq_len, mode='linear', align_corners=False)
                fft_out = self.fft_proj(fft_out)
                pos = pos + fft_out

            # Combine with input
            x = x + pos

        return x if current_layer == 0 else all_outputs


class CNNLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, cnn_input_channels,
                 wavelet_type='mexican_hat', wavelet_kernel_size=19, num_wavelets=3, multi_scale=1,
                 use_fourier=False, use_attention=False, use_conv1to3=False, num_recursive_layers=2):
        """
        Initialize the CNNLSTMNet with DenseNet121, wavelet encoding, and LSTM.

        Parameters:
        - input_size (int): Feature dimension for LSTM input (set to 128 to match DenseNet output)
        - hidden_size (int): Number of features in the LSTM hidden state
        - num_layers (int): Number of LSTM layers
        - output_size (int): Number of output classes
        - cnn_input_channels (int): Number of input channels for the images
        - wavelet_type (str): Type of wavelet (default: 'mexican_hat')
        - wavelet_kernel_size (int): Size of the wavelet kernel (default: 19)
        - num_wavelets (int): Number of wavelets (default: 3)
        - multi_scale (int): Multi-scale factor for wavelet encoding (default: 1)
        - use_fourier (bool): Whether to use Fourier features (default: False)
        - use_attention (bool): Whether to use attention mechanism (default: False)
        - use_conv1to3 (bool): Whether to convert 1 channel to 3 channels (default: False)
        """
        super(CNNLSTMNet, self).__init__()

        # **Load the Original DenseNet121**
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

        # **Define the Feature Extractor**
        layers = [self.DenseNet.features]  # 初始特征提取层

        # **控制注意力机制**
        self.use_attention = use_attention
        if self.use_attention:
            layers.append(se_block(1024))  # 在 1024 通道处添加 SE 块

        # 添加池化和全连接层
        layers.extend([
            nn.AdaptiveAvgPool2d(1),  # (batch_size * seq_len, 1024, 1, 1)
            nn.Flatten(),  # (batch_size * seq_len, 1024)
            nn.Linear(self.in_features, input_size, bias=True)  # (batch_size * seq_len, input_size)
        ])

        self.DenseNetAtt = nn.Sequential(*layers)

        # **Wavelet Encoding**
        self.pos_layer = WaveletEncoding(
            dim=input_size,
            wavelet_kernel_size=wavelet_kernel_size,
            wavelet_type=wavelet_type,
            num_wavelets=num_wavelets,
            multi_scale=multi_scale,
            use_fourier=use_fourier,
            num_recursive_layers=num_recursive_layers

        )

        # **LSTM Layer**
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.5
        )

        # **Classifier**
        self.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        """
        Forward pass of the CNNLSTMNet.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, channels, height, width)

        Returns:
        - output (torch.Tensor): Output tensor of shape (batch_size, output_size)
        """
        # **Extract Dimensions**
        batch_size, seq_len, channels, height, width = x.size()

        # **验证输入通道数**
        if channels != self.cnn_input_channels:
            raise ValueError(f"Expected input channels {self.cnn_input_channels}, but got {channels}")

        # **逐帧 CNN 处理**
        cnn_features = []
        for i in range(seq_len):
            frame = x[:, i, :, :, :]  # Shape: (batch_size, channels, height, width)
            if self.cnn_input_channels == 1 and self.use_conv1to3:
                frame = self.convert_to_rgb(frame)  # 单通道转三通道
            out = self.DenseNetAtt(frame)  # Shape: (batch_size, input_size)
            out = out.view(batch_size, -1)  # Flatten
            cnn_features.append(out)

        # **Stack 形成时序特征** (batch_size, seq_len, input_size)
        cnn_features = torch.stack(cnn_features, dim=1)

        # **Apply Wavelet Encoding**
        encoded_features = self.pos_layer(cnn_features)  # Shape: (batch_size, seq_len, input_size)

        # **LSTM Processing**
        lstm_out, _ = self.lstm(encoded_features)  # Shape: (batch_size, seq_len, hidden_size)

        # **Extract Last Time Step**
        lstm_out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # **Classification**
        output = self.fc(lstm_out)  # Shape: (batch_size, output_size)

        return output