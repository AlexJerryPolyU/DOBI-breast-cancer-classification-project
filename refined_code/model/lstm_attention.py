"""
lstm_attention.py - Part of fNIR Base Model.
"""

#lstm—注意力
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .Attention import se_block



class SequenceAttention(nn.Module):
    """Sequence-wise (Encoder-Decoder) Attention"""

    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        # self.first_forward = True

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden)
        # if self.first_forward:
        #     print('SequenceAttention')
        # self.first_forward = False
        attn_weights = F.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        return torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1), attn_weights


class TimeStepAttention(nn.Module):
    """Time-step (Self) Attention"""
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** -0.5

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden)
        Q, K = self.query(lstm_out), self.key(lstm_out)
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        return torch.bmm(attn_weights, lstm_out).mean(dim=1), attn_weights


class FeatureAttention(nn.Module):
    """Feature-wise Attention"""

    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden)
        feature_weights = self.attn(lstm_out.mean(dim=1))
        return torch.sum(lstm_out * feature_weights.unsqueeze(1), dim=1), feature_weights



class LocationAwareAttention(nn.Module):
    """Location-based Attention"""

    def __init__(self, hidden_size, max_len=23):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.attn = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, lstm_out):
        positions = torch.arange(lstm_out.size(1), device=lstm_out.device).unsqueeze(0)
        pos_features = self.pos_emb(positions)
        combined = torch.cat([lstm_out, pos_features.expand(lstm_out.size(0), -1, -1)], dim=-1)
        attn_weights = F.softmax(self.attn(combined).squeeze(-1), dim=1)
        return torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1), attn_weights

class LocalAttention(nn.Module):
    """
    A simplified local attention that focuses on a window around the last timestep.
    e.g. A window of size W around T-1 (the final).
    """
    def __init__(self, hidden_dim, window_size=5):
        super(LocalAttention, self).__init__()
        self.window_size = window_size
        self.bahdanau = BahdanauAttention(hidden_dim)

    def forward(self, lstm_outputs):
        B, T, H = lstm_outputs.shape
        last_idx = T - 1
        start = max(0, last_idx - self.window_size + 1)
        window_outputs = lstm_outputs[:, start:last_idx+1, :]  # shape: (B, W, H)

        # Use standard Bahdanau but only on this local window
        context, alpha = self.bahdanau(window_outputs)
        return context, alpha

class LocationBasedAttention(nn.Module):
    """
    Instead of using hidden states to compute alignment,
    we use only the time-step index (like an RBF over positions).
    For demonstration, we'll make a small trainable MLP over the
    normalized index [0..1].
    """
    def __init__(self, hidden_dim, max_len=100):
        super(LocationBasedAttention, self).__init__()
        self.max_len = max_len
        # MLP to produce a scalar
        self.mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_outputs):
        B, T, H = lstm_outputs.shape
        # build indices from 0..T-1, normalized by T
        idxs = torch.arange(T, device=lstm_outputs.device).float() / float(T-1 if T>1 else 1)
        idxs = idxs.unsqueeze(0).expand(B, T)  # (B, T)

        # pass each index into mlp -> scalar
        # shape => (B, T, 1)
        mlp_in = idxs.unsqueeze(-1)       # (B, T, 1)
        scores = self.mlp(mlp_in).squeeze(-1)  # (B, T)

        alpha = self.softmax(scores)      # (B, T)
        context = torch.bmm(alpha.unsqueeze(1), lstm_outputs).squeeze(1)
        return context, alpha

# ================== Attention Modules ==================
class TemporalAttention(nn.Module):
    """Simple temporal attention that learns weights for each time step"""

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, lstm_out):
        # lstm_out: [batch_size, seq_len, hidden_size]
        attn_weights = self.attention(lstm_out).squeeze(2)  # [batch, seq_len]
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, attn_weights


class BahdanauAttention(nn.Module):
    """Bahdanau-style additive attention"""

    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out: [batch_size, seq_len, hidden_size]
        query = lstm_out[:, -1, :]  # Last timestep as query
        keys = lstm_out

        # Expand query to match sequence length
        query = query.unsqueeze(1).repeat(1, keys.size(1), 1)  # [batch, seq, hid]

        # Calculate attention energies
        energy = torch.tanh(self.W(query) + self.U(keys))
        attn_weights = F.softmax(self.v(energy).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)
        return context, attn_weights


class LuongAttention(nn.Module):
    """Luong-style multiplicative attention"""

    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)

    def forward(self, lstm_out):
        # lstm_out: [batch_size, seq_len, hidden_size]
        query = lstm_out[:, -1, :].unsqueeze(2)  # [batch, hid, 1]
        keys = self.W(lstm_out)  # [batch, seq, hid]

        # Calculate attention scores
        scores = torch.bmm(keys, query).squeeze(2)  # [batch, seq]
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, attn_weights


class ScaledDotProductAttention(nn.Module):
    """Transformer-style scaled dot-product attention"""

    def __init__(self, hidden_size):
        super().__init__()
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size]))

    def forward(self, lstm_out):
        # lstm_out: [batch_size, seq_len, hidden_size]
        query = lstm_out[:, -1, :].unsqueeze(1)  # [batch, 1, hid]
        keys = lstm_out.transpose(1, 2)  # [batch, hid, seq]
        scores = torch.bmm(query, keys) / self.scale.to(lstm_out.device)
        attn_weights = F.softmax(scores.squeeze(1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, attn_weights


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention similar to Transformer"""

    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, lstm_out):
        batch_size, seq_len, _ = lstm_out.size()

        # Project inputs
        Q = self.W_q(lstm_out)
        K = self.W_k(lstm_out)
        V = self.W_v(lstm_out)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Calculate attention scores
        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(
            torch.FloatTensor([self.head_dim]).to(lstm_out.device))
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        context = torch.matmul(attn_weights, V).permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, -1)

        # Aggregate and project
        context = torch.mean(context, dim=1)
        context = self.fc_out(context)
        return context, attn_weights

class SelfAttention(nn.Module):
    """
    Simple single-head self-attention:
    - Each timestep attends to every other timestep in the same sequence.
    - Then we reduce (e.g. average) across timesteps to get a single vector.
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key   = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.scale_factor = math.sqrt(hidden_dim)

    def forward(self, lstm_outputs):
        """
        lstm_outputs: (B, T, H)
        Return:
          - context: (B, H)
          - att_weights: (B, T, T)
        """
        B, T, H = lstm_outputs.shape
        Q = self.query(lstm_outputs)  # (B, T, H)
        K = self.key(lstm_outputs)    # (B, T, H)
        V = self.value(lstm_outputs)  # (B, T, H)

        # (B, T, H) x (B, H, T) => (B, T, T)
        scores = torch.bmm(Q, K.transpose(1,2)) / self.scale_factor
        att_weights = self.softmax(scores)  # (B, T, T)
        att_weights = self.dropout(att_weights)

        # Weighted sum of V => (B, T, H)
        out = torch.bmm(att_weights, V)  # (B, T, H)

        # We can pool across T to get a single vector
        context = out.mean(dim=1)  # (B, H)
        return context, att_weights

class HierarchicalAttention(nn.Module):
    """
    Illustrative hierarchical attention:
    - We do "word-level" attention inside each small chunk,
      then "sentence-level" attention across chunks.
    - For a time series, pretend we chunk it into sub-sequences,
      attend within each, then attend across sub-sequence outputs.

    WARNING: This is just a toy approach to show the concept.
             You'd normally define 2-level RNN or something similar.
    """
    def __init__(self, hidden_dim, chunk_size=23):
        super(HierarchicalAttention, self).__init__()
        # We'll reuse a simple Bahdanau style for sub-chunk attention
        self.sub_attn = BahdanauAttention(hidden_dim)
        self.top_attn = BahdanauAttention(hidden_dim)
        self.chunk_size = chunk_size

    def forward(self, lstm_outputs):
        """
        lstm_outputs: (B, T, H)
        We'll chunk T into segments of size chunk_size. For each chunk:
          1) sub-attend
        Then attend across chunk outputs with top-level attention.
        """
        B, T, H = lstm_outputs.shape
        # break into sub-chunks
        chunked_outputs = []
        # simple chunk loop
        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            chunk = lstm_outputs[:, start:end, :]  # (B, chunk_len, H)
            # apply sub-attn
            sub_context, _ = self.sub_attn(chunk)
            chunked_outputs.append(sub_context.unsqueeze(1))  # (B, 1, H)
        # shape => (B, #chunks, H)
        chunked_outputs = torch.cat(chunked_outputs, dim=1)
        # top-level attention
        top_context, top_alpha = self.top_attn(chunked_outputs)
        return top_context, top_alpha

# ================== Modified CNN-LSTM Model ==================
class CNNLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, cnn_input_channels,
                 attention_type=0, use_attention=False, use_conv1to3=False):
        """
        Initialize the CNNLSTMNet with DenseNet121, LSTM, and optional attention mechanisms.

        Parameters:
        - input_size (int): Feature dimension for LSTM input (e.g., 128)
        - hidden_size (int): Number of features in the LSTM hidden state
        - num_layers (int): Number of LSTM layers
        - output_size (int): Number of output classes
        - cnn_input_channels (int): Number of input channels for the images
        - attention_type (int): Type of attention mechanism (0 for none, 1-13 for specific types)
        - use_attention (bool): Whether to use SE block in CNN (default: False)
        - use_conv1to3 (bool): Whether to convert 1 channel to 3 channels (default: False)
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
        self.use_attention = use_attention
        if self.use_attention:
            layers.append(se_block(1024))  # 在 1024 通道处添加 SE 块

        layers.extend([
            nn.AdaptiveAvgPool2d(1),  # (batch_size * seq_len, 1024, 1, 1)
            nn.Flatten(),  # (batch_size * seq_len, 1024)
            nn.Linear(self.in_features, input_size)  # (batch_size * seq_len, input_size)
        ])

        self.DenseNetAtt = nn.Sequential(*layers)

        # **LSTM 层**
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)

        # **基于 attention_type 的注意力机制**
        if attention_type != 0:  # 如果 attention_type 为 0，则跳过注意力机制
            attn_dict = {
                1: SequenceAttention(hidden_size),
                2: TimeStepAttention(hidden_size),
                3: FeatureAttention(hidden_size),
                4: HierarchicalAttention(hidden_size),
                5: LocationAwareAttention(hidden_size),
                6: TemporalAttention(hidden_size),
                7: BahdanauAttention(hidden_size),
                8: LuongAttention(hidden_size),
                9: ScaledDotProductAttention(hidden_size),
                10: MultiHeadAttention(hidden_size),
                11: LocationBasedAttention(hidden_size),
                12: LocalAttention(hidden_size),
                13: SelfAttention(hidden_size)
            }

            self.attention = attn_dict.get(attention_type, None)

            # 如果 attention_type 无效，抛出错误
            if self.attention is None:
                raise ValueError(f"Unsupported attention type index: {attention_type}. Choose from {list(attn_dict.keys())}.")
        else:
            self.attention = None  # attention_type 为 0 时无注意力机制

        # **分类器**
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
            cnn_features.append(out)
        cnn_features = torch.stack(cnn_features, dim=1)  # [batch, seq, input_size]

        # **LSTM 处理**
        lstm_out, _ = self.lstm(cnn_features)  # [batch, seq, hidden_size]

        # **应用注意力机制（如果有）**
        if self.attention is not None:
            context, _ = self.attention(lstm_out)  # 使用指定的注意力机制
        else:
            context = lstm_out[:, -1, :]  # 如果没有注意力机制，取最后一个时间步

        # **分类**
        out = self.fc(context)
        return out