import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# =====================================================
#  Attention
# =====================================================
class AdditiveAttention(nn.Module):
    """
    Bahdanau-style attention (additive attention).
    Can be used for BiLSTM (B, T, H) or TCN (B, C, T).
    """
    def __init__(self, input_dim, attn_dim=256, time_first=False):
        """
        input_dim: feature dimension (H or C)
        attn_dim: hidden dimension for attention
        time_first: if True, input shape is (B, input_dim, T) like TCN
        """
        super().__init__()
        self.time_first = time_first
        self.W_h = nn.Linear(input_dim, attn_dim, bias=False)
        self.W_q = nn.Linear(input_dim, attn_dim, bias=False)
        self.v   = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, H, q=None, lengths=None):
        """
        H: (B, T, H) if time_first=False, (B, H, T) if time_first=True
        q: query vector (B, H) optional; if None, use mean over time
        lengths: optional mask for variable-length sequences
        """
        if self.time_first:
            H = H.permute(0, 2, 1)  # (B, T, H)

        B, T, D = H.size()
        if q is None:
            q = H.mean(dim=1)  # global query

        scores = self.v(torch.tanh(self.W_h(H) + self.W_q(q).unsqueeze(1))).squeeze(-1)  # (B, T)

        if lengths is not None:
            device = scores.device
            arange = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            mask = arange >= lengths.unsqueeze(1)
            scores = scores.masked_fill(mask, float('-inf'))

        alpha = F.softmax(scores, dim=1)  # (B, T)
        context = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)  # (B, D)

        return context, alpha  # context: (B, D), alpha: (B, T)

# =====================================================
#  BiLSTM + Attention Classifier
# =====================================================
class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, num_layers=1,
                 num_classes=50, dropout=0.7, attn_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.attn = AdditiveAttention(input_dim=hidden_dim*2, attn_dim=attn_dim, time_first=False)
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x, lengths=None):
        # x: (B, T, input_dim)
        H, _ = self.lstm(x)   # (B, T, 2H)
        H = self.dropout(H)
        q = H.mean(dim=1)
        context, _ = self.attn(H, q, lengths)
        logits = self.fc(self.dropout(context))
        return logits

# =====================================================
#  Temporal Convolutional Network
# =====================================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1  = nn.ReLU(inplace=True)
        self.drop1  = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2  = nn.ReLU(inplace=True)
        self.drop2  = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.final_relu = nn.ReLU(inplace=True)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.conv1.weight, 0.0, 0.01)
        nn.init.normal_(self.conv2.weight, 0.0, 0.01)
        if isinstance(self.downsample, nn.Conv1d):
            nn.init.normal_(self.downsample.weight, 0.0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = self.downsample(x)
        return self.final_relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=5, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation_size = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                              dilation=dilation_size, padding=padding,
                              dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, C_in, T)
        return self.network(x)

# =====================================================
#  TCN + Attention Classifier
# =====================================================
class TCNClassifierWithAttention(nn.Module):
    def __init__(self, input_dim=2048, num_classes=50,
                 num_channels=(1024,512), kernel_size=6, dropout=0.7, attn_dim=256):
        super().__init__()
        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=list(num_channels),
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.attn = AdditiveAttention(input_dim=num_channels[-1], attn_dim=attn_dim, time_first=True)
        self.fc   = nn.Linear(num_channels[-1], num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, input_dim)
        x = x.permute(0, 2, 1)  # (B, input_dim, T)
        y = self.tcn(x)         # (B, C, T)
        context, _ = self.attn(y)  # (B, C)
        out = self.fc(self.drop(context))
        return out
