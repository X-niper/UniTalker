# Borrowed from https://github.com/EvelynFan/FaceFormer/blob/main/main.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# linear interpolation layer
def linear_interpolation(features, output_len: int):
    features = features.transpose(1, 2)
    output_features = F.interpolate(
        features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


# Temporal Bias
def init_biased_mask(n_head, max_seq_len, period):

    def get_slopes(n):

        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(
                2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.div(
        torch.arange(start=0, end=max_seq_len,
                     step=period).unsqueeze(1).repeat(1, period).view(-1),
        period,
        rounding_mode='floor')
    bias = -torch.flip(bias, dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i + 1] = bias[-(i + 1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len,
                                  max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
        mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


# Alignment Bias
def enc_dec_mask(device, T, S):
    mask = torch.ones(T, S)
    for i in range(T):
        mask[i, i] = 0
    return (mask == 1).to(device=device)


# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=3000):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, period, d_model)
        repeat_num = (max_seq_len // period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TCN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.first_net = SeqTranslator1D(
            in_dim, out_dim, min_layers_num=3, residual=True, norm='ln')
        self.dropout = nn.Dropout(0.1)

    def forward(self, spectrogram, style_embed, time_steps=None):

        spectrogram = spectrogram
        spectrogram = self.dropout(spectrogram)
        style_embed = style_embed.unsqueeze(2)
        style_embed = style_embed.repeat(1, 1, spectrogram.shape[2])
        spectrogram = torch.cat([spectrogram, style_embed], dim=1)
        x1 = self.first_net(spectrogram)  # .permute(0, 2, 1)
        if time_steps is not None:
            x1 = F.interpolate(
                x1, size=time_steps, align_corners=False, mode='linear')

        return x1


class SeqTranslator1D(nn.Module):
    """(B, C, T)->(B, C_out, T)"""

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=None,
                 stride=None,
                 min_layers_num=None,
                 residual=True,
                 norm='bn'):
        super(SeqTranslator1D, self).__init__()

        conv_layers = nn.ModuleList([])
        conv_layers.append(
            ConvNormRelu(
                in_channels=C_in,
                out_channels=C_out,
                type='1d',
                kernel_size=kernel_size,
                stride=stride,
                residual=residual,
                norm=norm))
        self.num_layers = 1
        if min_layers_num is not None and self.num_layers < min_layers_num:
            while self.num_layers < min_layers_num:
                conv_layers.append(
                    ConvNormRelu(
                        in_channels=C_out,
                        out_channels=C_out,
                        type='1d',
                        kernel_size=kernel_size,
                        stride=stride,
                        residual=residual,
                        norm=norm))
                self.num_layers += 1
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        return self.conv_layers(x)


class ConvNormRelu(nn.Module):
    """(B,C_in,H,W) -> (B, C_out, H, W) there exist some kernel size that makes
    the result is not H/s.

    #TODO: there might some problems with residual
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 type='1d',
                 leaky=False,
                 downsample=False,
                 kernel_size=None,
                 stride=None,
                 padding=None,
                 p=0,
                 groups=1,
                 residual=False,
                 norm='bn'):
        """conv-bn-relu."""
        super(ConvNormRelu, self).__init__()
        self.residual = residual
        self.norm_type = norm
        # kernel_size = k
        # stride = s

        if kernel_size is None and stride is None:
            if not downsample:
                kernel_size = 3
                stride = 1
            else:
                kernel_size = 4
                stride = 2

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(stride, tuple):
                padding = tuple(int((kernel_size - st) / 2) for st in stride)
            elif isinstance(kernel_size, tuple) and isinstance(stride, int):
                padding = tuple(int((ks - stride) / 2) for ks in kernel_size)
            elif isinstance(kernel_size, tuple) and isinstance(stride, tuple):
                padding = tuple(
                    int((ks - st) / 2) for ks, st in zip(kernel_size, stride))
            else:
                padding = int((kernel_size - stride) / 2)

        if self.residual:
            if downsample:
                if type == '1d':
                    self.residual_layer = nn.Sequential(
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding))
                elif type == '2d':
                    self.residual_layer = nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding))
            else:
                if in_channels == out_channels:
                    self.residual_layer = nn.Identity()
                else:
                    if type == '1d':
                        self.residual_layer = nn.Sequential(
                            nn.Conv1d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding))
                    elif type == '2d':
                        self.residual_layer = nn.Sequential(
                            nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding))

        in_channels = in_channels * groups
        out_channels = out_channels * groups
        if type == '1d':
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups)
            self.norm = nn.BatchNorm1d(out_channels)
            self.dropout = nn.Dropout(p=p)
        elif type == '2d':
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups)
            self.norm = nn.BatchNorm2d(out_channels)
            self.dropout = nn.Dropout2d(p=p)
        if norm == 'gn':
            self.norm = nn.GroupNorm(2, out_channels)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(out_channels)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = nn.ReLU()

    def forward(self, x, **kwargs):
        if self.norm_type == 'ln':
            out = self.dropout(self.conv(x))
            out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        else:
            out = self.norm(self.dropout(self.conv(x)))
        if self.residual:
            residual = self.residual_layer(x)
            out += residual
        return self.relu(out)
