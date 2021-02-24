import math

import torch
import torch.nn as nn

from .....utils.os_utl import check_options, check_types
# from utils.os_utl import check_options, check_types


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=padding,
                                   padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    _tick_sizes = dict(m=13, w=54, d=32, wd=7, H=24, M=60, S=60)
    _time_tick_map = dict(S=0, M=1, H=2, d=3, w=4, m=5)

    @check_types(minute_window=int)
    @check_options(embed_type=('fixed', 'learned'))
    def __init__(self, d_model, embed_type='fixed', time_tick='M', tick_scale=1):
        super(TemporalEmbedding, self).__init__()

        if self._tick_sizes[time_tick] % tick_scale:
            raise ValueError(f'"tick_scale" passed ({tick_scale}) must be an even divisor of respective '
                             f'time_tick ("{time_tick}") size ({self._tick_sizes[time_tick]}).')

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        _time_tick = self._time_tick_map[time_tick]
        _tick_sizes = self._tick_sizes.copy()
        _tick_sizes[time_tick] //= tick_scale

        # Generated needed embedders
        self.month_embed = Embed(_tick_sizes['m'], d_model)

        if _time_tick < 5:
            self.week_embed = Embed(_tick_sizes['w'], d_model)

        if _time_tick < 4:
            self.day_embed = Embed(_tick_sizes['d'], d_model)
            self.weekday_embed = Embed(_tick_sizes['wd'], d_model)

        if _time_tick < 3:
            self.hour_embed = Embed(_tick_sizes['H'], d_model)

        if _time_tick < 2:
            self.minute_embed = Embed(_tick_sizes['M'], d_model)

        if _time_tick < 1:
            self.second_embed = Embed(_tick_sizes['S'], d_model)

    def forward(self, x):
        x = x.long()
        dims = ('month', 'week', 'day', 'weekday', 'hour', 'minute', 'second')

        # get embedding for each dimension and sum them. If dim doesn't exist assign 0.
        out = [getattr(self, f'{attr}_embed', 0.) for attr in dims]
        return sum(emb(x[:, :, i]) if callable(emb) else emb for i, emb in enumerate(out))


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', time_tick='M', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, time_tick=time_tick)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)
