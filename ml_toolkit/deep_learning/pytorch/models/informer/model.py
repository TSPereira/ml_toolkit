from pandas import DataFrame
import torch
import torch.nn as nn

from .encoder import Encoder, EncoderLayer, ConvLayer
from .decoder import Decoder, DecoderLayer
from .attn import FullAttention, ProbAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding
from .dataset import InformerTemporalDataset

from ..core import Engine
from .....utils.os_utl import check_options, check_types


class Informer(Engine):
    @check_options(attn=('prob', 'full'))
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, factor=5, d_model=512, n_heads=8,
                 e_layers=3, d_layers=2, d_ff=512, dropout=0.0, attn='prob', embed='fixed', time_tick='M',
                 activation='gelu', enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                 device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.enc_seq_len = seq_len
        self.dec_seq_len = label_len
        self.pred_len = out_len
        self.attn = attn
        self.enc_self_mask = enc_self_mask
        self.dec_self_mask = dec_self_mask
        self.dec_enc_mask = dec_enc_mask
        self._time_tick = time_tick

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, time_tick, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, time_tick, dropout)

        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention

        # Encoder
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(Attn(False, factor, attention_dropout=dropout), d_model, n_heads),
                d_model, d_ff, dropout=dropout, activation=activation) for _ in range(e_layers)],
            [ConvLayer(d_model) for _ in range(e_layers-1)],
            norm_layer=torch.nn.LayerNorm(d_model))

        # Decoder
        self.decoder = Decoder(
            [DecoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout), d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout), d_model, n_heads),
                    d_model, d_ff, dropout=dropout, activation=activation) for _ in range(d_layers)],
            norm_layer=torch.nn.LayerNorm(d_model))

        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.double()

    def _init_weights(self, mod):
        if not isinstance(mod, (TokenEmbedding, nn.LayerNorm, nn.BatchNorm1d)):
            super(Informer, self)._init_weights(mod)

    # noinspection PyMethodOverriding
    @check_types(dataset=DataFrame)
    def create_dataloader(self, dataset, target, scale=True, val_size=0.1, batch_size=10, shuffle=False, **kwargs):
        if not isinstance(target, (list, tuple, set)):
            target = [target]

        if len(target) != self.projection.out_features:
            raise KeyError(f'Number of "target"s ({len(target)}) passed does not match the output size used '
                           f'({self.projection.out_features}).')

        if not all(x in dataset.columns for x in target):
            raise KeyError(f'Not all "target"s passed are in the dataframe provided.')

        dataset = InformerTemporalDataset(dataset, self.enc_seq_len, self.dec_seq_len, self.pred_len, target,
                                          self._time_tick, scale)
        super().create_dataloader(dataset, val_size, batch_size, shuffle, random_split=False, **kwargs)

    def forward(self, x_enc, x_dec, x_mark_enc, x_mark_dec):
        # convert everything to double
        x_enc = x_enc.double()
        x_dec = x_dec.double()
        x_mark_enc = x_mark_enc.double()
        x_mark_dec = x_mark_dec.double()

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=self.enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=self.dec_self_mask, cross_mask=self.dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
