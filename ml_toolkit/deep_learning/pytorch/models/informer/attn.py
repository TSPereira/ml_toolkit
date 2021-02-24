from math import sqrt

import numpy as np
import torch
import torch.nn as nn

from ...tools import TriangularCausalMask, ProbMask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        b, l, h, e = queries.shape
        _, s, _, d = values.shape
        scale = self.scale or 1./sqrt(e)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(b, l, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        a = self.dropout(torch.softmax(scale * scores, dim=-1))
        v = torch.einsum("bhls,bshd->blhd", a, values)
        return v.contiguous()


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    @staticmethod
    def _prob_qk(q, k, sample_k, n_top):
        # Q [B, H, L, D]
        b, h, l, e = k.shape
        _, _, s, _ = q.shape

        # calculate the sampled Q_K
        k_expand = k.unsqueeze(-3).expand(b, h, s, l, e)
        indx_sample = torch.randint(l, (s, sample_k))
        k_sample = k_expand[:, :, torch.arange(s).unsqueeze(1), indx_sample, :]
        q_k_sample = torch.matmul(q.unsqueeze(-2), k_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        m = q_k_sample.max(-1)[0] - torch.div(q_k_sample.sum(-1), l)
        m_top = m.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        q_reduce = q[torch.arange(b)[:, None, None],
                     torch.arange(h)[None, :, None],
                     m_top, :]
        q_k = torch.matmul(q_reduce, k.transpose(-2, -1))
        return q_k, m_top

    def _get_initial_context(self, v, l_q):
        b, h, l_v, d = v.shape

        if not self.mask_flag:
            v_sum = v.sum(dim=-2)
            context = v_sum.unsqueeze(-2).expand(b, h, l_q, v_sum.shape[-1]).clone()

        else:  # use mask
            assert(l_q == l_v)  # requires that L_Q == L_V, i.e. for self-attention only
            context = v.cumsum(dim=-1)

        return context

    def _update_context(self, context_in, v, scores, index, l_q, attn_mask):
        b, h, l_v, d = v.shape

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = ProbMask(b, h, l_q, index, scores, device=v.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)
        context_in[torch.arange(b)[:, None, None],
                   torch.arange(h)[None, :, None],
                   index, :] = torch.matmul(attn, v)

        return context_in

    def forward(self, queries, keys, values, attn_mask):
        b, l, h, d = queries.shape
        _, s, _, _ = keys.shape

        queries = queries.view(b, h, l, -1)
        keys = keys.view(b, h, s, -1)
        values = values.view(b, h, s, -1)

        U = self.factor * np.ceil(np.log(s)).astype('int').item()
        u = self.factor * np.ceil(np.log(l)).astype('int').item()
        
        scores_top, index = self._prob_qk(queries, keys, u, U)
        scale = self.scale or 1./sqrt(d)  # add scale factor
        if scale is not None:
            scores_top = scores_top * scale

        # get the context and update the context with selected top_k queries
        context = self._get_initial_context(values, l)
        context = self._update_context(context, values, scores_top, index, l, attn_mask)
        return context.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        b, l, _ = queries.shape
        _, s, _ = keys.shape
        h = self.n_heads

        queries = self.query_projection(queries).view(b, l, h, -1)
        keys = self.key_projection(keys).view(b, s, h, -1)
        values = self.value_projection(values).view(b, s, h, -1)

        out = self.inner_attention(queries, keys, values, attn_mask).view(b, l, -1)
        return self.out_projection(out)
