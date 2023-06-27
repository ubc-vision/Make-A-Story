# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
import numpy as np

import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.gpt.utils import shift_dim, view_range, tensor_slice


class P2QAttentionStack(nn.Module):
    def __init__(self, embd_dim, n_head, n_layer, dropout, attn_dropout,
                 video_shape=None, audio_seq_len=None, text_seq_len=None,
                 modality_p = None, modality_q = None
                 ):
        super().__init__()
        self.modality_p = modality_p
        self.modality_q = modality_q
        self.embd_dim = embd_dim
        self.right_shift = RightShift(embd_dim)
        total_len = 0
        if text_seq_len is not None:
            self.text_seq_len = text_seq_len
            self.text_pos_emb = nn.Embedding(text_seq_len, embd_dim)
            total_len += text_seq_len
        if audio_seq_len is not None:
            self.audio_seq_len = audio_seq_len
            self.audio_pos_emb = nn.Embedding(audio_seq_len, embd_dim)
            total_len += audio_seq_len
        if video_shape is not None:
            self.video_pos_emb = AddBroadcastPosEmbed(shape=video_shape, embd_dim=embd_dim)
            self.video_seq_len = np.prod(video_shape)
            self.decode_idxs = list(itertools.product(*[range(s) for s in video_shape]))
            total_len += self.video_seq_len

        self.attn_nets = nn.ModuleList(
            [
                AttentionBlock(
                    shape=[total_len],
                    embd_dim=embd_dim,
                    n_head=n_head,
                    n_layer=n_layer,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for i in range(n_layer)
            ]
        )

    def get_inference_embeddings(self, x, modality, decode_step):
        if modality == 'video':
            x += self.video_pos_emb(x.unsqueeze(1).unsqueeze(1), decode_step, self.decode_idxs[decode_step - 1])
        elif modality == 'audio':
            x += self.audio_pos_emb(torch.arange(decode_step, device=x.device))[decode_step - 1]
        elif modality == 'text':
            x += self.text_pos_emb(torch.arange(decode_step, device=x.device))[decode_step - 1]
        return x

    def get_training_embeddings(self, x, modality, decode_step, decode_idx):
        if modality == 'video':
            x += self.video_pos_emb(x, decode_step, decode_idx)
        elif modality == 'audio':
            x += self.audio_pos_emb(torch.arange(self.audio_seq_len, device=x.device))
        elif modality == 'text':
            x += self.text_pos_emb(torch.arange(self.text_seq_len, device=x.device))
        return x

    def forward(self, x_p, x_q, decode_step=None, decode_idx=None):
        if x_q is None:
            if decode_step > 0:
                x_p = self.get_inference_embeddings(x_p, self.modality_p, decode_step)
            x = x_p
        elif x_p is None:
            p_seq_len = getattr(self, f"{self.modality_p}_seq_len")
            x = self.get_inference_embeddings(x_q, self.modality_q, decode_step - p_seq_len)
        else:
            x_p = self.get_training_embeddings(x_p, self.modality_p, decode_step, decode_idx)
            x_q = self.get_training_embeddings(x_q, self.modality_q, decode_step, decode_idx)
            x = torch.cat((x_p, x_q), 1)
        x = self.right_shift(x, decode_step)
        for net in self.attn_nets:
            x = net(x, decode_step, decode_idx)

        return x


class AttentionBlock(nn.Module):
    def __init__(self, shape, embd_dim, n_head, n_layer, dropout, attn_dropout, attn_type='full', causal=True):
        super().__init__()
        self.pre_attn_norm = LayerNorm(embd_dim)
        self.post_attn_dp = nn.Dropout(dropout)
        self.attn = MultiHeadAttention(shape, embd_dim, embd_dim, n_head, n_layer, causal=causal, attn_type=attn_type,
                                       attn_kwargs=dict(attn_dropout=attn_dropout))

        self.pre_fc_norm = LayerNorm(embd_dim)
        self.post_fc_dp = nn.Dropout(dropout)
        self.fc_block = nn.Sequential(
            nn.Linear(in_features=embd_dim, out_features=embd_dim * 4),
            GeLU2(),
            nn.Linear(in_features=embd_dim * 4, out_features=embd_dim),
        )

    def forward(self, x, decode_step, decode_idx):
        h = self.pre_attn_norm(x)
        if self.training:
            h = checkpoint(self.attn, h, h, h, decode_step, decode_idx)
        else:
            h = self.attn(h, h, h, decode_step, decode_idx)
        h = self.post_attn_dp(h)
        x = x + h

        h = self.pre_fc_norm(x)
        if self.training:
            h = checkpoint(self.fc_block, h)
        else:
            h = self.fc_block(h)
        h = self.post_fc_dp(h)
        x = x + h

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, shape, dim_q, dim_kv, n_head, n_layer, causal, attn_type, attn_kwargs):
        super().__init__()
        self.causal = causal
        self.shape = shape

        self.d_k = dim_q // n_head
        self.d_v = dim_kv // n_head
        self.n_head = n_head
        self.w_qs = nn.Linear(dim_q, n_head * self.d_k, bias=False) # q
        self.w_qs.weight.data.normal_(std=1.0 / np.sqrt(dim_q))

        self.w_ks = nn.Linear(dim_kv, n_head * self.d_k, bias=False) # k
        self.w_ks.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.w_vs = nn.Linear(dim_kv, n_head * self.d_v, bias=False) # v
        self.w_vs.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.fc = nn.Linear(n_head * self.d_v, dim_q, bias=True) # c
        self.fc.weight.data.normal_(std=1.0 / np.sqrt(dim_q * n_layer))

        if attn_type == 'full':
            self.attn = FullAttention(shape, causal, **attn_kwargs)
        elif attn_type == 'axial':
            assert not causal, 'causal axial attention is not supported'
            self.attn = AxialAttention(len(shape), **attn_kwargs)

        self.cache = None

    def forward(self, q, k, v, decode_step=None, decode_idx=None):
        """ Compute multi-head attention
        Args
            q, k, v: a [b, d1, ..., dn, c] tensor or
                     a [b, 1, ..., 1, c] tensor if decode_step is not None

        Returns
            The output after performing attention
        """

        # compute k, q, v
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q = view_range(self.w_qs(q), -1, None, (n_head, d_k))
        k = view_range(self.w_ks(k), -1, None, (n_head, d_k))
        v = view_range(self.w_vs(v), -1, None, (n_head, d_v))

        # b x n_head x seq_len x d
        # (b, *d_shape, n_head, d) ->  (b, n_head, *d_shape, d)
        q = shift_dim(q, -2, 1)
        k = shift_dim(k, -2, 1)
        v = shift_dim(v, -2, 1)

        # fast decoding
        if decode_step is not None:
            if decode_step == 0:
                if self.causal:
                    k_shape = (q.shape[0], n_head, *self.shape, self.d_k) # B x n_head x 3 x 32 x 32 x d
                    v_shape = (q.shape[0], n_head, *self.shape, self.d_v)
                    self.cache = dict(k=torch.zeros(k_shape, dtype=k.dtype, device=q.device),
                                    v=torch.zeros(v_shape, dtype=v.dtype, device=q.device))
                else:
                    # cache only once in the non-causal case
                    self.cache = dict(k=k.clone(), v=v.clone())
            if self.causal:
                idx = (slice(None, None), slice(None, None), *[slice(i, i+ 1) for i in decode_idx])
                self.cache['k'][idx] = k
                self.cache['v'][idx] = v
            k, v = self.cache['k'], self.cache['v']

        a = self.attn(q, k, v, decode_step, decode_idx)

        # (b, *d_shape, n_head, d) -> (b, *d_shape, n_head * d)
        a = shift_dim(a, 1, -2).flatten(start_dim=-2)
        a = self.fc(a) # (b x seq_len x embd_dim)

        return a

############## Attention #######################
class FullAttention(nn.Module):
    def __init__(self, shape, causal, attn_dropout):
        super().__init__()
        self.causal = causal
        self.attn_dropout = attn_dropout

        seq_len = np.prod(shape)
        if self.causal:
            self.register_buffer('mask', torch.tril(torch.ones(seq_len, seq_len)))

    def forward(self, q, k, v, decode_step, decode_idx):
        mask = self.mask if self.causal else None
        if decode_step is not None and mask is not None:
            mask = mask[[decode_step]]

        elif mask is not None and q.size(2) < mask.size(0):
            mask = mask[range(q.size(2)),:][:, range(q.size(2))]


        old_shape = q.shape[2:-1]
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        out = scaled_dot_product_attention(q, k, v, mask=mask,
                                           attn_dropout=self.attn_dropout,
                                           training=self.training)

        return view_range(out, 2, 3, old_shape)

class AxialAttention(nn.Module):
    def __init__(self, n_dim, axial_dim):
        super().__init__()
        if axial_dim < 0:
            axial_dim = 2 + n_dim + 1 + axial_dim
        else:
            axial_dim += 2 # account for batch, head, dim
        self.axial_dim = axial_dim

    def forward(self, q, k, v, decode_step, decode_idx):
        q = shift_dim(q, self.axial_dim, -2).flatten(end_dim=-3)
        k = shift_dim(k, self.axial_dim, -2).flatten(end_dim=-3)
        v = shift_dim(v, self.axial_dim, -2)
        old_shape = list(v.shape)
        v = v.flatten(end_dim=-3)

        out = scaled_dot_product_attention(q, k, v, training=self.training)
        out = out.view(*old_shape)
        out = shift_dim(out, -2, self.axial_dim)
        return out

################ Spatiotemporal broadcasted positional embeddings ###############
class AddBroadcastPosEmbed(nn.Module):
    def __init__(self, shape, embd_dim, dim=-1):
        super().__init__()
        assert dim in [-1, 1] # only first or last dim supported
        self.shape = shape
        self.n_dim = n_dim = len(shape)
        self.embd_dim = embd_dim
        self.dim = dim

        assert embd_dim % n_dim == 0, f"{embd_dim} % {n_dim} != 0"
        self.emb = nn.ParameterDict({
             f'd_{i}': nn.Parameter(torch.randn(shape[i], embd_dim // n_dim) * 0.01
                                    if dim == -1 else
                                    torch.randn(embd_dim // n_dim, shape[i]) * 0.01)
             for i in range(n_dim)
        })

    def forward(self, x, decode_step=None, decode_idx=None):
        embs = []
        for i in range(self.n_dim):
            e = self.emb[f'd_{i}']
            if self.dim == -1:
                # (1, 1, ..., 1, self.shape[i], 1, ..., -1)
                e = e.view(1, *((1,) * i), self.shape[i], *((1,) * (self.n_dim - i - 1)), -1)
                e = e.expand(1, *self.shape, -1)
            else:
                e = e.view(1, -1, *((1,) * i), self.shape[i], *((1,) * (self.n_dim - i - 1)))
                e = e.expand(1, -1, *self.shape)
            embs.append(e)

        embs = torch.cat(embs, dim=self.dim)
        if decode_step is not None:
            embs = tensor_slice(embs, [0, *decode_idx, 0],
                                [x.shape[0], *(1,) * self.n_dim, x.shape[-1]])

        return embs.flatten(start_dim=1, end_dim=-2)

################# Helper Functions ###################################
def scaled_dot_product_attention(q, k, v, mask=None, attn_dropout=0., training=True):
    # Performs scaled dot-product attention over the second to last dimension dn

    # (b, n_head, d1, ..., dn, d)
    attn = torch.matmul(q, k.transpose(-1, -2))
    attn = attn / np.sqrt(q.shape[-1])
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float('-inf'))
    attn_float = F.softmax(attn, dim=-1)
    attn = attn_float.type_as(attn) # b x n_head x d1 x ... x dn x d
    attn = F.dropout(attn, p=attn_dropout, training=training)

    a = torch.matmul(attn, v) # b x n_head x d1 x ... x dn x d

    return a


class RightShift(nn.Module):
    def __init__(self, embd_dim):
        super().__init__()
        self.embd_dim = embd_dim
        self.sos = nn.Parameter(torch.FloatTensor(embd_dim).normal_(std=0.02), requires_grad=True)

    def forward(self, x, decode_step):
        if decode_step is not None and decode_step > 0:
            return x

        x_shape = list(x.shape)
        x = x.flatten(start_dim=1, end_dim=-2) # (b, seq_len, embd_dim)
        sos = torch.ones(x_shape[0], 1, self.embd_dim, dtype=torch.float32).to(self.sos) * self.sos
        sos = sos.type_as(x)
        x = torch.cat([sos, x[:, :-1, :]], axis=1)
        x = x.view(*x_shape)
        return x


class GeLU2(nn.Module):
    def forward(self, x):
        return (1.702 * x).sigmoid() * x


class LayerNorm(nn.Module):
    def __init__(self, embd_dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(embd_dim, dtype=torch.float32), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(embd_dim, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        g = self.g  # (embd_dim,)
        b = self.b

        x_float = x.float()

        mu = x_float.mean(dim=-1, keepdims=True)
        s = (x_float - mu).square().mean(dim=-1, keepdims=True)
        x_float = (x_float - mu) * (1e-5 + s.rsqrt())  # (b, ..., embd_dim)
        x_float = x_float * g + b

        x = x_float.type_as(x)
        return x
