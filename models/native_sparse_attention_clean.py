from __future__ import annotations

from copy import deepcopy
from math import ceil
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, arange, stack, cat, tensor, Tensor
from torch.nn import Module, ModuleList

# einstein notation

import einx
from einops import einsum, repeat, rearrange, reduce, pack, unpack
from einops.layers.torch import Rearrange

# b - batch
# h - heads
# qh - grouped query heads
# n - sequence (token level or compressed)
# w - windows, for fine or compressed
# i, j - query / key sequence
# d - feature dimension
# s - strategies

# flex attention
# https://pytorch.org/blog/flexattention/

flex_attention = None

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass

# flex attn sliding attention mask


def create_sliding_mask(seq_len, window_size):

    def sliding_mask(_, __, q_idx, kv_idx):

        distance = q_idx - kv_idx
        backward_sliding_mask = distance <= window_size
        forward_sliding_mask = distance >= -window_size

        return backward_sliding_mask & forward_sliding_mask

    block_mask = create_block_mask(sliding_mask, B = None, H = None, Q_LEN = seq_len, KV_LEN = seq_len, _compile = True)
    return block_mask

def create_compress_mask(seq_len, kv_seq_len, compress_block_sliding_stride, mem_kv_len = 0):
    return None


def create_fine_mask(seq_len, fine_block_size):

    def inner(selected_block_indices: Tensor, num_grouped_queries = 1):
        device = selected_block_indices.device
        batch, kv_heads = selected_block_indices.shape[:2]

        one_hot_selected_block_indices = torch.zeros((*selected_block_indices.shape[:-1], seq_len // fine_block_size), device = device, dtype = torch.bool)
        one_hot_selected_block_indices.scatter_(-1, selected_block_indices, True)

        def fine_mask(b_idx, h_idx, q_idx, kv_idx):

            compressed_q_idx = q_idx // fine_block_size
            compressed_kv_idx = kv_idx // fine_block_size
            kv_head_idx = h_idx // num_grouped_queries

            is_selected = one_hot_selected_block_indices[b_idx, kv_head_idx, q_idx, compressed_kv_idx]

            return is_selected

        block_mask = create_block_mask(fine_mask, B = batch, H = kv_heads * num_grouped_queries, Q_LEN = seq_len, KV_LEN = seq_len, _compile = True)
        return block_mask

    return inner

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def round_down_mult(n, mult):
    return n // mult * mult

def round_up_mult(n, mult):
    return ceil(n / mult) * mult

def divisible_by(num, den):
    return (num % den) == 0

def is_empty(t):
    return t.numel() == 0

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def pack_one_with_inverse(t, pattern):
    packed, ps = pack([t], pattern)
    def inverse(out):
        return unpack(out, ps, pattern)[0]

    return packed, inverse

# tensor helpers

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def straight_through(t, target):
    return t + (target - t).detach()

# attend function

def attend(
    q, k, v,
    mask = None,
    return_sim = False,
    scale = None
):
    scale = default(scale, q.shape[-1] ** -0.5)

    q_heads, k_heads = q.shape[1], k.shape[1]
    num_grouped_queries = q_heads // k_heads

    q = rearrange(q, 'b (h qh) ... -> b h qh ...', qh = num_grouped_queries)

    sim = einsum(q, k, 'b h qh i d, b h j d -> b h qh i j') * scale

    mask_value = max_neg_value(sim)

    if exists(mask):
        sim = sim.masked_fill(~mask, mask_value // 10)

    attn = sim.softmax(dim = -1)

    attn_out = einsum(attn, v, 'b h qh i j, b h j d -> b h qh i d')

    attn_out = rearrange(attn_out, 'b h qh ... -> b (h qh) ...')

    if not return_sim:
        return attn_out

    sim = rearrange(sim, 'b h qh ... -> b (h qh) ...')

    return attn_out, sim

# classes

class SparseAttention(Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
        compress_block_size,
        compress_block_sliding_stride,
        selection_block_size,
        num_selected_blocks,
        kv_heads = None,
        num_compressed_mem_kv = 1,
        norm = True,
        use_diff_topk = False,
        query_heads_share_selected_kv = True, # if set to True, importance score is averaged across query heads to select top-n buckets of kv per kv head - but can be set to False for each query head within a group to look at different sets of kv buckets. will be more memory and compute of course
        compress_mlp: Module | None = None,
        compress_mlp_expand_factor = 1.,
        strategy_combine_mlp: Module | None = None
    ):
        super().__init__()

        # attention heads
        # handling gqa if `kv_heads` is set

        kv_heads = default(kv_heads, heads)
        assert kv_heads <= heads and divisible_by(heads, kv_heads)

        self.heads = heads
        self.dim_head = dim_head
        self.kv_heads = kv_heads
        self.num_grouped_queries = heads // kv_heads

        # scale

        self.scale = dim_head ** -0.5

        dim_inner = dim_head * heads
        dim_kv_inner = dim_head * kv_heads

        self.norm = nn.RMSNorm(dim) if norm else nn.Identity()

        # qkv

        qkv_split = (dim_inner, dim_kv_inner, dim_kv_inner)

        self.to_qkv = nn.Linear(dim, sum(qkv_split), bias = False)

        self.qkv_split = qkv_split

        # compress strategy

        self.compress_block_size = compress_block_size
        self.compress_block_sliding_stride = compress_block_sliding_stride
        assert self.compress_block_size >= self.compress_block_sliding_stride, 'compress_block_size must be >= compress_block_sliding_stride'
        assert self.compress_block_sliding_stride > 0, 'compress_block_sliding_stride must be greater than 0'
        assert divisible_by(selection_block_size, self.compress_block_sliding_stride), f'selection_block_size {selection_block_size} must be divisible by compress_block_sliding_stride {self.compress_block_sliding_stride}'

        # Compression window splitting
        self.split_compress_window = nn.Sequential(
            Rearrange('b h n d -> (b h) d 1 n'),
            nn.ZeroPad2d(((compress_block_size - compress_block_sliding_stride), 0, 0, 0)),
            nn.Unfold(kernel_size=(1, self.compress_block_size), stride=(1, self.compress_block_sliding_stride)),
            Rearrange('(b h) (d n) w -> b h w n d', d=dim_head, h=kv_heads, n=self.compress_block_size)
        )

        assert num_compressed_mem_kv > 0
        self.num_mem_compress_kv = num_compressed_mem_kv
        self.compress_mem_kv = nn.Parameter(torch.zeros(2, kv_heads, num_compressed_mem_kv, dim_head))
        
        self.k_intrablock_positions = nn.Parameter(torch.zeros(kv_heads, self.compress_block_size, dim_head))
        self.v_intrablock_positions = nn.Parameter(torch.zeros(kv_heads, self.compress_block_size, dim_head))

        if not exists(compress_mlp):
            compress_dim = self.compress_block_size * dim_head
            compress_mlp_dim_hidden = int(compress_mlp_expand_factor * compress_dim)

            compress_mlp = nn.Sequential(
                Rearrange('b h w n d -> b h w (n d)'),
                nn.Linear(compress_dim, compress_mlp_dim_hidden),
                nn.ReLU(),
                nn.Linear(compress_mlp_dim_hidden, dim_head),
            )

        self.k_compress = deepcopy(compress_mlp)
        self.v_compress = deepcopy(compress_mlp)

        # local window strategy

        self.local_attention_mask = None
        self.sigma_att = nn.Parameter(-1 + 0.01 * torch.randn((1, heads, 1, 1)))

        # selection related

        self.use_diff_topk = use_diff_topk
        self.query_heads_share_selected_kv = query_heads_share_selected_kv
        self.selection_block_size = selection_block_size

        assert num_selected_blocks >= 0

        if num_selected_blocks == 0:
            print(f'`num_selected_blocks` should be set greater than 0, unless if you are ablating it for experimental purposes')

        self.num_selected_blocks = num_selected_blocks

        # they combine the three sparse branches through a learned combine with sigmoid activation

        if not exists(strategy_combine_mlp):
            strategy_combine_mlp = nn.Linear(dim, 3 * heads)

            # init to sliding windows first, as network tends to pick up on local patterns first before distant ones

            nn.init.zeros_(strategy_combine_mlp.weight)
            strategy_combine_mlp.bias.data.copy_(tensor([-2., -2., 2.] * heads))

        self.to_strategy_combine = nn.Sequential(
            strategy_combine_mlp,
            nn.Sigmoid(),
            Rearrange('b n (h s) -> b h n s', h = heads)
        )

        # split and merging heads

        self.split_heads = Rearrange('b n (h d) -> b h n d', d = dim_head)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        # combining heads

        self.combine_heads = nn.Linear(dim_inner, dim, bias = False)

    @torch.no_grad()
    def create_attention_mask(self, pos: torch.Tensor):
        """ Distance-based attention bias (eq. 10). """
        pos = rearrange(pos, '(n m) d -> n m d', m=self.compress_block_size)
        return self.sigma_att * torch.cdist(pos, pos, p=2).unsqueeze(1)

    def local_attention_mask_mod(self, batch, head, q_idx, kv_idx):
        # True if query and key indices lie in the same B-sized block
        return (q_idx // self.compress_block_size) == (kv_idx // self.compress_block_size)


    def pos_bias_score_mod(self, score, b, h, q_idx, kv_idx):
        return score + pos_attn_bias

    def forward(
        self,
        inp,
        pos,
        sliding_window_flex_mask = None,
        fine_selection_flex_mask = None,
    ):

        batch, seq_len, scale, heads, kv_heads, device = *inp.shape[:2], self.scale, self.heads, self.kv_heads, inp.device

        if not exists(self.local_attention_mask): 
            self.local_attention_mask = create_block_mask(
                self.local_attention_mask_mod,
                B=None,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                BLOCK_SIZE=self.compress_block_size,
                device=device
            )
            self.local_attn = partial(flex_attention, block_mask=self.local_attention_mask, enable_gqa=True)

        compress_divisible_seq_len = round_down_mult(seq_len, self.compress_block_sliding_stride)
        num_compress_blocks = compress_divisible_seq_len // self.compress_block_sliding_stride

        compress_overlap_len = self.compress_block_size - self.compress_block_sliding_stride
        has_compress_overlap = compress_overlap_len > 0

        fine_divisible_seq_len = round_up_mult(seq_len, self.selection_block_size)
        num_fine_blocks = fine_divisible_seq_len // self.selection_block_size

        # maybe prenorm

        inp = self.norm(inp)

        # queries, keys, values

        q, k, v = self.to_qkv(inp).split(self.qkv_split, dim = -1)

        q, k, v = map(self.split_heads, (q, k, v))

        # compressed key / values - variables prepended with `c` stands for compressed

        k_compress_input, v_compress_input = k[..., :compress_divisible_seq_len, :], v[..., :compress_divisible_seq_len, :]

        if not is_empty(k_compress_input):
            k_compress_input = self.split_compress_window(k_compress_input)
            v_compress_input = self.split_compress_window(v_compress_input)
        else:
            k_compress_input, v_compress_input = tuple(t.reshape(batch, kv_heads, 0, self.compress_block_size, self.dim_head) for t in (k_compress_input, v_compress_input))

        # add the intra block positions

        if not is_empty(k_compress_input):
            k_compress_input = einx.add('b h w n d, h n d', k_compress_input, self.k_intrablock_positions)
            v_compress_input = einx.add('b h w n d, h n d', v_compress_input, self.v_intrablock_positions)

        run_k, run_v = k, v


        run_k = run_k[..., compress_divisible_seq_len:, :]
        run_v = run_v[..., compress_divisible_seq_len:, :]

        cq = q
        ck = self.k_compress(k_compress_input)   # Equation (7) of the Native Sparse Attention paper
        cv = self.v_compress(v_compress_input)


        # 1. coarse attention over compressed

        mem_ck, mem_cv = repeat(self.compress_mem_kv, 'kv ... -> kv b ...', b = batch)

        num_mem_compress_kv = mem_ck.shape[-2]

        ck = cat((mem_ck, ck), dim = -2)
        cv = cat((mem_cv, cv), dim = -2)

        # compressed masking


        compressed_attn_out, csim = attend(cq, ck, cv, return_sim = True)

        # for 2. and 3., will give them relative positions with rotary - compressed needs to be handled separately (even if they already have intra block absolute positions)

        # 2. fine attention over selected based on compressed attention logits - variables prepended with `f` stands for the fine attention pathway

        importance_scores = csim[..., num_mem_compress_kv:]

        num_selected = min(self.num_selected_blocks, num_compress_blocks)
        has_selected_kv_for_fine_attn = num_selected > 0

        # maybe average the compressed attention across each grouped queries (per key / values)

        if self.query_heads_share_selected_kv:
            importance_scores = reduce(importance_scores, 'b (h grouped_queries) ... -> b h ...', 'mean', grouped_queries = self.num_grouped_queries)

            fine_num_grouped_queries = self.num_grouped_queries
        else:
            fine_num_grouped_queries = 1

        # handle if compress block size does not equal to the fine block size
        # cannot parse their equation, so will just improvise
        # first we expand all the compressed scores to the full sequence length, then average within each fine / selection block size - pad on the right to 0s, which should be fine as sliding window convers the local anyways

        if has_selected_kv_for_fine_attn:

            if self.compress_block_sliding_stride != self.selection_block_size:

                num_compress_per_fine = self.selection_block_size // self.compress_block_sliding_stride

                round_down_score_len = round_down_mult(importance_scores.shape[-1], num_compress_per_fine)
                importance_scores = importance_scores[..., :round_down_score_len]

                if not is_empty(importance_scores):
                    importance_scores = reduce(importance_scores, '... (j num_compress_per_fine) -> ... j', 'mean', num_compress_per_fine = num_compress_per_fine)

                    i, j = importance_scores.shape[-2:]

                    # mask out block diagonal

                    q_seq = arange(i, device = device) // self.selection_block_size
                    k_seq = arange(j, device = device)

                    block_diagonal_mask = einx.equal('i, j -> i j', q_seq, k_seq)

                    importance_scores = importance_scores.masked_fill(block_diagonal_mask, max_neg_value(csim))

            importance_scores = F.pad(importance_scores, (1, 0), value = -1e3)
            importance_scores = importance_scores.softmax(dim = -1)
            importance_scores = importance_scores[..., 1:]

        # handle if number of total blocks is less than number to select for fine attention

        fq = q
        fk = k
        fv = v

        num_selected = min(num_selected, importance_scores.shape[-1])
        has_selected_kv_for_fine_attn = num_selected > 0

        remainder = fine_divisible_seq_len - seq_len
        pad_to_multiple = partial(pad_at_dim, pad = (0, remainder), dim = -2)

        if has_selected_kv_for_fine_attn:

            # get the top-n kv segments for fine attention

            selected_importance_values, selected_block_indices = importance_scores.topk(num_selected, dim = -1)

            gates = straight_through(selected_importance_values, 1.) if self.use_diff_topk else None

            if exists(fine_selection_flex_mask):
                assert not self.use_diff_topk, 'differential topk is not available for flex attention'

                # flex attention for the selection for fine attention

                fine_block_mask = fine_selection_flex_mask(selected_block_indices, num_grouped_queries = fine_num_grouped_queries)

                fine_attn_out = flex_attention(fq, fk, fv, block_mask = fine_block_mask, enable_gqa = True)

            else:
                fmask = selected_importance_values > 1e-10

                if seq_len < fine_divisible_seq_len:
                    fk, fv, fq = map(pad_to_multiple, (fk, fv, fq))

                    fmask = pad_at_dim(fmask, (0, remainder), value = False, dim = -2)

                    selected_block_indices = pad_at_dim(selected_block_indices, (0, remainder), value = 0, dim = -2)

                    if exists(gates):
                        gates = pad_at_dim(gates, (0, remainder), value = 0, dim = -2)

                
                fmask = repeat(fmask, 'b h i w -> b h 1 i (w j)', j = self.selection_block_size)

                # select out the spatial crops of keys / values for fine attention

                fk = rearrange(fk, 'b h (w n) d -> b h w n d', w = num_fine_blocks)
                fv = rearrange(fv, 'b h (w n) d -> b h w n d', w = num_fine_blocks)

                # get_at("b h [w] j d, b h i selected -> b h i selected j d", fkv, selected_block_indices)

                if self.query_heads_share_selected_kv:
                    fk = repeat(fk, 'b h w j d -> b h i w j d', i = selected_block_indices.shape[2])
                    fv = repeat(fv, 'b h w j d -> b h i w j d', i = selected_block_indices.shape[2])
                else:
                    fk = repeat(fk, 'b h w j d -> b (h qh) i w j d', i = selected_block_indices.shape[2], qh = self.num_grouped_queries)
                    fv = repeat(fv, 'b h w j d -> b (h qh) i w j d', i = selected_block_indices.shape[2], qh = self.num_grouped_queries)

                selected_block_indices = repeat(selected_block_indices, 'b h i sel -> b h i sel j d', j = fk.shape[-2], d = fk.shape[-1])

                fk = fk.gather(3, selected_block_indices)
                fv = fv.gather(3, selected_block_indices)

                # differential topk gating

                if self.use_diff_topk:
                    

                    fk = einx.multiply('b h i sel, b h i sel j d -> b h i sel j d', gates, fk)

                # merge selected key values

                fk, fv = tuple(rearrange(t, 'b h i w j d -> b h i (w j) d') for t in (fk, fv))

                # fine attention

                fq = rearrange(fq, 'b (h qh) ... -> b h qh ...', qh = fine_num_grouped_queries)

                fsim = einsum(fq, fk, 'b h qh i d, b h i j d -> b h qh i j') * self.scale

                mask_value = max_neg_value(fsim)

                fsim = fsim.masked_fill(~fmask, mask_value)

                fattn = fsim.softmax(dim = -1)

                fine_attn_out = einsum(fattn, fv, 'b h qh i j, b h i j d -> b h qh i d')

                fine_attn_out = rearrange(fine_attn_out, 'b h qh ... -> b (h qh) ...')

                fine_attn_out = fine_attn_out[..., :seq_len, :]

        else:
            # if only first block, just do a simple block attention

            seq_len = fk.shape[-2]
            fmask = None

            fk, fv, fq = map(pad_to_multiple, (fk, fv, fq))

            fq, fk, fv = tuple(rearrange(t, 'b h (w n) d -> (b w) h n d', n = self.selection_block_size) for t in (fq, fk, fv))

            
            fine_attn_out = attend(fq, fk, fv, mask = fmask)

            fine_attn_out = rearrange(fine_attn_out, '(b w) h n d -> b h (w n) d', b = batch)
            fine_attn_out = fine_attn_out[..., :seq_len, :]

        # 3. overlapping sliding window, this is unsurprising and expected - `s` for sliding

        sq = q
        sk = k
        sv = v


        pos_attn_bias = self.create_attention_mask(pos)
        local_attn_out = self.local_attn(sq, sk, sv)

        # combine strategies

        strategy_weighted_combine = self.to_strategy_combine(inp)

        out = einsum(strategy_weighted_combine, stack([compressed_attn_out, fine_attn_out, local_attn_out]), 'b h n s, s b h n d -> b h n d')

        # merge heads and combine them

        out = self.merge_heads(out)

        out = self.combine_heads(out)

        
        return out

        
