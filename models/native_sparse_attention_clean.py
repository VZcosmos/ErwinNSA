from __future__ import annotations
 
# ──────────────────────────────────────────────────────────────────────────────
# Std-lib & typing
# ──────────────────────────────────────────────────────────────────────────────
from copy import deepcopy
from functools import partial
from math import ceil
from typing import Callable, Tuple
 
# ──────────────────────────────────────────────────────────────────────────────
# Third-party
# ──────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
from torch import Tensor, arange, cat, stack, nn
from torch.nn import Module, ModuleList, Linear, Identity, RMSNorm, ZeroPad2d
 
from einops import einsum, rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange

import einx
 
from native_sparse_attention.ops.parallel import ParallelNSAFunction
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════
def exists(v) -> bool: return v is not None
def default(v, d):    return v if exists(v) else d
def round_down_mult(n: int, k: int) -> int: return n // k * k
def round_up_mult(n: int, k: int) -> int:   return ceil(n / k) * k
def divisible_by(n: int, k: int) -> bool:   return (n % k) == 0
def is_empty(t: Tensor) -> bool:            return t.numel() == 0
def max_neg_value(t: Tensor) -> Tensor:     return -torch.finfo(t.dtype).max
 
def pad_at_dim(t: Tensor, pad: tuple[int, int], *, dim: int = -1, value: float = 0.):
    """Zero/right-pad *t* at *dim* by (*pad_left*, *pad_right*)."""
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)
 
def straight_through(src: Tensor, tgt: Tensor) -> Tensor:
    """Straight-through estimator."""
    return src + (tgt - src).detach()
 
def pack_one_with_inverse(t: Tensor, pattern: str) -> Tuple[Tensor, Callable[[Tensor], Tensor]]:
    packed, layout = pack([t], pattern)
    return packed, lambda out: unpack(out, layout, pattern)[0]
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Generic scaled-dot-product attention
# ══════════════════════════════════════════════════════════════════════════════
def attend(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    mask: Tensor | None = None,
    scale: float | None = None,
    attn_bias: Tensor | None = None,
    return_sim: bool = False,
):
    """
    All-purpose GQA-aware attention used by the three NSA branches.
    Shapes
        q : [B,      Hq,      Lq, D]
        k : [B,      Hk,      Lk, D]
        v : [B,      Hk,      Lk, D]
        mask / bias broadcast to [B, Hk, Hq/G, Lq, Lk]
    """
    scale = default(scale, q.shape[-1] ** -0.5)
 
    q_heads, k_heads = q.shape[1], k.shape[1]
    grouped = q_heads // k_heads                                   # GQA ratio
 
    q = rearrange(q, "b (h g) l d -> b h g l d", g=grouped)
    sim = einsum(q, k, "b h g i d, b h j d -> b h g i j") * scale
 
    if exists(attn_bias):                                          # distance bias
        b, h, g, *_ = sim.shape
        bias = attn_bias
        if b != bias.shape[0]:  # handle packing for local windows
            bias = bias.repeat_interleave(b // bias.shape[0], 0)
        sim = sim + bias.view(b, h, g, *bias.shape[-2:])
 
    if exists(mask):
        sim = sim.masked_fill(~mask, max_neg_value(sim) // 10)
 
    attn = sim.softmax(dim=-1)
    out = einsum(attn, v, "b h g i j, b h j d -> b h g i d")
    out = rearrange(out, "b h g l d -> b (h g) l d")
 
    if not return_sim:
        return out
 
    sim = rearrange(sim, "b h g i j -> b (h g) i j")
    return out, sim
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SparseAttention – native sparse three-branch attention
# ══════════════════════════════════════════════════════════════════════════════
class SparseAttention(Module):
    """
    Native Sparse Attention (Tang & Koltun, 2024) – cleaned, same math.
 
    The class still *intentionally* overrides the requested head
    configuration so that
        heads      = 16
        kv_heads   = 1
    (ensuring a GQA ratio of 16 for FLA-Org kernels).
    """
 
    # NOTE: public signature may **not** be changed – BallNSA depends on it.
    def __init__(  # noqa: PLR0913 – many hyper-params are intrinsic to NSA
        self,
        dim: int,
        dim_head: int,
        heads: int,
        ball_size: int,
        compress_block_size: int,
        compress_block_sliding_stride: int,
        selection_block_size: int,
        num_selected_blocks: int,
        *,
        kv_heads: int | None = None,
        num_compressed_mem_kv: int = 1,
        norm: bool = True,
        use_diff_topk: bool = False,
        query_heads_share_selected_kv: bool = True,
        compress_mlp: Module | None = None,
        compress_mlp_expand_factor: float = 1.0,
        naive_fine_attn: bool = False,
        strategy_combine_mlp: Module | None = None,
    ):
        super().__init__()
 
        # ──────────────────────────────────────────────────────────────────
        # Override – “dumb way” kept exactly as in the original code
        # ──────────────────────────────────────────────────────────────────
        print("!!! OVERRIDING IN A DUMB WAY")
        heads = 16
        kv_heads = 1
 
        # ──────────────────────────────────────────────────────────────────
        # Core hyper-parameters
        # ──────────────────────────────────────────────────────────────────
        self.naive_fine_attn = naive_fine_attn
 
        self.heads = heads
        self.kv_heads = kv_heads
        self.num_grouped_queries = heads // kv_heads
        self.dim_head = dim_head
        self.ball_size = ball_size
        self.scale = dim_head ** -0.5
 
        # ──────────────────────────────────────────────────────────────────
        # Normalisation and QKV projection
        # ──────────────────────────────────────────────────────────────────
        inner_dim, kv_inner = dim_head * heads, dim_head * kv_heads
        self.norm = RMSNorm(dim) if norm else Identity()
 
        self.qkv_split = (inner_dim, kv_inner, kv_inner)
        self.to_qkv = Linear(dim, sum(self.qkv_split), bias=False)
 
        # ──────────────────────────────────────────────────────────────────
        # Compression branch (coarse attention)
        # ──────────────────────────────────────────────────────────────────
        self.compress_block_size = compress_block_size
        self.compress_block_sliding_stride = compress_block_sliding_stride
        assert (
            compress_block_size >= compress_block_sliding_stride > 0
        ), "Invalid compress block params"
        assert divisible_by(
            selection_block_size, compress_block_sliding_stride
        ), "`selection_block_size` must be divisible by sliding stride"
 
        self.split_compress_window = nn.Sequential(
            Rearrange("b h n d -> (b h) d 1 n"),
            ZeroPad2d((compress_block_size - compress_block_sliding_stride, 0, 0, 0)),
            nn.Unfold(kernel_size=(1, compress_block_size), stride=(1, compress_block_sliding_stride)),
            Rearrange(
                "(b h) (d n) w -> b h w n d",
                d=dim_head,
                h=kv_heads,
                n=compress_block_size,
            ),
        )
 
        self.num_mem_compress_kv = max(1, num_compressed_mem_kv)
        self.compress_mem_kv = nn.Parameter(torch.zeros(2, kv_heads, self.num_mem_compress_kv, dim_head))
 
        if compress_mlp is None:
            comp_dim = compress_block_size * dim_head
            hidden = int(compress_mlp_expand_factor * comp_dim)
            compress_mlp = nn.Sequential(
                Rearrange("b h w n d -> b h w (n d)"),
                Linear(comp_dim, hidden),
                nn.ReLU(),
                Linear(hidden, dim_head),
            )
        self.k_compress = deepcopy(compress_mlp)
        self.v_compress = deepcopy(compress_mlp)
 
        # ──────────────────────────────────────────────────────────────────
        # Sliding / local branch
        # ──────────────────────────────────────────────────────────────────
        self.sigma_att = nn.Parameter(-1 + 0.01 * torch.randn((1, heads, 1, 1)))
 
        # ──────────────────────────────────────────────────────────────────
        # Fine-attention selection branch
        # ──────────────────────────────────────────────────────────────────
        self.selection_block_size = selection_block_size
        self.num_selected_blocks = max(0, num_selected_blocks)
        self.use_diff_topk = use_diff_topk
        self.query_heads_share_selected_kv = query_heads_share_selected_kv
 
        if self.num_selected_blocks == 0:
            print("`num_selected_blocks` set to 0 → fine branch disabled")
 
        # ──────────────────────────────────────────────────────────────────
        # Strategy mixer
        # ──────────────────────────────────────────────────────────────────
        if strategy_combine_mlp is None:
            strategy_combine_mlp = Linear(dim, 3 * heads)
            nn.init.zeros_(strategy_combine_mlp.weight)
            strategy_combine_mlp.bias.data.copy_(torch.tensor([-2., -2., 2.] * heads))
 
        self.to_strategy_combine = nn.Sequential(
            strategy_combine_mlp,
            nn.Sigmoid(),
            Rearrange("b n (h s) -> b h n s", h=heads),
        )
 
        # ──────────────────────────────────────────────────────────────────
        # Head split / merge helpers
        # ──────────────────────────────────────────────────────────────────
        self.split_heads = Rearrange("b n (h d) -> b h n d", d=dim_head)
        self.merge_heads = Rearrange("b h n d -> b n (h d)")
        self.combine_heads = Linear(inner_dim, dim, bias=False)
 
    # ──────────────────────────────────────────────────────────────────────────
    # Distance-based positional bias (eq. 10)
    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def create_attention_bias(self, pos: Tensor) -> Tensor:
        pos = rearrange(pos, "(n m) d -> n m d", m=self.ball_size)
        return self.sigma_att * torch.cdist(pos, pos, p=2).unsqueeze(1)
 
    # pylint: disable=too-many-locals,too-many-statements
    def forward(  # noqa: PLR0915 – NSA has inherent complexity
        self,
        inp: Tensor,
        *,
        pos: Tensor,
        sliding_window_flex_mask: Tensor | None = None,
        fine_selection_flex_mask: Callable[[Tensor, int], Tensor] | None = None,
        pos_emb: Tensor | None = None,  # added to sliding branch
    ) -> Tensor:
        """
        Args
        ----
        inp : [B, T, D]
        pos : [B·T, dim_pos]  (flattened leaf positions – required for bias)
        """
        B, T = inp.shape[:2]
        device = inp.device
 
        # ─── Pre-norm & positional embed (sliding branch only) ───────────
        inp_norm = self.norm(inp)
        inp_sliding = inp_norm if pos_emb is None else inp_norm + pos_emb
 
        # ─── Project QKV once (shared heads) ─────────────────────────────
        q, k, v = self.to_qkv(inp_norm).split(self.qkv_split, dim=-1)
        q, k, v = map(self.split_heads, (q, k, v))
 
        # =================================================================
        # 1) COARSE / COMPRESS BRANCH
        # =================================================================
        stride = self.compress_block_sliding_stride
        T_c = round_down_mult(T, stride)
        num_c_blocks = T_c // stride
 
        k_c_inp, v_c_inp = k[..., :T_c, :], v[..., :T_c, :]
 
        if not is_empty(k_c_inp):
            k_c_inp = self.split_compress_window(k_c_inp)
            v_c_inp = self.split_compress_window(v_c_inp)
        else:  # keep shapes consistent even if empty
            shape = (B, self.kv_heads, 0, self.compress_block_size, self.dim_head)
            k_c_inp, v_c_inp = (t.reshape(shape) for t in (k_c_inp, v_c_inp))
 
        ck = self.k_compress(k_c_inp)
        cv = self.v_compress(v_c_inp)
 
        # prepend global memory keys / values
        mem_k, mem_v = repeat(self.compress_mem_kv, "kv ... -> kv b ...", b=B)
        ck = cat([mem_k, ck], dim=-2)
        cv = cat([mem_v, cv], dim=-2)
        cq = q
 
        c_out, c_sim = attend(cq, ck, cv, return_sim=True, scale=self.scale)
        imp_scores = c_sim[..., mem_k.shape[-2]:]  # drop mem entries
 
        # =================================================================
        # 2) FINE BRANCH (selection from compressed)
        # =================================================================
        num_sel = min(self.num_selected_blocks, num_c_blocks)
 
        if self.query_heads_share_selected_kv:
            imp_scores = reduce(
                imp_scores, "b (h g) q k -> b h q k", "mean", g=self.num_grouped_queries
            )
            fine_grouped = self.num_grouped_queries
        else:
            fine_grouped = 1

        # (--) bucket-level mean if compress stride ≠ fine block
        if stride != self.selection_block_size:
            k_per_fine = self.selection_block_size // stride
            Lq, Lk = imp_scores.shape[-2:]
            imp_scores = imp_scores[..., : round_down_mult(Lk, k_per_fine)]
            if not is_empty(imp_scores):
                imp_scores = reduce(
                    imp_scores,
                    "... (k s) -> ... k",
                    "mean",
                    s=k_per_fine,
                )
                i_idx = arange(Lq, device=device) // self.selection_block_size
                j_idx = arange(imp_scores.shape[-1], device=device)
                diag = einx.equal("i, j -> i j", i_idx, j_idx)
                imp_scores = imp_scores.masked_fill(diag, max_neg_value(imp_scores))

        # softmax over blocks, ignore mem-kv
        imp_scores = F.pad(imp_scores, (1, 0), value=-1e3)
        imp_scores = imp_scores.softmax(dim=-1)[..., 1:]

        # top-k selection
        sel_val, sel_idx = imp_scores.topk(num_sel, dim=-1)
        gates = straight_through(sel_val, torch.ones_like(sel_val)) if self.use_diff_topk else None

        # fast gather with ParallelNSA fused kernel (Fla-org fine attn)
        gqa_ratio = self.heads // self.kv_heads
        assert gqa_ratio % 16 == 0, "Fla-org kernels need GQA ratio multiple of 16"

        block = self.selection_block_size
        S = num_sel
        I = sel_idx.shape[-2]
        tok2blk = torch.arange(T, device=device) // block

        blk_idx = sel_idx[:, :, tok2blk, :].permute(0, 2, 1, 3).contiguous()
        blk_idx = torch.minimum(blk_idx, tok2blk[:, None, None] // block)

        q_fla = rearrange(q, "b h t d -> b t h d")
        k_fla = rearrange(k, "b h t d -> b t h d")
        v_fla = rearrange(v, "b h t d -> b t h d")

        fine_fla = ParallelNSAFunction.apply(
            q_fla,
            k_fla,
            v_fla,
            blk_idx,
            S,
            block,
            self.scale,
            None,
        )
        fine_out = rearrange(fine_fla, "b t h d -> b h t d")

        if self.use_diff_topk:  # simple multiplicative gating
            fine_out = fine_out * repeat(gates.mean(-1), "b h t -> b h t d", d=self.dim_head)
 
        # =================================================================
        # 3) LOCAL / SLIDING BRANCH
        # =================================================================
        q_s, k_s, v_s = self.to_qkv(inp_sliding).split(self.qkv_split, dim=-1)
        q_s, k_s, v_s = map(self.split_heads, (q_s, k_s, v_s))
 
        pad_T = round_up_mult(T, self.ball_size)
        pad = partial(pad_at_dim, pad=(0, pad_T - T), dim=-2)
        q_s, k_s, v_s = map(pad, (q_s, k_s, v_s))
 
        q_s, k_s, v_s = (rearrange(t, "b h (w n) d -> (b w) h n d", n=self.ball_size) for t in (q_s, k_s, v_s))
        bias = self.create_attention_bias(pos)
 
        local_out = attend(q_s, k_s, v_s, attn_bias=bias)
        local_out = rearrange(local_out, "(b w) h n d -> b h (w n) d", b=B)[..., :T, :]
 
        # =================================================================
        # Strategy mixing & head merge
        # =================================================================
        mix = self.to_strategy_combine(inp_norm)        # [B, H, T, 3]
        out = einsum(mix, stack([c_out, fine_out, local_out]), "b h t s, s b h t d -> b h t d")
        return self.combine_heads(self.merge_heads(out))
