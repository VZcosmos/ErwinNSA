# import torch
# from native_sparse_attention_clean import SparseAttention

# batch = 2
# seq_len = 8
# dim = 32

# # 构造输入
# x = torch.randn(batch, seq_len, dim)
# pos_emb = torch.randn(batch, seq_len, dim)  # 假设你用 pe_proj(compute_rel_pos(pos)) 得到

# # 构造NSA
# nsa = SparseAttention(
#     dim=dim,
#     dim_head=8,
#     heads=4,
#     sliding_window_size=2,
#     compress_block_size=4,
#     compress_block_sliding_stride=2,
#     selection_block_size=4,
#     num_selected_blocks=2
# )

# out = nsa(x, pos_emb=pos_emb)
# out_nope = nsa(x)
# print("Input shape:", x.shape)
# print("Pos emb shape:", pos_emb.shape)
# print("NSA output shape:", out.shape)
# print("Difference:", (out_nope - out).abs().sum())

from erwin import NSABallformer
import torch

c_in = 16
c_hidden = 32
rotate = 0
depth = 2
num_heads = 4
compress_ball_size = 4
sliding_window_size = 4
num_selected_blocks = 2
mlp_ratio = 2
dimensionality = 3
mp_steps = 2
num_layers = 1

model = NSABallformer(
    c_in=c_in,
    c_hidden=c_hidden,
    rotate=rotate,
    depth=depth,
    num_heads=num_heads,
    compress_ball_size=compress_ball_size,
    sliding_window_size=sliding_window_size,
    num_selected_blocks=num_selected_blocks,
    mlp_ratio=mlp_ratio,
    dimensionality=dimensionality,
    mp_steps=mp_steps,
    num_layers=num_layers
)

batch_size = 2
num_nodes = 8
node_features = torch.randn(batch_size * num_nodes, c_in)
node_positions = torch.randn(batch_size * num_nodes, dimensionality)
batch_idx = torch.arange(batch_size).repeat_interleave(num_nodes)
edge_index = torch.randint(0, batch_size * num_nodes, (2, batch_size * num_nodes * 2))

output = model(
    node_features=node_features,
    node_positions=node_positions,
    batch_idx=batch_idx,
    edge_index=edge_index,
    radius=1.0
)

print("Output shape:", output.shape)
assert output.shape[-1] == c_hidden
print("Forward and shape check passed!")