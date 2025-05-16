import torch
import matplotlib.pyplot as plt
from native_sparse_attention_clean import SparseAttention
# from native_sparse_attention import SparseAttention

def plot_heatmap(matrix, title, filename):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def test_sparse_attention():
    # Initialize parameters
    dim = 64  # Input dimension
    dim_head = 32  # Dimension per attention head
    heads = 2  # Number of attention heads
    ball_size = 8  # Size of the ball for local attention
    compress_block_size = 4  # Size of compression blocks
    compress_block_sliding_stride = 1  # Stride for sliding window
    selection_block_size = 4  # Size of selection blocks
    num_selected_blocks = 1  # Number of blocks to select

    # Create dummy input tensors
    batch_size = 2
    seq_len = 16

    # Create the SparseAttention module
    sparse_attn = SparseAttention(
        dim=dim,
        dim_head=dim_head,
        heads=heads,
        ball_size=ball_size,
        # sliding_window_size=ball_size,
        compress_block_size=compress_block_size,
        compress_block_sliding_stride=compress_block_sliding_stride,
        selection_block_size=selection_block_size,
        num_selected_blocks=num_selected_blocks
    )
    
    # Input features
    x = torch.randn(batch_size, seq_len, dim)
    
    # Position embeddings (3D coordinates for each point)
    pos = torch.randn(seq_len, 3)

    # Forward pass
    def f(input_x):
        return sparse_attn(input_x, pos)
        # return sparse_attn(input_x)
    
    jacobian = torch.autograd.functional.jacobian(f, x, create_graph=False)
    influence = torch.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            influence[i, j] = (jacobian[0, j, :, 0, i, :] ** 2).sum()

    influence = torch.log(influence + 1e-9)
    plot_heatmap(influence.detach().numpy(), 'Influence Matrix (Selection)', 'influence_heatmap.png')
    

if __name__ == "__main__":
    test_sparse_attention() 