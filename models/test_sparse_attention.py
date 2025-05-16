import torch
import matplotlib.pyplot as plt
from native_sparse_attention_clean import SparseAttention
import random

def plot_heatmap(matrix, title, filename):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def measure_interaction_batch(model, x, pos, i_s):
    x = x.detach().clone().requires_grad_(True)
    N = x.shape[0]
    
    influences = torch.zeros((i_s, N), device=x.device)

    for j in range(N):
        def f(input_x):
            return model(input_x, pos)[0]
        jacobian = torch.autograd.functional.jacobian(f, x, create_graph=False)
        print("jacobian", jacobian.shape)
        for i in range(i_s):
            influences[i, j] = (jacobian[:, i, :] ** 2).sum()

    return influences

def evaluate_interactions(model, batch_size, seq_len, dim):
    # Create dummy input tensors
    x = torch.randn(batch_size, seq_len, dim)
    pos = torch.randn(seq_len, 3)
    
    # Measure interactions
    influences = measure_interaction_batch(model, x, pos, batch_size)
    print("influences", influences.shape)
    plot_heatmap(influences[0].detach().numpy(), 'Influence Matrix', 'influence_heatmap.png')
    
    # Count number of influenced nodes (nodes with influence > 1e-10)
    n_influenced = (influences[0] > 1e-10).sum().item()
    
    return n_influenced

def test_sparse_attention():
    # Initialize parameters
    dim = 64  # Input dimension
    dim_head = 32  # Dimension per attention head
    heads = 4  # Number of attention heads
    ball_size = 8  # Size of the ball for local attention
    compress_block_size = 4  # Size of compression blocks
    compress_block_sliding_stride = 2  # Stride for sliding window
    selection_block_size = 4  # Size of selection blocks
    num_selected_blocks = 2  # Number of blocks to select

    # Create the SparseAttention module
    sparse_attn = SparseAttention(
        dim=dim,
        dim_head=dim_head,
        heads=heads,
        ball_size=ball_size,
        compress_block_size=compress_block_size,
        compress_block_sliding_stride=compress_block_sliding_stride,
        selection_block_size=selection_block_size,
        num_selected_blocks=num_selected_blocks
    )

    # Create dummy input tensors
    batch_size = 2
    seq_len = 32  # As requested, 32 points in the sequence
    
    # Input features
    x = torch.randn(batch_size, seq_len, dim)
    
    # Position embeddings (3D coordinates for each point)
    pos = torch.randn(seq_len, 3)

    # Forward pass
    # try:
    output, compressed_attn_out, fine_attn_out, local_attn_out = sparse_attn(x, pos)
    print("Forward pass successful!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Convert tensors to numpy arrays for plotting
    csim_np = compressed_attn_out[0, 0].detach().numpy()  # Take first batch and head
    fsim_np = fine_attn_out[0, 0].detach().numpy()  # Take first batch and head
    local_lse_np = local_attn_out[0, 0].detach().numpy()  # Take first batch and head
    
    # Plot and save heatmaps
    plot_heatmap(csim_np, 'Coarse Similarity Matrix', 'csim_heatmap.png')
    plot_heatmap(fsim_np, 'Fine Similarity Matrix', 'fsim_heatmap.png')
    plot_heatmap(local_lse_np, 'Local LSE Matrix', 'local_lse_heatmap.png')
    print("Heatmaps have been saved as 'csim_heatmap.png' and 'fsim_heatmap.png'")
    
    # Evaluate interactions
    print("\nEvaluating interactions...")
    n_influenced = evaluate_interactions(sparse_attn, batch_size, seq_len, dim)
    print(f"Number of influenced nodes: {n_influenced}")
    
    return True
    # except Exception as e:
    #     print(f"Error during forward pass: {str(e)}")
    #     return False

if __name__ == "__main__":
    test_sparse_attention() 