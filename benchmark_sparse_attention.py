import torch
import time
from models.native_sparse_attention_clean import SparseAttention

def benchmark_sparse_attention(
    batch_size=2,
    seq_len=8192,
    dim=16,
    num_runs=100,
    warmup_runs=10,
    device='cuda'
):
    # Create model
    model = SparseAttention(
        dim=dim,
        dim_head=16,
        heads=4,
        ball_size=64,
        compress_block_size=64,
        compress_block_sliding_stride=32,
        selection_block_size=64,
        num_selected_blocks=4,
        kv_heads=2,
        num_compressed_mem_kv=1,
        norm=True,
        use_diff_topk=False,
        query_heads_share_selected_kv=True
    ).to(device)
    
    # Create dummy input tensors
    x = torch.randn(batch_size, seq_len, dim, device=device)
    pos = torch.randn(seq_len, 3, device=device)
    
    # Warmup runs
    print("Running warmup...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(x, pos)
    
    # Benchmark runs
    print("Running benchmark...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x, pos)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate statistics
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    tokens_per_second = (batch_size * seq_len * num_runs) / total_time
    
    print(f"\nBenchmark Results:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per forward pass: {avg_time*1000:.2f}ms")
    print(f"Tokens processed per second: {tokens_per_second:.2f}")
    
    return avg_time, tokens_per_second

if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a GPU.")
        exit(1)
        
    # Run benchmark
    avg_time, tokens_per_second = benchmark_sparse_attention() 