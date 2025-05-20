import torch
import matplotlib.pyplot as plt
import seaborn as sns
from models import NSABallformer
from experiments.datasets import ShapenetCarDataset
from torch.utils.data import DataLoader
import numpy as np

def visualize_attention_patterns(model, data_loader, save_dir="attention_visualizations"):
    """
    Visualize attention patterns from different components of NSA.
    """
    model.eval()
    batch = next(iter(data_loader))
    batch = {k: v.cuda() for k, v in batch.items()}
    
    # Get attention patterns
    with torch.no_grad():
        # Forward pass with hooks to capture attention
        attention_patterns = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    attention_patterns[name] = output[1]  # Assuming attention weights are second output
                else:
                    attention_patterns[name] = output
            return hook
        
        # Register hooks for each attention component
        for name, module in model.named_modules():
            if isinstance(module, (BallNSA)):
                module.register_forward_hook(hook_fn(f"nsa_{name}"))
        
        # Forward pass
        _ = model(batch['node_features'], batch['node_positions'], batch['batch_idx'])
    
    # Visualize patterns
    for name, pattern in attention_patterns.items():
        if pattern is None:
            continue
            
        # Convert to numpy and handle different pattern types
        if isinstance(pattern, torch.Tensor):
            pattern = pattern.cpu().numpy()
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        if pattern.ndim == 4:  # Multi-head attention
            # Average over heads
            pattern = pattern.mean(axis=1)
        
        # Plot heatmap
        sns.heatmap(pattern[0], cmap='viridis')  # Plot first batch item
        plt.title(f"Attention Pattern: {name}")
        plt.savefig(f"{save_dir}/{name}_attention.png")
        plt.close()
        
        # Plot distribution
        plt.figure(figsize=(10, 4))
        sns.histplot(pattern[0].flatten(), bins=50)
        plt.title(f"Attention Distribution: {name}")
        plt.savefig(f"{save_dir}/{name}_distribution.png")
        plt.close()

def visualize_ball_structure(model, data_loader, save_dir="ball_visualizations"):
    """
    Visualize the ball tree structure and how it affects attention.
    """
    model.eval()
    batch = next(iter(data_loader))
    batch = {k: v.cuda() for k, v in batch.items()}
    
    # Get ball assignments
    with torch.no_grad():
        tree_idx, tree_mask, _ = model.build_balltree(
            batch['node_positions'], 
            batch['batch_idx']
        )
        
        # Visualize ball assignments
        pos = batch['node_positions'].cpu().numpy()
        ball_assignments = tree_idx.cpu().numpy()
        
        # Create 3D scatter plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points colored by ball assignment
        scatter = ax.scatter(
            pos[:, 0], pos[:, 1], pos[:, 2],
            c=ball_assignments,
            cmap='tab20'
        )
        
        plt.colorbar(scatter, label='Ball Assignment')
        plt.title('Ball Tree Structure')
        plt.savefig(f"{save_dir}/ball_structure.png")
        plt.close()

if __name__ == "__main__":
    # Initialize model
    model = NSABallformer(
        c_in=16,
        c_hidden=32,
        rotate=0,
        depth=2,
        num_heads=4,
        compress_ball_size=128,
        local_ball_size=128,
        num_selected_blocks=4,
        mlp_ratio=4,
        dimensionality=3,
        mp_steps=3,
        num_layers=3
    ).cuda()
    
    # Load dataset
    dataset = ShapenetCarDataset(
        data_path="path/to/shapenet",
        split="train",
        knn=8
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=4
    )
    
    # Create visualization directories
    import os
    os.makedirs("attention_visualizations", exist_ok=True)
    os.makedirs("ball_visualizations", exist_ok=True)
    
    # Generate visualizations
    visualize_attention_patterns(model, data_loader)
    visualize_ball_structure(model, data_loader) 