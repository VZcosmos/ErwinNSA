import torch
import wandb
from models import NSABallformer
from experiments.datasets import ShapenetCarDataset
from torch.utils.data import DataLoader
from training import fit
import itertools

def run_ablation(base_config, ablation_params):
    """
    Run ablation studies by varying specified parameters.
    
    Args:
        base_config: Base configuration dictionary
        ablation_params: Dictionary of parameters to ablate and their possible values
    """
    # Generate all combinations of ablation parameters
    param_names = list(ablation_params.keys())
    param_values = list(ablation_params.values())
    param_combinations = list(itertools.product(*param_values))
    
    for params in param_combinations:
        # Update base config with current ablation parameters
        current_config = base_config.copy()
        for name, value in zip(param_names, params):
            current_config[name] = value
            
        # Initialize model with current config
        model = NSABallformer(
            c_in=current_config["c_in"],
            c_hidden=current_config["c_hidden"],
            rotate=current_config["rotate"],
            depth=current_config["depth"],
            num_heads=current_config["num_heads"],
            compress_ball_size=current_config["compress_ball_size"],
            local_ball_size=current_config["local_ball_size"],
            num_selected_blocks=current_config["num_selected_blocks"],
            mlp_ratio=current_config["mlp_ratio"],
            dimensionality=current_config["dimensionality"],
            mp_steps=current_config["mp_steps"],
            num_layers=current_config["num_layers"]
        ).cuda()
        
        # Load dataset
        train_dataset = ShapenetCarDataset(
            data_path=current_config["data_path"],
            split="train",
            knn=current_config["knn"]
        )
        val_dataset = ShapenetCarDataset(
            data_path=current_config["data_path"],
            split="val",
            knn=current_config["knn"]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=current_config["batch_size"],
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            num_workers=current_config["num_workers"]
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=current_config["batch_size"],
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
            num_workers=current_config["num_workers"]
        )
        
        # Create run name from ablation parameters
        run_name = "_".join(f"{name}_{value}" for name, value in zip(param_names, params))
        
        with wandb.init(
            project="erwin_nsa_ablation",
            name=run_name,
            config=current_config
        ):
            optimizer = torch.optim.AdamW(model.parameters(), lr=current_config["lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=current_config["max_steps"]
            )
            
            fit(
                config=current_config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                val_loader=val_loader,
                timing_window_start=100,
                timing_window_size=500
            )

if __name__ == "__main__":
    # Base configuration
    base_config = {
        "c_in": 16,
        "c_hidden": 32,
        "rotate": 0,
        "depth": 2,
        "num_heads": 4,
        "compress_ball_size": 128,
        "local_ball_size": 128,
        "num_selected_blocks": 4,
        "mlp_ratio": 4,
        "dimensionality": 3,
        "mp_steps": 3,
        "num_layers": 3,
        "batch_size": 16,
        "lr": 1e-4,
        "max_steps": 1000,
        "knn": 8,
        "num_workers": 4,
        "data_path": "path/to/shapenet",
        "use_wandb": True
    }
    
    # Parameters to ablate
    ablation_params = {
        "compress_ball_size": [64, 128, 256],
        "local_ball_size": [64, 128, 256],
        "num_selected_blocks": [2, 4, 8],
        "num_layers": [1, 2, 3],
        "depth": [1, 2, 4]
    }
    
    run_ablation(base_config, ablation_params) 