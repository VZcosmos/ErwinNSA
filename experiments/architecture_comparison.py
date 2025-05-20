import torch
import wandb
from models import ErwinTransformer, NSABallformer
from experiments.datasets import ShapenetCarDataset
from torch.utils.data import DataLoader
from training import fit

def run_comparison(config):
    # Initialize both models with equivalent parameters
    erwin_config = {
        "c_in": config["c_in"],
        "c_hidden": config["c_hidden"],
        "ball_sizes": [config["ball_size"]] * config["num_layers"],
        "enc_num_heads": [config["num_heads"]] * config["num_layers"],
        "enc_depths": [config["depth"]] * config["num_layers"],
        "dec_num_heads": [config["num_heads"]] * (config["num_layers"] - 1),
        "dec_depths": [config["depth"]] * (config["num_layers"] - 1),
        "strides": [2] * (config["num_layers"] - 1),
        "rotate": config["rotate"],
        "mlp_ratio": config["mlp_ratio"],
        "dimensionality": config["dimensionality"],
        "mp_steps": config["mp_steps"]
    }

    nsa_config = {
        "c_in": config["c_in"],
        "c_hidden": config["c_hidden"],
        "rotate": config["rotate"],
        "depth": config["depth"],
        "num_heads": config["num_heads"],
        "compress_ball_size": config["ball_size"],
        "local_ball_size": config["ball_size"],
        "num_selected_blocks": config["num_selected_blocks"],
        "mlp_ratio": config["mlp_ratio"],
        "dimensionality": config["dimensionality"],
        "mp_steps": config["mp_steps"],
        "num_layers": config["num_layers"]
    }

    # Initialize models
    erwin_model = ErwinTransformer(**erwin_config).cuda()
    nsa_model = NSABallformer(**nsa_config).cuda()

    # Load dataset
    train_dataset = ShapenetCarDataset(
        data_path=config["data_path"],
        split="train",
        knn=config["knn"]
    )
    val_dataset = ShapenetCarDataset(
        data_path=config["data_path"],
        split="val",
        knn=config["knn"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=config["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=config["num_workers"]
    )

    # Train and evaluate both models
    for model_name, model in [("erwin", erwin_model), ("nsa", nsa_model)]:
        with wandb.init(
            project="erwin_architecture_comparison",
            name=f"{model_name}_{wandb.util.generate_id()}",
            config={**config, "model": model_name}
        ):
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["max_steps"]
            )
            
            fit(
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                val_loader=val_loader,
                timing_window_start=100,
                timing_window_size=500
            )

if __name__ == "__main__":
    config = {
        "c_in": 16,
        "c_hidden": 32,
        "ball_size": 128,
        "num_heads": 4,
        "depth": 2,
        "num_layers": 3,
        "rotate": 0,
        "mlp_ratio": 4,
        "dimensionality": 3,
        "mp_steps": 3,
        "num_selected_blocks": 4,
        "batch_size": 16,
        "lr": 1e-4,
        "max_steps": 1000,
        "knn": 8,
        "num_workers": 4,
        "data_path": "path/to/shapenet",
        "use_wandb": True
    }
    
    run_comparison(config) 