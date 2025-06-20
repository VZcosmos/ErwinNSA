import copy

from collections import defaultdict
import random

import wandb
import torch
import time
import os
from tqdm import tqdm
import torch.profiler
import matplotlib.pyplot as plt
import numpy as np


def plot_log_distribution(aTensor: torch.Tensor, aOutputPath: str) -> None:
    if aTensor.ndim != 1:
        raise ValueError("Input tensor must be 1-dimensional.")

    myValues = aTensor.detach().cpu().numpy()
    myValues = myValues[myValues > 0]  # log scale can't handle zero or negative

    if len(myValues) == 0:
        raise ValueError("Tensor must contain positive values for log scale.")

    myLogValues = np.log10(myValues)

    plt.figure()
    plt.hist(myLogValues, bins=50, edgecolor='black')
    plt.xlabel('log10(Value)')
    plt.ylabel('Count')
    plt.title('Log-Scaled Value Distribution')
    plt.tight_layout()
    plt.savefig(aOutputPath)
    plt.close()


def save_influence_tensor(aTensor: torch.Tensor, aOutputPath: str) -> None:
    if aTensor.ndim != 1:
        raise ValueError("Input tensor must be 1-dimensional.")

    myValues = aTensor.detach().cpu().numpy()
    np.save(aOutputPath, myValues) 


def setup_wandb_logging(model, config, project_name="ballformer"):
    wandb.init(project=project_name, config=config, name=config["model"] + '_' + config["experiment"])
    wandb.watch(model)
    wandb.config.update({"num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}, allow_val_change=True)


files_to_save = []

def path_to_file(config, file_name):
    if config.get("use_wandb", False):
        file_path = os.path.join(wandb.run.dir, file_name)
        files_to_save.append(file_path)
        return file_path
    else:
        return os.path.join(".", file_name)

def path_to_dir(config, dir_name):
    if config.get("use_wandb", False):
        return os.path.join(wandb.run.dir, dir_name)
    else:
        return os.path.join(".", dir_name)


def save_checkpoint(model, optimizer, scheduler, config, val_loss, global_step, accumulation_steps=1):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'val_loss': val_loss,
        'global_step': global_step // accumulation_steps,
        'config': config
    }
    
    save_dir = config.get('checkpoint_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    if config['model'] in ['erwin', 'pointtransformer']:
        checkpoint_path = os.path.join(save_dir, f"{config['model']}_{config['experiment']}_{config['size']}_{config['seed']}_best.pt")
    else:
        checkpoint_path = os.path.join(save_dir, f"{config['model']}_{config['experiment']}_{config['seed']}_best.pt")
    torch.save(checkpoint, checkpoint_path)
    
    if config.get("use_wandb", False):
        wandb.log({"checkpoint/best_val_loss": val_loss}, step=global_step // accumulation_steps)


def load_checkpoint(model, optimizer, scheduler, config):
    save_dir = config.get('checkpoint_dir', 'checkpoints')
    if config['model'] in ['erwin', 'pointtransformer']:
        checkpoint_path = os.path.join(save_dir, f"{config['model']}_{config['experiment']}_{config['size']}_{config['seed']}_best.pt")
    else:
        checkpoint_path = os.path.join(save_dir, f"{config['model']}_{config['experiment']}_{config['seed']}_best.pt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['val_loss'], checkpoint['global_step']


def train_step(model, batch, optimizer, scheduler, global_step, config, accumulation_steps=1):
    # optimizer.zero_grad()
    stat_dict = model.training_step(batch)
    loss = stat_dict["train/loss"] / accumulation_steps
    loss.backward()

    if (global_step + 1) % accumulation_steps == 0:
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
    stat_dict['train/lr'] = optimizer.param_groups[0]['lr']

    if global_step == 0 or global_step == 10:
        torch.cuda.memory._dump_snapshot(path_to_file(config, f"spike_{global_step}.pickle.gz"))

    return stat_dict


def validate(model, val_loader, config):
    model.eval()
    val_stats = {}
    num_batches = 0
    
    use_tqdm = not config.get("use_wandb", False)
    iterator = tqdm(val_loader, desc="Validation") if use_tqdm else val_loader
    
    for batch in iterator:
        batch = {k: v.cuda() for k, v in batch.items()}
        stat_dict = model.validation_step(batch)
        
        for k, v in stat_dict.items():
            if k not in val_stats:
                val_stats[k] = 0
            val_stats[k] += v.cpu().detach()
        
        if use_tqdm:
            iterator.set_postfix({"Loss": f"{stat_dict['val/loss'].item():.4f}"})
            
        num_batches += 1
    
    avg_stats = {f"avg/{k}": v / num_batches for k, v in val_stats.items()}
    return avg_stats

def get_leaf_values(nested_dict):
    """
    Recursively traverses a nested dictionary (including defaultdict)
    and collects all non-dictionary values (leaves).
    """
    leaf_values = []
    # Iterate through the values of the current dictionary level
    for value in nested_dict.values():
        # Check if the value is another dictionary or defaultdict
        if isinstance(value, (dict, defaultdict)):
            # If it is, recurse deeper and extend the list with results
            leaf_values.extend(get_leaf_values(value))
        else:
            # If it's not a dictionary, it's a leaf value, add it to the list
            leaf_values.append(value)
    return leaf_values


def measure_interaction_batch(model, x, pos, i_s, bs, batch):
    x = x.detach().clone().requires_grad_(True)
    N, F = x.shape
    batch_idx = torch.repeat_interleave(torch.arange(bs), N).cuda()
    
    influences = torch.zeros((len(i_s), N), device=x.device)

    for idx, j in enumerate(i_s):
        def f(input_x):
            return model(input_x, pos, batch_idx, edge_index=batch['edge_index'], num_nodes=batch['num_nodes'])[j]
        jacobian = torch.autograd.functional.jacobian(f, x, create_graph=False)
        # jacobian shape: (feature_dim, N, F)
        # For each input node i, sum the squared jacobian over output and feature dimensions
        influences[idx] = (jacobian ** 2).sum(dim=(0, 2))

    return influences

def evaluate_interactions_from_batch(model, batch, config, i_s=None):
    batch = {k: v.cuda() for k, v in batch.items()}
    # print("Batch keys:", batch.keys())
    node_positions = batch['node_positions']
    node_features = model.pos_enc(node_positions)
    def model_from_node_features(node_feats, *args, **kwargs):
        return model.pred_head(model.main_model(node_feats, *args, **kwargs))
    bs = 1
    if i_s is None:
        i_s = random.sample(range(node_features.shape[0]), 1)

    influences = measure_interaction_batch(model_from_node_features, node_features, node_positions, i_s, bs, batch)
    plot_log_distribution(influences[0], path_to_file(config, "influences.png"))
    save_influence_tensor(influences[0], path_to_file(config, "influences_tensor.npy"))
    return (influences[0] > 0).sum().item()


def measure_interaction_batch_md(model, vel_seq, node_type, node_positions, i_s, bs, batch):
    vel_seq = vel_seq.detach().clone().requires_grad_(True)
    N, F = vel_seq.shape
    batch_idx = batch["batch_idx"]
    influences = torch.zeros((len(i_s), N), device=vel_seq.device)

    for idx, j in enumerate(i_s):
        def f(input_vel_seq):
            acc_mean, _ = model(input_vel_seq, node_positions, node_type, batch_idx, edge_index=batch['edge_index'], num_nodes=batch['num_nodes'])
            return acc_mean[j]
        jacobian = torch.autograd.functional.jacobian(f, vel_seq, create_graph=False)
        influences[idx] = (jacobian ** 2).sum(dim=(0, 2))  # sum over output and feature dim

    return influences

def evaluate_interactions_from_batch_md(model, batch, config, i_s=None):
    batch = {k: v.cuda() for k, v in batch.items()}
    vel_seq = batch['vel_seq']
    node_type = batch['node_type']
    node_positions = batch['node_positions']
    bs = 1
    if i_s is None:
        i_s = random.sample(range(vel_seq.shape[0]), 1)

    influences = measure_interaction_batch_md(model, vel_seq, node_type, node_positions, i_s, bs, batch)
    plot_log_distribution(influences[0], path_to_file(config, "md_influences.png"))
    save_influence_tensor(influences[0], path_to_file(config, "md_influences_tensor.npy"))
    return (influences[0] > 0).sum().item()

def measure_interaction_batch_cosmology(model, node_positions, i_s, bs, batch):
    node_positions = node_positions.detach().clone().requires_grad_(True)
    N, F = node_positions.shape
    batch_idx = batch["batch_idx"]
    influences = torch.zeros((len(i_s), N), device=node_positions.device)

    for idx, j in enumerate(i_s):
        def f(input_pos):
            pred = model(input_pos, batch_idx=batch_idx, edge_index=batch['edge_index'])
            return pred[j]
        jacobian = torch.autograd.functional.jacobian(f, node_positions, create_graph=False)
        influences[idx] = (jacobian ** 2).sum(dim=(0, 2))  

    return influences

def evaluate_interactions_from_batch_cosmology(model, batch, config, i_s=None):
    batch = {k: v.cuda() for k, v in batch.items()}
    node_positions = batch['pos']
    bs = 1
    if i_s is None:
        i_s = random.sample(range(node_positions.shape[0]), 1)

    influences = measure_interaction_batch_cosmology(model, node_positions, i_s, bs, batch)
    plot_log_distribution(influences[0], path_to_file(config, "cosmo_influences.png"))
    save_influence_tensor(influences[0], path_to_file(config, "cosmo_influences_tensor.npy"))
    return (influences[0] > 0).sum().item()

def benchmark_flops(model, data_loader, config):
    """
    Calculate the number of FLOPs for the model.
    """
    sample_batch = next(iter(data_loader))
    sample_batch = {k: v.cuda() for k, v in sample_batch.items()}
    from torchtnt.utils.flops import FlopTensorDispatchMode
    with FlopTensorDispatchMode(model) as ftdm:
        res = model.training_step(sample_batch)["train/loss"]
        flops_forward = copy.deepcopy(ftdm.flop_counts)
        total_forward_flops = sum(get_leaf_values(flops_forward))
        ftdm.reset()
    return total_forward_flops

def count_parameters(model):
    """
    Count the number of trainable and total parameters in the model.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params


def fit(config, model, optimizer, scheduler, train_loader, val_loader, test_loader=None, timing_window_start=100, timing_window_size=500, dataset_type="shapenet", accumulation_steps=1):
    if config.get("use_wandb", False):
        setup_wandb_logging(model, config)
    
    use_tqdm = not config.get("use_wandb", False)
    running_train_stats = {}
    num_train_batches = 0
    global_step = 0
    best_val_loss = float('inf')
    max_steps = config["num_epochs"]

    start_time = time.time()
    peak_memory_gb = "N/A"  # Default value if not using CUDA

    # Memory Measurement Setup (Peak GPU Memory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Reset peak memory stats for the target device before the function call
        #torch.cuda.reset_peak_memory_stats(device)
        memory_before_gb = torch.cuda.memory_allocated(device) / (1024**3)

    trainable_params, total_params = count_parameters(model)
    flops_per_step = benchmark_flops(model, train_loader, config)
    if config.get("use_wandb", False):
        wandb.log({"stats/flops_per_step": flops_per_step})
        wandb.log({"stats/trainable_params": trainable_params})
        wandb.log({"stats/total_params": total_params})
    prof = None
    if config.get("profile", False):
        os.makedirs(path_to_file(config, "tb_trace"), exist_ok=True)

        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(path_to_dir(config, "tb_trace"))
        )
        prof.__enter__()

    optimizer.zero_grad()
    while global_step < max_steps:
        iterator = tqdm(train_loader, desc=f"Training (step {global_step + 1}/{max_steps})") if use_tqdm else train_loader
        for batch in iterator:
            if global_step >= max_steps:
                break
                
            model.train()
            batch = {k: v.cuda() for k, v in batch.items()}
            
            # measure runtime statistics
            if global_step == timing_window_start:
                timing_start = time.perf_counter()
            
            if global_step == timing_window_start + timing_window_size:
                timing_end = time.perf_counter()
                total_time = timing_end - timing_start
                steps_per_second = timing_window_size / total_time
                if config.get("use_wandb", False):
                    wandb.log({"stats/steps_per_second": steps_per_second}, step=global_step // accumulation_steps)
                else:
                    print(f"Steps per second: {steps_per_second:.2f}")
            
            # Add optimizer zero_grad so that gradient accumulation works correctly inside of train_step
            stat_dict = train_step(model, batch, optimizer, scheduler, global_step, config, accumulation_steps=accumulation_steps)

            if global_step == 1:
                print(f"{accumulation_steps=}")            
            for k, v in stat_dict.items():
                if "lr" not in k:
                    if k not in running_train_stats:
                        running_train_stats[k] = 0
                    running_train_stats[k] += v.cpu().detach()
            num_train_batches += 1
            
            if use_tqdm:
                loss_keys = [k for k in stat_dict.keys() if "loss" in k]
                iterator.set_postfix({
                    "step": f"{global_step + 1}/{max_steps}",
                    **{k: f"{stat_dict[k].item():.4f}" for k in loss_keys}
                })
            else:
                if global_step % accumulation_steps == 0:
                    wandb.log({f"{k}": v.item() for k, v in stat_dict.items() if "lr" not in k}, step=global_step // accumulation_steps)
            
            # Validation and checkpointing
            if global_step % accumulation_steps == 0 and (global_step // accumulation_steps) % config["val_every_iter"] == 0:
                train_stats = {f"avg/{k}": v / num_train_batches for k, v in running_train_stats.items()}
                
                running_train_stats = {}
                num_train_batches = 0
                
                val_stats = validate(model, val_loader, config)
                current_val_loss = val_stats['avg/val/loss']
                
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    save_checkpoint(model, optimizer, scheduler, config, best_val_loss, global_step, accumulation_steps=accumulation_steps)
                    if not config.get("use_wandb", False):
                        print(f"New best validation loss: {best_val_loss:.4f}, saved checkpoint")
                
                if config.get("use_wandb", False):
                    wandb.log({**train_stats, **val_stats, 'global_step': global_step}, step=global_step // accumulation_steps)
                else:
                    loss_keys = [k for k in val_stats.keys() if "loss" in k]
                    for k in loss_keys: 
                        print(f"Validation {k}: {val_stats[k]:.4f}")
            
            if prof is not None:
                prof.step()
            global_step += 1

    if test_loader is not None and config.get('test', False):
        print("Loading best checkpoint for testing...")
        best_val_loss, best_step = load_checkpoint(model, optimizer, scheduler, config)
        print(f"Loaded checkpoint from step {best_step} with validation loss {best_val_loss:.4f}")
        
        test_stats = validate(model, test_loader, config)
        if config.get("use_wandb", False):
            wandb.log({
                **{f"test/{k.replace('val/', '')}": v for k, v in test_stats.items()},
                'global_step': global_step // accumulation_steps,
            }, step=global_step // accumulation_steps)
        else:
            loss_keys = [k for k in test_stats.keys() if "loss" in k]
            for k in loss_keys:
                print(f"Test {k}: {test_stats[k]:.4f}")

    # --- Finalize Measurements & Report ---
    end_time = time.time()
    runtime_seconds = end_time - start_time

    # Memory Measurement Finalization (Peak GPU Memory)
    if device.type == "cuda":
        # Get peak memory allocated during the function call [[3]]
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_gb = peak_memory_bytes / (1024**3)
        memory_after_gb = torch.cuda.memory_allocated(device) / (1024**3)
        print(
            f"GPU Memory allocated after: {memory_after_gb:.4f} GB"
        )
    
    if dataset_type == "shapenet":
        n_influenced = evaluate_interactions_from_batch(model, next(iter(train_loader)), config)
    elif dataset_type == "md":
        n_influenced = evaluate_interactions_from_batch_md(model, next(iter(train_loader)), config)
    elif dataset_type == "cosmology":
        n_influenced = evaluate_interactions_from_batch_cosmology(model, next(iter(train_loader)), config)
    flops_per_second = flops_per_step * (config["num_epochs"] * len(train_loader) / runtime_seconds)
    if config.get("use_wandb", False):
        wandb.log({"stats/runtime": runtime_seconds})
        wandb.log({"stats/peak_gpu_memory": peak_memory_gb})
        wandb.log({"stats/before_gpu_memory": memory_before_gb})
        wandb.log({"stats/after_gpu_memory": memory_after_gb})
        wandb.log({"stats/flops_per_second": flops_per_second})


        wandb.log({"stats/n_influenced": n_influenced})
        for file in files_to_save:
            print(f"Saving {file} to wandb")
            wandb.save(file)
    print("\n--- Monitoring Results ---")
    print(f"Total Runtime: {runtime_seconds:.2f} seconds")
    print(
        f"Peak GPU Memory Allocated during execution: {peak_memory_gb:.4f} GB"
    )
    print(
        f"GPU Memory allocated before: {memory_before_gb:.4f} GB"
    )
    print(
        f"GPU Memory allocated after: {memory_after_gb:.4f} GB"
    )
    print(
        f"Number of influenced nodes: {n_influenced}"
    )

    if prof is not None:
        prof.__exit__(None, None, None)
        snapshot_path = path_to_file(config, "snapshot.pickle.gz")
        torch.cuda.memory._dump_snapshot(snapshot_path)
        if config.get("use_wandb", False):
            wandb.save(os.path.join(path_to_dir(config, "tb_trace"), "*"))
            wandb.save(snapshot_path)
    
    return model
