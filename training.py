import copy

from collections import defaultdict
import random

import wandb
import torch
import time
import os
from tqdm import tqdm
from torchtnt.utils.flops import FlopTensorDispatchMode


def setup_wandb_logging(model, config, project_name="ballformer"):
    wandb.init(project=project_name, config=config, name=config["model"] + '_' + config["experiment"])
    wandb.watch(model)
    wandb.config.update({"num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}, allow_val_change=True)


def save_checkpoint(model, optimizer, scheduler, config, val_loss, global_step):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'val_loss': val_loss,
        'global_step': global_step,
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
        wandb.log({"checkpoint/best_val_loss": val_loss}, step=global_step)


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


def train_step(model, batch, optimizer, scheduler):
    optimizer.zero_grad()
    stat_dict = model.training_step(batch)
    stat_dict["train/loss"].backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    stat_dict['train/lr'] = optimizer.param_groups[0]['lr']
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
    N = x.shape[0]
    batch_idx = torch.repeat_interleave(torch.arange(bs), N).cuda()
    
    influences = torch.zeros((len(i_s), N), device=x.device)

    for j in range(N):
        def f(input_x):
            return model(input_x, pos, batch_idx, edge_index=batch['edge_index'], num_nodes=batch['num_nodes'])[j]
        jacobian = torch.autograd.functional.jacobian(f, x, create_graph=False)
        #print(f'Jacobian {jacobian.shape}')
        for num_i, i in enumerate(i_s):
            influences[num_i, j] = (jacobian[:, i, :] ** 2).sum()

    return influences

def evaluate_interactions_from_batch(model, batch, config, i_s=None):
    batch = {k: v.cuda() for k, v in batch.items()}
    print("Batch keys:", batch.keys())
    node_positions = batch['node_positions']
    node_features = model.pos_enc(node_positions)
    def model_from_node_features(node_feats, *args, **kwargs):
        return model.pred_head(model.main_model(node_feats, *args, **kwargs))
    bs = 1
    if i_s is None:
        i_s = random.sample(range(node_features.shape[0]), 1)

    influences = measure_interaction_batch(model_from_node_features, node_features, node_positions, i_s, bs, batch)
    return (influences[0] > 1e-10).sum().item()

def benchmark_flops(model, data_loader, config):
    """
    Calculate the number of FLOPs for the model.
    """
    sample_batch = next(iter(data_loader))
    sample_batch = {k: v.cuda() for k, v in sample_batch.items()}

    with FlopTensorDispatchMode(model) as ftdm:
        res = model.training_step(sample_batch)["train/loss"]
        flops_forward = copy.deepcopy(ftdm.flop_counts)
        total_forward_flops = sum(get_leaf_values(flops_forward))
        ftdm.reset()
        
        if config.get("use_wandb", False):
            wandb.log({"stats/forward_flops": total_forward_flops}, step=0)
        else:
            print(f"Forward FLOPs: {total_forward_flops}")


def fit(config, model, optimizer, scheduler, train_loader, val_loader, test_loader=None, timing_window_start=100, timing_window_size=500):
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
        torch.cuda.reset_peak_memory_stats(device)
        memory_before_gb = torch.cuda.memory_allocated(device) / (1024**3)

    benchmark_flops(model, train_loader, config)

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
                    wandb.log({"stats/steps_per_second": steps_per_second}, step=global_step)
                else:
                    print(f"Steps per second: {steps_per_second:.2f}")
            
            stat_dict = train_step(model, batch, optimizer, scheduler)
            
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
                wandb.log({f"{k}": v.item() for k, v in stat_dict.items() if "lr" not in k}, step=global_step)
            
            # Validation and checkpointing
            if (global_step + 1) % config["val_every_iter"] == 0:
                train_stats = {f"avg/{k}": v / num_train_batches for k, v in running_train_stats.items()}
                
                running_train_stats = {}
                num_train_batches = 0
                
                val_stats = validate(model, val_loader, config)
                current_val_loss = val_stats['avg/val/loss']
                
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    save_checkpoint(model, optimizer, scheduler, config, best_val_loss, global_step)
                    if not config.get("use_wandb", False):
                        print(f"New best validation loss: {best_val_loss:.4f}, saved checkpoint")
                
                if config.get("use_wandb", False):
                    wandb.log({**train_stats, **val_stats, 'global_step': global_step}, step=global_step)
                else:
                    loss_keys = [k for k in val_stats.keys() if "loss" in k]
                    for k in loss_keys: 
                        print(f"Validation {k}: {val_stats[k]:.4f}")
            
            global_step += 1

    if test_loader is not None and config.get('test', False):
        print("Loading best checkpoint for testing...")
        best_val_loss, best_step = load_checkpoint(model, optimizer, scheduler, config)
        print(f"Loaded checkpoint from step {best_step} with validation loss {best_val_loss:.4f}")
        
        test_stats = validate(model, test_loader, config)
        if config.get("use_wandb", False):
            wandb.log({
                **{f"test/{k.replace('val/', '')}": v for k, v in test_stats.items()},
                'global_step': global_step
            }, step=global_step)
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

    n_influenced = evaluate_interactions_from_batch(model, next(iter(train_loader)), config)
    if config.get("use_wandb", False):
        wandb.log({"stats/runtime": runtime_seconds}, step=0)
        wandb.log({"stats/peak_gpu_memory": peak_memory_gb}, step=0)
        wandb.log({"stats/before_gpu_memory": memory_before_gb}, step=0)
        wandb.log({"stats/after_gpu_memory": memory_after_gb}, step=0)
        wandb.log({"stats/n_influenced": n_influenced}, step=0)
    else:
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

    return model