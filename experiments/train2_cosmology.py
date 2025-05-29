import sys
sys.path.append("../../")

import argparse
import torch
torch.set_float32_matmul_precision("high")
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from erwin.training import fit
from erwin.models.erwin import ErwinTransformer, NSABallformer
from erwin.models.native_sparse_attention_clean import SparseAttention
from erwin.experiments.datasets import CosmologyDataset
from erwin.experiments.wrappers import CosmologyModel

from collections import defaultdict
import wandb
def _size_mb(t): return t.numel()*t.element_size()/1e6
def log_hook(mod, inp, out):
    wandb.log({f"{mod.__class__.__name__}/in_MB": sum(_size_mb(x) for x in inp),
               f"{mod.__class__.__name__}/out_MB": _size_mb(out)})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="erwin",
                        choices=('mpnn', 'pointtransformer', 'pointnetpp', 'erwin', 'erwin_nsa'))
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--size", type=str, default="small",
                        choices=('small', 'medium', 'large'))
    parser.add_argument("--num-samples", type=int, default=8192,
                        help="Number of samples for training")
    parser.add_argument("--num-epochs", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--use-wandb", type=int, default=1)
    parser.add_argument("--profile", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val-every-iter", type=int, default=500,
                        help="Validation frequency")
    parser.add_argument("--experiment", type=str, default="cosmology",
                        help="Experiment name in wandb")
    parser.add_argument("--test", action="store_true", default=True,
                        help="Whether to run testing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--accumulation-steps", type=int, default=1,
                        help="Number of gradient accumulation steps")


    return parser.parse_args()

erwin_configs = {
    "small": {
        "c_in": 32,
        "c_hidden": 32,
        "enc_num_heads": [2, 4, 8, 16],
        "enc_depths": [2, 2, 6, 2],
        "dec_num_heads": [2, 4, 8],
        "dec_depths": [2, 2, 2],
        "strides": [2, 2, 2],
        "ball_sizes": [256, 256, 256, 256],
	"rotate": 45,
    },
    "medium": {
        "c_in": 64,
        "c_hidden": 64,
        "enc_num_heads": [2, 4, 8, 16],
        "enc_depths": [2, 2, 6, 2],
        "dec_num_heads": [2, 4, 8],
        "dec_depths": [2, 2, 2],
        "strides": [2, 2, 2],
        "ball_sizes": [256, 256, 256, 256],
    },
    "large": {
        "c_in": 128,
        "c_hidden": 128,
        "enc_num_heads": [2, 4, 8, 16],
        "enc_depths": [2, 2, 6, 2],
        "dec_num_heads": [2, 4, 8],
        "dec_depths": [2, 2, 2],
        "strides": [2, 2, 2],
        "ball_sizes": [256, 256, 256, 256],
	"rotate": 45,
    },
}

erwin_nsa_configs = {
    "small": {
        "c_in": 64,
        "c_hidden": 64,
        "rotate": 45,
        "depth": 12,
        "num_heads": 16,
        "compress_ball_size": 32,
        "local_ball_size": 128,
        "num_selected_blocks": 16,
        "min_nsa_heads": 16,
        "num_layers": 1,
    },
}

model_cls = {
    "erwin": ErwinTransformer,
    "erwin_nsa": NSABallformer,
}


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_dataset = CosmologyDataset(
        task='node', 
        split='train', 
        num_samples=args.num_samples, 
        tfrecords_path=args.data_path, 
        knn=10,
    )
    val_dataset = CosmologyDataset(
        task='node', 
        split='val', 
        num_samples=512, 
        tfrecords_path=args.data_path, 
        knn=10,
    )
    test_dataset = CosmologyDataset(
        task='node', 
        split='test', 
        num_samples=512, 
        tfrecords_path=args.data_path, 
        knn=10,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=16,  
    )
    
    valid_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=16,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=16,
    )

    if args.model == "erwin":
        model_config = erwin_configs[args.size]
    elif args.model == "erwin_nsa":
        model_config = erwin_nsa_configs[args.size]
    else:
        raise NotImplementedError(f"Unknown model: {args.model}")
    dynamic_model = model_cls[args.model](**model_config)
    if args.profile:
        # THIS COULD SLOW DOWN TRAINING BUT IS NEEDED FOR PROFILING
        print("Memory profiling enabled!")
        torch.cuda.memory._record_memory_history()

    torch.cuda.reset_peak_memory_stats(torch.device("cuda"))

    model = CosmologyModel(dynamic_model).cuda()
    # DO NOT UNCOMMENT UNTIL FIXED.
    # The problem is that the log_hook advances the step counter too much,
    # and then all our logs inside the fit() func will not be logged.
    # for m in model.modules():
    #     if isinstance(m, SparseAttention):
    #         m.register_forward_hook(log_hook)

    #model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    config = vars(args)
    config.update(model_config)

    fit(config, model, optimizer, scheduler, train_loader, valid_loader, test_loader, 100, 200, dataset_type="cosmology")
