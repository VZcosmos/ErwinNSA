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
from erwin.experiments.datasets import MDDataset
from erwin.experiments.wrappers import MDModel


from collections import defaultdict
import wandb
def _size_mb(t): return t.numel()*t.element_size()/1e6
def log_hook(mod, inp, out):
    wandb.log({f"{mod.__class__.__name__}/in_MB": sum(_size_mb(x) for x in inp),
               f"{mod.__class__.__name__}/out_MB": _size_mb(out)})

               
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="erwin", 
                        choices=('mpnn', 'pointtransformer', 'pointnetpp', 'erwin', 'erwin_nsa'),
                        help="Model type (mpnn, pointtransformer, pointnetpp, erwin, erwin_nsa)")
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--size", type=str, default="small",
                        choices=["small", "medium", "large"],
                        help="Model size (tiny, small, base)")
    parser.add_argument("--dilation", type=int, default=1,
                        help="Dilation factor for the dataset")
    parser.add_argument("--num-epochs", type=int, default=100000,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--use-wandb", type=int, default=1,
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--val-every-iter", type=int, default=5000,
                        help="Validation frequency in iterations")
    parser.add_argument("--experiment", type=str, default="md",
                        help="Experiment name")
    parser.add_argument("--test", type=int, default=1,
                        help="Whether to run testing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", type=int, default=0)
    parser.add_argument("--accumulation-steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    
    return parser.parse_args()


erwin_configs = {
    "small": {
        "c_in": 16,
        "c_hidden": 16,
        "enc_num_heads": [2, 4, 8, 16, 32],
        "enc_depths": [2, 2, 2, 6, 2],
        "dec_num_heads": [4, 4, 8, 16],
        "dec_depths": [2, 2, 2, 2],
        "strides": [2, 2, 2, 2],
        "ball_sizes": [128, 128, 128, 64, 32],
        "rotate": 45,
    },
    "medium": {
        "c_in": 16,
        "c_hidden": 32,
        "enc_num_heads": [2, 4, 8, 16, 32],
        "enc_depths": [2, 2, 2, 6, 2],
        "dec_num_heads": [4, 4, 8, 16],
        "dec_depths": [2, 2, 2, 2],
        "strides": [2, 2, 2, 2],
        "ball_sizes": [128, 128, 128, 64, 32],
        "rotate": 45,
    },
    "large": {
        "c_in": 32,
        "c_hidden": 64,
        "enc_num_heads": [2, 4, 8, 16, 32],
        "enc_depths": [2, 2, 2, 6, 2],
        "dec_num_heads": [4, 4, 8, 16],
        "dec_depths": [2, 2, 2, 2],
        "strides": [2, 2, 2, 2],
        "ball_sizes": [128, 128, 128, 64, 32],
        "rotate": 45,
    },
}

erwin_nsa_configs = {
    "small": {
        "c_in": 16,
        "c_hidden": 64,
        "rotate": 45,
        "depth": 6,
        "num_heads": 16,
        "compress_ball_size": 32,
        "local_ball_size": 32,
        "num_selected_blocks": 16,
        "min_nsa_heads": 16,
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

    train_dataset = MDDataset(
        directory=f"{args.data_path}/polymer_train",
        split=f"{args.data_path}/splits/train.txt",
        seq_len=16,
        traj_len=10000,
    )

    valid_dataset = MDDataset(
        directory=f"{args.data_path}/polymer_train",
        split=f"{args.data_path}/splits/val.txt",
        seq_len=16,
        traj_len=10000,
    )
    # Choose a subset of valid_dataset randomly
    valid_dataset = torch.utils.data.Subset(valid_dataset, torch.randperm(len(valid_dataset))[:5000])

    test_dataset = MDDataset(
        directory=f"{args.data_path}/polymer_test",
        split=f"{args.data_path}/splits/test_class2.txt",
        seq_len=16,
        traj_len=1000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=16,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
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

    # ###### DEBUGGING CODE ######
    # train_batch = next(iter(train_loader))
    # val_batch = next(iter(valid_loader))
    # test_batch = next(iter(test_loader))

    # print("--"*20)
    # print("Training batch")
    # for k, v in train_batch.items():
    #     if isinstance(v, torch.Tensor):
    #         print(f"{k}: {v.shape} ({v.dtype})")
    #     else:
    #         print(f"{k}: {v}")
    # print("Samples in training set:", len(train_dataset))
    # print("--"*20)
    # print("Validation batch")
    # for k, v in val_batch.items():
    #     if isinstance(v, torch.Tensor):
    #         print(f"{k}: {v.shape} ({v.dtype})")
    #     else:
    #         print(f"{k}: {v}")
    # print("Samples in validation set:", len(valid_dataset))
    # print("--"*20)
    # print("Testing batch")
    # for k, v in test_batch.items():
    #     if isinstance(v, torch.Tensor):
    #         print(f"{k}: {v.shape} ({v.dtype})")
    #     else:
    #         print(f"{k}: {v}")
    # print("Samples in testing set:", len(test_dataset))
    # print("--"*20)

    # if args.model == "erwin":
    #     model_config = erwin_configs[args.size]
    # elif args.model == "erwin_nsa":
    #     model_config = erwin_nsa_configs[args.size]
    # else:
    #     raise NotImplementedError(f"Unknown model: {args.model}")

    # dynamics_model = model_cls[args.model](**model_config)
    # if args.profile:
    #     # THIS COULD SLOW DOWN TRAINING BUT IS NEEDED FOR PROFILING
    #     print("Memory profiling enabled!")
    #     torch.cuda.memory._record_memory_history()
    # torch.cuda.reset_peak_memory_stats(torch.device("cuda"))
    # model = MDModel(seq_len=train_dataset.seq_len, dynamics_model=dynamics_model).cuda()

    # train_batch = {k: v.cuda() for k, v in train_batch.items()}
    # val_batch = {k: v.cuda() for k, v in val_batch.items()}
    # test_batch = {k: v.cuda() for k, v in test_batch.items()}

    # print("--"*20)
    # start_time = torch.cuda.Event(enable_timing=True)
    # end_time = torch.cuda.Event(enable_timing=True)
    # print("Model Traning Step:")
    # torch.cuda.synchronize()
    
    # start_time.record()
    # for _ in range(10):
    #     _ = model.training_step(train_batch)
    # torch.cuda.synchronize()
    # end_time.record()
    # elapsed_time = start_time.elapsed_time(end_time) / 10
    # print(f"Average training step time: {elapsed_time:.2f} ms")

    # print("\n")
    # print("--"*20)
    # print("Model Validation Step:")
    # torch.cuda.synchronize()
    # start_time.record()
    # for _ in range(10):
    #     _ = model.validation_step(val_batch)
    # torch.cuda.synchronize()
    # end_time.record()
    # elapsed_time = start_time.elapsed_time(end_time) / 10
    # print(f"Average validation step time: {elapsed_time:.2f} ms")
    # print("\n")
    # print("--"*20)
    # print("Model Testing Step:")
    # torch.cuda.synchronize()
    # start_time.record()
    # for _ in range(10):
    #     _ = model.validation_step(test_batch)
    # torch.cuda.synchronize()
    # end_time.record()
    # elapsed_time = start_time.elapsed_time(end_time) / 10
    # print(f"Average testing step time: {elapsed_time:.2f} ms")

    # import sys; sys.exit(0)  # Exit early for debugging purposes
    # ##### END OF DEBUGGING CODE ######

    if args.model == "erwin":
        model_config = erwin_configs[args.size]
    elif args.model == "erwin_nsa":
        model_config = erwin_nsa_configs[args.size]
    else:
        raise NotImplementedError(f"Unknown model: {args.model}")

    dynamics_model = model_cls[args.model](**model_config)
    if args.profile:
        # THIS COULD SLOW DOWN TRAINING BUT IS NEEDED FOR PROFILING
        print("Memory profiling enabled!")
        torch.cuda.memory._record_memory_history()
    torch.cuda.reset_peak_memory_stats(torch.device("cuda"))
    model = MDModel(seq_len=train_dataset.seq_len, dynamics_model=dynamics_model).cuda()
    # DO NOT UNCOMMENT UNTIL FIXED.
    # The problem is that the log_hook advances the step counter too much,
    # and then all our logs inside the fit() func will not be logged.
    # for m in model.modules():
    #     if isinstance(m, SparseAttention):
    #         m.register_forward_hook(log_hook)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-7)

    config = vars(args)
    config.update(model_config)

    fit(config, model, optimizer, scheduler, train_loader, valid_loader, test_loader, 100, 300, dataset_type="md", accumulation_steps=args.accumulation_steps)
