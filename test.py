import sys
sys.path.append("../../")

import argparse
import torch
torch.set_float32_matmul_precision("high")
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from experiments.datasets import ShapenetCarDataset
from experiments.wrappers import ShapenetCarModel

train_dataset = ShapenetCarDataset(
    data_path="../nico_directory/shapenet_car_processed",
    split="train",
    knn=8,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=train_dataset.collate_fn,
    num_workers=1,
)

batch = next(iter(train_loader))
batch = {k: v for k, v in batch.items()}

import pdb; pdb.set_trace()