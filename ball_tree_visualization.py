import torch
from balltree import build_balltree
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = "/gpfs/home5/scur2687/nico_directory/shapenet_car_processed/param0/100715345ee54d7ae38b52b4ee9d36a3/mesh_points.th"
points = torch.load(path)
batch_idx = torch.zeros(points.shape[0], dtype=torch.long)

tree_idx, tree_mask = build_balltree(points, batch_idx)
grouped_points = points[tree_idx]

level_to_node_size = lambda level: 2**level


def visualize_level(groups, level_idx):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for group in groups:
        group_np = group.numpy()
        ax.scatter(group_np[:,2], group_np[:, 0], group_np[:, 1]) #apparently shapenet mesh points are in (width,height,length) order
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_zlim(-0.5,2)
    plt.title(f"Level {level_idx}")
    plt.savefig(f"visualization_level_{level_idx}.png")
    plt.close()
    print(f"Saved visualization_level_{level_idx}.png with matplotlib")


for level in range(0,6):
    node_size = level_to_node_size(level)
    groups = grouped_points.reshape(-1, node_size, 3)
    visualize_level(groups, level)
