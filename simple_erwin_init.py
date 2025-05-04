import torch
from models import ErwinTransformer
from balltree import build_balltree_with_rotations
import matplotlib.pyplot as plt

config = {
    "c_in": 16,
    "c_hidden": 16,
    "ball_sizes": [16], # 128
    "enc_num_heads": [1,],
    "enc_depths": [2,],
    "dec_num_heads": [],
    "dec_depths": [],
    "strides": [], # no coarsening
    "mp_steps": 0, # no MPNN
    "decode": True, # no decoder
    "dimensionality": 2, # for visualization
    "rotate": 0,
}

model = ErwinTransformer(**config).cuda()

bs = 1
num_points = 64 # 1024

node_features = torch.randn(num_points * bs, config["c_in"]).cuda()
node_positions = torch.rand(num_points * bs, config["dimensionality"]).cuda()

node_features.requires_grad_(True) # to be sure gradient is computed

batch_idx = torch.repeat_interleave(torch.arange(bs), num_points).cuda()
radius = None


def measure_interaction(model, x, pos, i, j, bs, num_points, radius):
    x = x.detach().clone().requires_grad_(True)
    batch_idx = torch.repeat_interleave(torch.arange(bs), num_points).cuda()
    def f(input_x):
        return model(input_x, pos, batch_idx, radius=radius)[j]
    jac = torch.autograd.functional.jacobian(f, x, create_graph=False)
    jac_i = jac[:, i, :]
    return torch.norm(jac_i)

tree_idx, tree_mask, tree_idx_rot = build_balltree_with_rotations(node_positions, batch_idx, config['strides'], config['ball_sizes'], config['rotate'])

i = 0
j0, j1 = None, None
balls = [tree_idx[start:start + config["ball_sizes"][0]] for start in range(0, num_points, config["ball_sizes"][0])]
for ball in balls:
    if i in ball:
        j0 = ball[0] if ball[0] != i else ball[1]
    if i not in ball:
        j1 = ball[0]
    if j0 is not None and j1 is not None:
        break

print(j0, j1)

print(f'Same ball: {measure_interaction(model, node_features, node_positions, i, j0, bs, num_points, radius)}')
print(f'Different ball: {measure_interaction(model, node_features, node_positions, i, j1, bs, num_points, radius)}')

def measure_interaction_batch(model, x, pos, bs, num_points, radius):
    x = x.detach().clone().requires_grad_(True)
    batch_idx = torch.repeat_interleave(torch.arange(bs), num_points).cuda()
    
    out = model(x, pos, batch_idx, radius=radius)
    N, D = out.shape

    influences = torch.zeros((N, N), device=x.device)

    def f(input_x):
        return model(input_x, pos, batch_idx, radius=radius)
    jacobian = torch.autograd.functional.jacobian(f, x, create_graph=False)
    print(f'Jacobian {jacobian.shape}')
    for i in range(N):
        for j in range(N):
            influences[i, j] = (jacobian[j, :, i, :] ** 2).sum()

    return influences

def plot_heatmap(data, filename="heatmap.png"):
    data = data.detach().cpu().numpy()
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.colorbar(label="Influence (sum of squared Jacobian entries)")
    plt.xlabel("Output point index (j)")
    plt.ylabel("Input point index (i)")
    plt.title("Influence Heatmap: dErwin_j/di")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def log_transform(matrix, eps=1e-8): # heatmap uses logarithmic scale, since for points from the same ball it's around e-4 and for i=j around 1
    matrix = matrix.clone()
    # matrix.fill_diagonal_(0)
    return torch.log10(matrix + eps)

def reorder_matrix(matrix, permutation):
    return matrix[permutation][:, permutation]

influences = measure_interaction_batch(model, node_features, node_positions, bs, num_points, radius)
permuted_influences = reorder_matrix(influences, tree_idx)
plot_heatmap(log_transform(permuted_influences))
