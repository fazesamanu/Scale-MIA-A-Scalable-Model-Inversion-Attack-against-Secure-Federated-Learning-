# This file is used to plot the original and reconstructed samples
# For convenience, we have plotted all figures and they can be found in the figs folder

import argparse
import os
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mse_fn = MSELoss()

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--iter", type=int, default=1)

args = parser.parse_args()

batch_size = args.batch_size
iter_num = args.iter

dim = 28
channel = 1

# Result directory
save_dir = f"../results/fmnist/iter{iter_num}"
os.makedirs(save_dir, exist_ok=True)

# Load tensors
input_data = torch.load(
    f"data/data_batch_{batch_size}.pt",
    map_location=device
)

recovered = torch.load(
    f"data/recoved_{batch_size}.pt",
    map_location=device
)

sorted_list = torch.load(
    f"data/list_{batch_size}.pt",
    map_location=device
)

input_data = input_data.reshape(batch_size, channel, dim, dim)
recovered = recovered.reshape(batch_size, channel, dim, dim)

# Display max 64 images
display_num = min(64, batch_size)

# ---------------- ORIGINAL ----------------
plt.figure(figsize=(16, 16))

img = torch.zeros((dim, dim, channel))

for i in range(display_num):

    for j in range(channel):
        img[:, :, j] = input_data[
            sorted_list[batch_size - 1 - i],
            j,
            :,
            :
        ]

    plt.subplot(8, 8, i + 1)
    plt.imshow(img.detach().numpy(), cmap="gray")
    plt.axis("off")

plt.suptitle(
    f"Original Images - Batch {batch_size} - Iter {iter_num}",
    fontsize=18
)

plt.tight_layout()

original_path = os.path.join(
    save_dir,
    f"bs{batch_size}_original.png"
)

plt.savefig(
    original_path,
    dpi=300,
    bbox_inches="tight"
)

# ---------------- RECONSTRUCTED ----------------
plt.figure(figsize=(16, 16))

img = torch.zeros((dim, dim, channel))

for i in range(display_num):

    for j in range(channel):
        img[:, :, j] = recovered[i, j, :, :]

    plt.subplot(8, 8, i + 1)
    plt.imshow(img.detach().numpy(), cmap="gray")
    plt.axis("off")

plt.suptitle(
    f"Reconstructed Images - Batch {batch_size} - Iter {iter_num}",
    fontsize=18
)

plt.tight_layout()

reconstructed_path = os.path.join(
    save_dir,
    f"bs{batch_size}_reconstructed.png"
)

plt.savefig(
    reconstructed_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nSaved:")
print(original_path)
print(reconstructed_path)
