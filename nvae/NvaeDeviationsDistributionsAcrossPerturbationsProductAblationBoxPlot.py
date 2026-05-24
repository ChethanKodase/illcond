'''

conda deactivate
conda deactivate
conda deactivate
export CUDA_VISIBLE_DEVICES=7
cd NVAE/
source nvaeenv1/bin/activate
cd ..
cd illcond/
python nvae/NvaeDeviationsDistributionsAcrossPerturbationsProductAblationBoxPlot.py 

'''


import torch
import torch.nn as nn
from model import AutoEncoder
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
from torchvision import datasets, transforms
import os

# ============================================================
# Setup
# ============================================================

checkpoint_path = '../NVAE/pretrained_checkpoint/checkpoint.pt'

checkpoint = torch.load(checkpoint_path, map_location='cpu')
args = checkpoint['args']

if not hasattr(args, 'ada_groups'):
    args.ada_groups = False

if not hasattr(args, 'min_groups_per_scale'):
    args.min_groups_per_scale = 1

if not hasattr(args, 'num_mixture_dec'):
    args.num_mixture_dec = 10

arch_instance = utils.get_arch_cells(args.arch_instance)

model = AutoEncoder(args, None, arch_instance)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model = model.cuda()
model.eval()

# ============================================================
# Attack settings
# ============================================================

desired_norm_l_inf = 0.025

attck_types = [
    "grill_l2_kf_allSum",
    "grill_wass_kf_allSum",
    "grill_cos_kf_allSum",
    "grill_l2_kf"
]

objective_names = [
    "LLS-L-2",
    "LLS-wass",
    "LLS-cos",
    "GRILL-L-2"
]

# ============================================================
# Data loading
# ============================================================

img_list = os.listdir('../data_cel1/smile/')
img_list.extend(os.listdir('../data_cel1/no_smile/'))

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

batch_size = 1000

celeba_data = datasets.ImageFolder('../data_cel1', transform=transform)

split_train_frac = 0.95

train_set, test_set = torch.utils.data.random_split(
    celeba_data,
    [
        int(len(img_list) * split_train_frac),
        len(img_list) - int(len(img_list) * split_train_frac)
    ]
)

train_data_size = len(train_set)
test_data_size = len(test_set)

print('train_data_size', train_data_size)
print('test_data_size', test_data_size)

trainLoader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True
)

testLoader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=True
)

del trainLoader

# ============================================================
# Compute L-2 distances for box plot
# ============================================================

boxplot_data = []

with torch.no_grad():

    for idx, (source_im, _) in enumerate(testLoader):
        source_im = source_im.cuda()
        break

    mi, ma = source_im.min(), source_im.max()

    for i in range(len(attck_types)):

        optimized_noise = torch.load(
            "nvae/univ_attack_storage/NVAE_attack_type"
            + str(attck_types[i])
            + "_norm_bound_"
            + str(desired_norm_l_inf)
            + "_.pt"
        )

        print("source_im.shape", source_im.shape)
        print("optimized_noise.shape", optimized_noise.shape)

        normalized_attacked = torch.clamp(
            source_im + optimized_noise,
            mi,
            ma
        )

        print(
            "1 normalized_attacked.min(), normalized_attacked.max()",
            normalized_attacked.min(),
            normalized_attacked.max()
        )

        normalized_attacked = (
            normalized_attacked - normalized_attacked.min()
        ) / (
            normalized_attacked.max() - normalized_attacked.min()
        )

        print(
            "2 normalized_attacked.min(), normalized_attacked.max()",
            normalized_attacked.min(),
            normalized_attacked.max()
        )

        adv_logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps = model(
            normalized_attacked
        )

        reconstructed_output = model.decoder_output(adv_logits)
        adv_gen = reconstructed_output.sample()

        l2_distance_per_image = torch.norm(
            normalized_attacked - adv_gen,
            p=2,
            dim=[1, 2, 3]
        )

        print("i", i)
        print("attck_types[i]", attck_types[i])
        print("desired_norm_l_inf", desired_norm_l_inf)
        print("l2_distance_per_image_mean", l2_distance_per_image.mean().item())
        print("l2_distance_per_image_std", l2_distance_per_image.std().item())

        for val in l2_distance_per_image.detach().cpu().numpy():
            boxplot_data.append({
                "Attack Method": objective_names[i],
                "L-2 distance": val
            })

del testLoader

# ============================================================
# Compact but readable plot for two-column paper
# Same settings as DiffAE plot
# ============================================================

df_boxplot = pd.DataFrame(boxplot_data)

colors = ['blue', 'orange', 'red', 'lime']

sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(3.6, 2.8))

sns.boxplot(
    data=df_boxplot,
    x="Attack Method",
    y="L-2 distance",
    palette=colors,
    width=0.78,
    linewidth=1.3,
    showfliers=False,
    ax=ax
)

sns.stripplot(
    data=df_boxplot,
    x="Attack Method",
    y="L-2 distance",
    color="black",
    alpha=0.28,
    jitter=0.14,
    size=2.4,
    ax=ax
)

ax.set_xlabel("")
ax.set_ylabel("L-2 dist.", fontsize=15)

from matplotlib.ticker import FuncFormatter

formatter = FuncFormatter(lambda x, _: f'{x:.2f}')

ax.yaxis.set_major_formatter(formatter)

ax.tick_params(axis='x', labelsize=12, pad=1)

ax.tick_params(axis='y', labelsize=12, pad=1)

plt.xticks(rotation=30)

ax.margins(x=0.015)

ax.grid(True, axis="y", alpha=0.30, linewidth=0.6)
ax.grid(False, axis="x")

sns.despine()

plt.tight_layout(pad=0.2)

plt.savefig(
    "nvae/grill_perturbation_analysis/NvaeOutputDistortionBoxplot_0_025_twocol.pdf",
    bbox_inches="tight",
    pad_inches=0.01
)

plt.savefig(
    "nvae/grill_perturbation_analysis/NvaeOutputDistortionBoxplot_0_025_twocol.png",
    dpi=400,
    bbox_inches="tight",
    pad_inches=0.01
)

plt.show()