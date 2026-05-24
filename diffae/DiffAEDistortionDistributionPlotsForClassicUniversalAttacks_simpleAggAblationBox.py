
'''

cd illcond
conda activate dt2
python diffae/DiffAEDistortionDistributionPlotsForClassicUniversalAttacks_simpleAggAblationBox.py --epsilon_list 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.30 0.31


'''

'''

cd illcond
conda activate dt2
python diffae/DiffAEDistortionDistributionPlotsForClassicUniversalAttacks_simpleAggAblationBox.py

# Layer-wise Loss Summation (LLS)

'''

import numpy as np
from matplotlib import pyplot as plt
import torch
from templates import *
from matplotlib.ticker import FuncFormatter

import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='DiffAE celebA training')

# Optional now, since we only use 0.21
parser.add_argument(
    "--epsilon_list",
    type=float,
    nargs='+',
    default=[0.21],
    help="List of epsilon values"
)

args = parser.parse_args()

which_gpu = 7
source_segment = 0

# device = 'cuda:' + str(which_gpu)

def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()


xts = []
for i in range(100):
    xts.append(i * 10000)


# ============================================================
# Attack settings
# ============================================================

attack_types = [
    "simpAgg_l2_kfAdamNoScheduler1",
    "simpAgg_wass_kfAdamNoScheduler1",
    "bsa_kfAdamNoScheduler1",
    "grill_cos_kfAdamNoScheduler1"
]

objective_names = [
    "LLS-L-2",
    "LLS-wass",
    "LLS-cos",
    "GRILL-cos"
]

colors = [
    "blue",
    "orange",
    "red",
    "gold"
]

considered_attack_inds = [0, 1, 2, 3]

all_metric_types = [
    "adv_recons",
    "adv_divs",
    "adv_divs_wass",
    "adv_divs_abs",
    "adv_divs_wass",
    "ssim",
    "psnr"
]

metric_type = all_metric_types[1]

# Only perturbation budget 0.21
desired_norm_l_inf = 0.21


# ============================================================
# Load values for box plot
# ============================================================

boxplot_data = []

for i in considered_attack_inds:

    print("attack_types[i]", attack_types[i])

    ar0 = np.load(
        "diffae/attack_qualitative_untargeted_univ_quantitative/deviations_p/"
        + metric_type
        + "_DiffAE_attack_type"
        + str(attack_types[i])
        + "_norm_bound_"
        + str(desired_norm_l_inf)
        + "_segment_"
        + str(source_segment)
        + ".npy",
        allow_pickle=True
    )

    ar0 = np.array(ar0).astype(float)

    ar0_mean = np.mean(ar0)
    ar0_std = np.std(ar0)

    print("desired_norm_l_inf", desired_norm_l_inf)
    print("ar0_mean", ar0_mean)
    print("ar0_std", ar0_std)
    print()

    for val in ar0:
        boxplot_data.append({
            "Attack Method": objective_names[i],
            "L-2 distance": val
        })

# ============================================================
# Compact but readable plot for two-column paper
# (meant to occupy ~half-column width beside another figure)
# ============================================================

df_boxplot = pd.DataFrame(boxplot_data)

sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(3.6, 2.8))

sns.boxplot(
    data=df_boxplot,
    x="Attack Method",
    y="L-2 distance",
    palette=colors,
    width=0.78,              # larger boxes -> less empty space
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

# ------------------------------------------------------------
# Larger fonts for readability after LaTeX scaling
# ------------------------------------------------------------

ax.set_xlabel("")
ax.set_ylabel("L-2 dist.", fontsize=15)

formatter = FuncFormatter(lambda x, _: f'{x:.2f}')
ax.yaxis.set_major_formatter(formatter)

ax.tick_params(axis='x', labelsize=12, pad=1)
ax.tick_params(axis='y', labelsize=12, pad=1)

plt.xticks(rotation=30)

# ------------------------------------------------------------
# Reduce wasted space
# ------------------------------------------------------------

ax.margins(x=0.015)

ax.grid(True, axis="y", alpha=0.30, linewidth=0.6)
ax.grid(False, axis="x")

sns.despine()

plt.tight_layout(pad=0.2)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

plt.savefig(
    "diffae/grill_damage_distributions_variation/diffAE_bpbSimpleAggDemo_boxplot_0_21_twocol.pdf",
    bbox_inches="tight",
    pad_inches=0.01
)

plt.savefig(
    "diffae/grill_damage_distributions_variation/diffAE_bpbSimpleAggDemo_boxplot_0_21_twocol.png",
    dpi=400,
    bbox_inches="tight",
    pad_inches=0.01
)

plt.show()