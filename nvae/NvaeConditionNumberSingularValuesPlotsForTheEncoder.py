import torch
import torch.nn as nn
from model import AutoEncoder
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import random

from torchvision import datasets, transforms
import os
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

#device = "cuda:1" if torch.cuda.is_available() else "cpu"


'''

0, 1, 2, 3

conda deactivate
conda deactivate
conda deactivate
cd NVAE/
export CUDA_VISIBLE_DEVICES=3
source nvaeenv1/bin/activate
python nvae/NvaeConditionNumberSingularValuesPlotsForTheEncoder.py


'''

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




# Replace the placeholder values with your actual checkpoint path and parameters
checkpoint_path = '/mdadm0/chethan_krishnamurth/NVAE/pretrained_checkpoint/checkpoint.pt'
save_path = '/path/to/save'
eval_mode = 'sample'  # Choose between 'sample', 'evaluate', 'evaluate_fid'
batch_size = 0

# Load the model
checkpoint = torch.load(checkpoint_path, map_location='cpu')
args = checkpoint['args']

if not hasattr(args, 'ada_groups'):
    args.ada_groups = False

if not hasattr(args, 'min_groups_per_scale'):
    args.min_groups_per_scale = 1

if not hasattr(args, 'num_mixture_dec'):
    args.num_mixture_dec = 10

arch_instance = utils.get_arch_cells(args.arch_instance)  # You may need to replace this with the actual function or import it
model = AutoEncoder(args, None, arch_instance)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model = model.cuda()
model.eval()





allCondNums = []
allSmallSingvals = []
alllargestSingVal = []
for name, param in model.enc_tower.named_parameters():
    print(f"{name:60s} {tuple(param.shape)}")

    if 'weight' in name and len(param.shape)>1:
        print("param.shape", param.shape)
        print("len(param.shape)", len(param.shape))
        W_matrix = param.view(param.shape[0], -1)  # Flatten kernels into a 2D matrix
        U, S, Vt = torch.linalg.svd(W_matrix.float(), full_matrices=False)
        condition_number = S.max() / S.min()
        print("condition_number", condition_number)
        allCondNums.append(condition_number.item())
        allSmallSingvals.append(S.min().item())
        alllargestSingVal.append(S.max().item())

print("allCondNums", allCondNums)
print('allSmallSingvals', allSmallSingvals)
print('alllargestSingVal', alllargestSingVal)
largestAmongMaxes = max(alllargestSingVal)
print("largestAmongMaxes", largestAmongMaxes)
####################################### actual condition numbers

filtered_g_cond_nums = allCondNums # [value for value in actual_cond_nums_array if value != 1.0]
plt.figure(figsize=(4, 6))  # Set figure size
#plt.barh(range(len(filtered_g_cond_nums)), filtered_g_cond_nums, color='blue', alpha=0.7)
plt.barh(range(len(filtered_g_cond_nums)),
    filtered_g_cond_nums,
    color='blue',
    edgecolor='blue',
    linewidth=2)

plt.xscale("symlog", linthresh=1000.0)
plt.xlim(left=0)

plt.ylabel("Layer index", fontsize=28)
plt.xlabel("$\kappa$", fontsize=28)
def sci_notation_formatter(y, _):
    if y == 0:
        return "0"  # Display zero as "0" instead of "0e0"
    return f"{int(y):.0e}".replace("+", "").replace("e0", "e")
formatter = ticker.FuncFormatter(sci_notation_formatter)
plt.gca().xaxis.set_major_formatter(formatter)
plt.xticks(fontsize=28, rotation=45)
#plt.yticks(range(len(filtered_g_cond_nums)), range(len(filtered_g_cond_nums)), fontsize=24)
stepY = 100
yticks = list(range(1, len(filtered_g_cond_nums), stepY))

plt.yticks(yticks, yticks, fontsize=28)

plt.tight_layout()  # Adjust layout to prevent cutoff of labels
plt.savefig("nvae/conditionNumberSingularValuesPlots/nvae_conditioning_chart_actual_k.png")
plt.show()
plt.close()

####################################### minimum condition numbers
#####################----------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

vals = np.array(allSmallSingvals, dtype=float)

fig, ax = plt.subplots(figsize=(4, 6))  # wider helps a lot
ax.barh(np.arange(len(vals)), vals, color="blue", alpha=0.7)

linthresh = 1e-3
ax.set_xscale("symlog", linthresh=linthresh)

# set x-limits to your data range (DON'T force left=0 unless you need it)
pos = vals[vals > 0]
xmin = pos.min()
xmax = pos.max()
ax.set_xlim(0, xmax * 1.05)

# --- choose a small set of decade ticks within data range ---
dmin = int(np.floor(np.log10(max(xmin, linthresh))))
dmax = int(np.ceil(np.log10(xmax)))
decade_ticks = [10.0**k for k in range(dmin, dmax + 1)]

# include 0 tick + a few decades only
ticks = [0.0] + decade_ticks
ax.xaxis.set_major_locator(FixedLocator(ticks))

def exp_formatter(x, _):
    if x == 0:
        return "0"
    return f"{x:.0e}".replace("+", "").replace("e0", "e")  # 1e-3 style

ax.xaxis.set_major_formatter(FuncFormatter(exp_formatter))

ax.set_ylabel("Layer index", fontsize=28)
ax.set_xlabel(r"$\sigma_{min}$", fontsize=28)

ax.tick_params(axis="x", labelsize=22, rotation=45)  # rotation=0 avoids the pile-up

step = 100
yticks = list(range(1, len(vals), step))
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=28)

fig.tight_layout()
fig.savefig("nvae/conditionNumberSingularValuesPlots/nvae_min_sing_valsCC.png", dpi=300)
plt.show()
plt.close()


#-----------------#

vals = np.array(alllargestSingVal, dtype=float)

fig, ax = plt.subplots(figsize=(4, 6))  # wider helps a lot
ax.barh(np.arange(len(vals)), vals, color="blue", alpha=0.7)

linthresh = 1e-1
ax.set_xscale("symlog", linthresh=linthresh)

# set x-limits to your data range (DON'T force left=0 unless you need it)
pos = vals[vals > 0]
xmin = pos.min()
xmax = pos.max()
ax.set_xlim(0, xmax * 1.05)

# --- choose a small set of decade ticks within data range ---
dmin = int(np.floor(np.log10(max(xmin, linthresh))))
dmax = int(np.ceil(np.log10(xmax)))
decade_ticks = [10.0**k for k in range(dmin, dmax + 1)]

# include 0 tick + a few decades only
ticks = [0.0] + decade_ticks
ax.xaxis.set_major_locator(FixedLocator(ticks))

def exp_formatter(x, _):
    if x == 0:
        return "0"
    return f"{x:.0e}".replace("+", "").replace("e0", "e")  # 1e-3 style

ax.xaxis.set_major_formatter(FuncFormatter(exp_formatter))

ax.set_ylabel("Layer index", fontsize=28)
ax.set_xlabel(r"$\sigma_{max}$", fontsize=28)

ax.tick_params(axis="x", labelsize=22, rotation=45)  # rotation=0 avoids the pile-up

step = 100
yticks = list(range(1, len(vals), step))
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=28)

fig.tight_layout()
fig.savefig("nvae/conditionNumberSingularValuesPlots/nvae_max_sing_valsCC.png", dpi=300)
plt.show()




#-----------------#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

vals = np.array(allCondNums, dtype=float)

# Keep only finite values for axis scaling / ticks
finite = np.isfinite(vals)
vals_finite = vals[finite]

# If you want to *plot* inf/nan as a capped value instead of dropping them:
cap = np.nanmax(vals_finite) * 1.05
vals_plot = vals.copy()
#vals_plot[~finite] = cap  # put inf/nan at the right edge

fig, ax = plt.subplots(figsize=(4, 6))
#ax.barh(np.arange(len(allCondNums)), allCondNums, color="blue", alpha=0.7)

ax.barh(range(len(allCondNums)),
    allCondNums,
    color='blue',
    edgecolor='blue',
    linewidth=2)


linthresh = 1e3
ax.set_xscale("symlog", linthresh=linthresh)

# Safe limits (finite only)
xmax = np.nanmax(vals_finite)
ax.set_xlim(0, xmax * 1.05)

# decade ticks
xmin_pos = np.nanmin(vals_finite[vals_finite > 0])
dmin = int(np.floor(np.log10(max(xmin_pos, linthresh))))
dmax = int(np.ceil(np.log10(xmax)))
decade_ticks = [10.0**k for k in range(dmin, dmax + 1)]

ticks = [0.0] + decade_ticks
ax.xaxis.set_major_locator(FixedLocator(ticks))

def exp_formatter(x, _):
    if x == 0:
        return "0"
    return f"{x:.0e}".replace("+", "").replace("e0", "e")

ax.xaxis.set_major_formatter(FuncFormatter(exp_formatter))

ax.set_ylabel("Layer index", fontsize=28)
ax.set_xlabel(r"$\kappa$", fontsize=28)
ax.tick_params(axis="x", labelsize=22, rotation=45)

step = 100
yticks = list(range(1, len(vals_plot), step))
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=28)

fig.tight_layout()
fig.savefig("nvae/conditionNumberSingularValuesPlots/CondNumCC.png", dpi=300)
plt.show()