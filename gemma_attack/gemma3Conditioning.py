



'''

export CUDA_VISIBLE_DEVICES=4
conda activate gemma3
cd illcond
python gemma_attack/gemma3Conditioning.py 


'''

import os
import sys
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)


def main():
    MODEL_PATH = "gemma_attack/Gemma3-4b"

    os.makedirs("outputsStorage", exist_ok=True)
    os.makedirs("outputsStorage/convergence", exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"device={device}, dtype={dtype}")

    print("Loading processor...")

    print("Loading model...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    print("\n=== MODEL PARAMETERS (name → shape) ===")
    allCondNums = []
    allSmallSingvals = []
    alllargestSingVal = []
    for name, param in model.named_parameters():
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
    ####################################### actual condition numbers

    ####################################### minimum condition numbers
    #####################----------------------------------



    vals = np.array(allSmallSingvals, dtype=float)

    fig, ax = plt.subplots(figsize=(4, 6))  # wider helps a lot
    ax.barh(np.arange(len(vals)), vals, color="blue", alpha=0.7)

    linthresh = 1e-4
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
    fig.savefig("gemma_attack/conditioningAnalysis/gemma3_min_sing_valsCC.png", dpi=300)
    plt.show()
    plt.close()


    #-----------------#

    vals = np.array(alllargestSingVal, dtype=float)

    fig, ax = plt.subplots(figsize=(4, 6))  # wider helps a lot
    ax.barh(np.arange(len(vals)), vals, color="blue", alpha=0.7)

    linthresh = 1e1
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
    fig.savefig("gemma_attack/conditioningAnalysis/gemma3_max_sing_valsCC.png", dpi=300)
    plt.show()




    #-----------------#


    vals = np.array(allCondNums, dtype=float)

    '''# Keep only finite values for axis scaling / ticks
    finite = np.isfinite(vals)
    vals_finite = vals[finite]

    # If you want to *plot* inf/nan as a capped value instead of dropping them:
    cap = np.nanmax(vals_finite) * 1.05
    vals_plot = vals.copy()
    vals_plot[~finite] = cap  # put inf/nan at the right edge'''

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
    xmax = np.max(vals)
    ax.set_xlim(0, xmax * 1.05)

    # decade ticks
    xmin_pos = np.min(vals[vals > 0])
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
    yticks = list(range(1, len(vals), step))
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=28)

    fig.tight_layout()
    fig.savefig("gemma_attack/conditioningAnalysis/CondNumCC.png", dpi=300)
    plt.show()



if __name__ == "__main__":
    main()

