



'''

export CUDA_VISIBLE_DEVICES=4
conda activate gemma3
cd /data1/chethan/gemma_attack
python gemma3Conditioning.py 


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
    MODEL_PATH = "/data1/chethan/gemma_attack/Gemma3-4b"

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

    filtered_g_cond_nums = allCondNums # [value for value in actual_cond_nums_array if value != 1.0]
    plt.figure(figsize=(4, 6))  # Set figure size
    #plt.barh(range(len(filtered_g_cond_nums)), filtered_g_cond_nums, color='blue', alpha=0.7)
    plt.barh(range(len(filtered_g_cond_nums)),
        filtered_g_cond_nums,
        color='blue',
        edgecolor='blue',
        linewidth=2)


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
    step = 100
    yticks = list(range(1, len(filtered_g_cond_nums), step))

    plt.yticks(yticks, yticks, fontsize=28)

    plt.tight_layout()  # Adjust layout to prevent cutoff of labels
    plt.savefig("/data1/chethan/gemma_attack/conditioningAnalysis/gemma3_conditioning_chart_actual_k.png")
    plt.show()
    plt.close()

    ####################################### minimum condition numbers

    filtered_g_cond_nums = allSmallSingvals #[value for value in min_sing_vals_array if value != 1e10]
    print("min singular values ", filtered_g_cond_nums)
    plt.figure(figsize=(4, 6))  # Set figure size
    #plt.barh(range(len(filtered_g_cond_nums)), filtered_g_cond_nums, color='blue', alpha=0.7)
    plt.barh(range(len(filtered_g_cond_nums)),
         filtered_g_cond_nums,
         color='blue',
         edgecolor='blue',
         linewidth=2)

    plt.ylabel("Layer index", fontsize=28)
    plt.xlabel("$\sigma_{min}$", fontsize=28)
    '''def sci_notation_formatter(y, _):
        if y == 0:
            return "0"  # Display zero as "0" instead of "0e0"
        return f"{int(y):.0e}".replace("+", "").replace("e0", "e")'''
    #formatter = ticker.FuncFormatter(sci_notation_formatter)
    #plt.gca().xaxis.set_major_formatter(formatter)
    plt.xticks(fontsize=28, rotation=45)
    #plt.yticks(range(len(filtered_g_cond_nums)), range(len(filtered_g_cond_nums)), fontsize=24)
    step = 100
    yticks = list(range(1, len(filtered_g_cond_nums), step))
    plt.yticks(yticks, yticks, fontsize=28)
    plt.tight_layout()  # Adjust layout to prevent cutoff of labels
    plt.savefig("/data1/chethan/gemma_attack/conditioningAnalysis/gemma3_min_sing_vals.png")
    plt.show()

    print()

    filtered_g_cond_nums = alllargestSingVal #[value for value in max_sing_vals_array if value != 1e-10]
    print("max singular values ", filtered_g_cond_nums)
    plt.figure(figsize=(4, 6))  # Set figure size
    #plt.barh(range(len(filtered_g_cond_nums)), filtered_g_cond_nums, color='blue', alpha=0.7, linewidth=2)

    plt.barh(range(len(filtered_g_cond_nums)),
         filtered_g_cond_nums,
         color='blue',
         edgecolor='blue',
         linewidth=2)
    
    plt.ylabel("Layer index", fontsize=28)
    plt.xlabel("$\sigma_{max}$", fontsize=28)
    '''def sci_notation_formatter(y, _):
        if y == 0:
            return "0"  # Display zero as "0" instead of "0e0"
        return f"{int(y):.0e}".replace("+", "").replace("e0", "e")'''
    #formatter = ticker.FuncFormatter(sci_notation_formatter)
    #plt.gca().xaxis.set_major_formatter(formatter)
    plt.xticks(fontsize=28, rotation=45)
    #plt.yticks(range(len(filtered_g_cond_nums)), range(len(filtered_g_cond_nums)), fontsize=24)
    step = 100
    yticks = list(range(1, len(filtered_g_cond_nums), step))
    plt.yticks(yticks, yticks, fontsize=28)
    plt.tight_layout()  # Adjust layout to prevent cutoff of labels
    plt.savefig("/data1/chethan/gemma_attack/conditioningAnalysis/gemma3_max_sing_vals.png")
    plt.show()


if __name__ == "__main__":
    main()

