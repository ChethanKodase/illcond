


#%load_ext autoreload
#%autoreload 2

'''




cd alma
conda activate dt2
python diffae/DiffAEconditionAnalysisPlots.py --which_gpu 7 --diffae_checkpoint diffae/checkpoints


####################################################################################################################################



'''


from templates import *
import matplotlib.pyplot as plt
import torch.optim as optim

from torch.nn import DataParallel
import torch.nn.functional as F

from torch.utils.data import DataLoader

import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import argparse

parser = argparse.ArgumentParser(description='DiffAE celebA training')

parser.add_argument('--which_gpu', type=int, default=0, help='Index of the GPU to use (0-N)')
parser.add_argument('--diffae_checkpoint', type=str, default=5, help='Type of attack')

args = parser.parse_args()



which_gpu = args.which_gpu
diffae_checkpoint = args.diffae_checkpoint


device = 'cuda:'+str(which_gpu)+''


conf = ffhq256_autoenc()

#conf = ffhq256_autoenc_latent()
print(conf.name)
model = LitModel(conf)
state = torch.load(f"{diffae_checkpoint}/{conf.name}/last.ckpt", map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device);



import torch

act_cond_num = []
min_sing_vals = []
max_sing_vals = []

def get_layer_pert_recon(model):
    g_cond_nums = []
    for i, block in enumerate(model.ema_model.encoder.input_blocks):  

        b_cond_nums = []
        for name, param in block.named_parameters():
            if "weight" in name:
                original_param_wt = param.clone()
                if (len(original_param_wt.shape)==4):
                    W_matrix = original_param_wt.view(original_param_wt.shape[0], -1)  # Flatten kernels into a 2D matrix
                    U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
                    condition_number = S.max() / S.min()
                    act_cond_num.append(condition_number.item())
                    min_sing_vals.append(S.min().item())
                    max_sing_vals.append(S.max().item())

                    b_cond_nums.append(condition_number.item())
                else:
                    b_cond_nums.append(1.0)
                    act_cond_num.append(1.0)
                    min_sing_vals.append(1e10)
                    max_sing_vals.append(1e-10)



        b_cond_nums = np.array(b_cond_nums)
        b_mean_cond = np.mean(b_cond_nums)
        g_cond_nums.append(b_mean_cond)

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        b_cond_nums = []
        for name, param in block.named_parameters():
            if "weight" in name:
                original_param_wt = param.clone()
                if (len(original_param_wt.shape)==4):
                    W_matrix = original_param_wt.view(original_param_wt.shape[0], -1)  # Flatten kernels into a 2D matrix
                    U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
                    condition_number = S.max() / S.min()
                    act_cond_num.append(condition_number.item())
                    min_sing_vals.append(S.min().item())
                    max_sing_vals.append(S.max().item())
                    b_cond_nums.append(condition_number.item())
                else:
                    b_cond_nums.append(1.0)
                    act_cond_num.append(1.0)
                    min_sing_vals.append(1e10)
                    max_sing_vals.append(1e-10)

        b_cond_nums = np.array(b_cond_nums)
        b_mean_cond = np.mean(b_cond_nums)
        g_cond_nums.append(b_mean_cond)

    for i, block in enumerate(model.ema_model.encoder.out):
        b_cond_nums = []
        for name, param in block.named_parameters():
            if "weight" in name:
                original_param_wt = param.clone()
                if (len(original_param_wt.shape)==4):
                    W_matrix = original_param_wt.view(original_param_wt.shape[0], -1)  # Flatten kernels into a 2D matrix
                    U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
                    condition_number = S.max() / S.min()
                    act_cond_num.append(condition_number.item())
                    min_sing_vals.append(S.min().item())
                    max_sing_vals.append(S.max().item())
                    b_cond_nums.append(condition_number.item())
                else:
                    condition_number = 1.0
                    b_cond_nums.append(condition_number)
                    act_cond_num.append(1.0)
                    min_sing_vals.append(1e10)
                    max_sing_vals.append(1e-10)

            else:
                b_cond_nums.append(1.0)
                act_cond_num.append(1.0)
                min_sing_vals.append(1e10)
                max_sing_vals.append(1e-10)



        if isinstance(block, (nn.SiLU, nn.AdaptiveAvgPool2d, nn.Flatten)):
            b_cond_nums = [1.0]
        b_cond_nums = np.array(b_cond_nums)
        b_mean_cond = np.mean(b_cond_nums)
        g_cond_nums.append(b_mean_cond)

    cond_nums_array = np.array(g_cond_nums)

    actual_cond_nums_array = np.array(act_cond_num)
    min_sing_vals_array = np.array(min_sing_vals)
    max_sing_vals_array = np.array(max_sing_vals)


    cond_nums_normalized = (cond_nums_array) / np.sum(cond_nums_array)

    return cond_nums_normalized, cond_nums_array, actual_cond_nums_array, min_sing_vals_array, max_sing_vals_array

cond_nums_normalized, g_cond_nums, actual_cond_nums_array, min_sing_vals_array, max_sing_vals_array = get_layer_pert_recon(model)


filtered_g_cond_nums = [value for value in g_cond_nums if value != 1.0]



# Create a bar chart
plt.figure(figsize=(6, 9))  # Set figure size
plt.barh(range(len(filtered_g_cond_nums)), filtered_g_cond_nums, color='blue', alpha=0.7)

# Labels and title
plt.ylabel("Block index", fontsize=24)
plt.xlabel("$\kappa$", fontsize=24)
#plt.title("Bar Chart of g_cond_nums")


def sci_notation_formatter(y, _):
    if y == 0:
        return "0"  # Display zero as "0" instead of "0e0"
    return f"{int(y):.0e}".replace("+", "").replace("e0", "e")

formatter = ticker.FuncFormatter(sci_notation_formatter)
plt.gca().xaxis.set_major_formatter(formatter)


plt.xticks(fontsize=24)
plt.yticks(range(len(filtered_g_cond_nums)), range(len(filtered_g_cond_nums)), fontsize=24)

# Show the plot
plt.tight_layout()  # Adjust layout to prevent cutoff of labels

plt.savefig("diffae/conditioning_analysis/diff_ae_conditioning_chart_.png")

plt.show()


####################################### actual condition numbers

filtered_g_cond_nums = [value for value in actual_cond_nums_array if value != 1.0]
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
step = 4
yticks = list(range(1, len(filtered_g_cond_nums), step))

plt.yticks(yticks, yticks, fontsize=28)

plt.tight_layout()  # Adjust layout to prevent cutoff of labels
plt.savefig("diffae/conditioning_analysis/diff_ae_conditioning_chart_actual_k.png")
plt.show()
plt.close()
plt.close()

####################################### minimum condition numbers

filtered_g_cond_nums = [value for value in min_sing_vals_array if value != 1e10]
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
step = 4
yticks = list(range(1, len(filtered_g_cond_nums), step))
plt.yticks(yticks, yticks, fontsize=28)
plt.tight_layout()  # Adjust layout to prevent cutoff of labels
plt.savefig("diffae/conditioning_analysis/diff_ae_min_sing_vals.png")
plt.show()
plt.close()

print()

filtered_g_cond_nums = [value for value in max_sing_vals_array if value != 1e-10]
print("max singular values ", filtered_g_cond_nums)
plt.figure(figsize=(4, 6))  # Set figure size
#plt.barh(range(len(filtered_g_cond_nums)), filtered_g_cond_nums, color='blue', alpha=0.7)
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
step = 4
yticks = list(range(1, len(filtered_g_cond_nums), step))
plt.yticks(yticks, yticks, fontsize=28)
plt.tight_layout()  # Adjust layout to prevent cutoff of labels
plt.savefig("diffae/conditioning_analysis/diff_ae_max_sing_vals.png")
plt.show()
plt.close()


# ----------------------------------------- -----------------------------------------  ----------------------------------------- ----------------------------------------- -----------------------------------------  ----------------------------------------- ----------------------------------------- -----------------------------------------  -----------------------------------------
#vals = np.array(min_sing_vals_array, dtype=float)

filtered_g_cond_nums = [value for value in min_sing_vals_array if value != 1e10]
vals = np.array(filtered_g_cond_nums, dtype=float)
fig, ax = plt.subplots(figsize=(4, 6))  # wider helps a lot
ax.barh(np.arange(len(vals)), vals, color="blue", alpha=0.7)

linthresh = 1e-2
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

step = 4
yticks = list(range(1, len(vals), step))
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=28)

fig.tight_layout()
fig.savefig("diffae/conditioning_analysis/DiffAE_min_sing_valsCC.png", dpi=300)
plt.show()
plt.close()


#-----------------#

#vals = np.array(max_sing_vals_array, dtype=float)
#placeholder removed
filtered_g_cond_nums = [value for value in max_sing_vals_array if value != 1e-10]

vals = np.array(filtered_g_cond_nums, dtype=float)
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

step = 4
yticks = list(range(1, len(vals), step))
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=28)

fig.tight_layout()
fig.savefig("diffae/conditioning_analysis/DiffAE_max_sing_valsCC.png", dpi=300)
plt.show()
plt.close()




#-----------------#


#vals = np.array(allCondNums, dtype=float)

'''# Keep only finite values for axis scaling / ticks
finite = np.isfinite(vals)
vals_finite = vals[finite]

# If you want to *plot* inf/nan as a capped value instead of dropping them:
cap = np.nanmax(vals_finite) * 1.05
vals_plot = vals.copy()
vals_plot[~finite] = cap  # put inf/nan at the right edge'''

# placeholder removed
filtered_g_cond_nums = [value for value in actual_cond_nums_array if value != 1.0]
filtered_g_cond_nums = np.array(filtered_g_cond_nums, dtype=float)
fig, ax = plt.subplots(figsize=(4, 6))
#ax.barh(np.arange(len(allCondNums)), allCondNums, color="blue", alpha=0.7)

ax.barh(range(len(filtered_g_cond_nums)),
    filtered_g_cond_nums,
    color='blue',
    edgecolor='blue',
    linewidth=2)


linthresh = 1e1
ax.set_xscale("symlog", linthresh=linthresh)

# Safe limits (finite only)
xmax = np.max(filtered_g_cond_nums)
ax.set_xlim(0, xmax * 1.05)

# decade ticks
xmin_pos = np.min(filtered_g_cond_nums[filtered_g_cond_nums > 0])
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

step = 4
yticks = list(range(1, len(filtered_g_cond_nums), step))
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=28)

fig.tight_layout()
fig.savefig("diffae/conditioning_analysis/DiffAECondNumCC.png", dpi=300)
plt.show()
plt.close()
