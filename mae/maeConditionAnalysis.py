import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from datasets import load_dataset

import random



import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # for CPU/cuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for dataloader workers
    #torch.use_deterministic_algorithms(True)

set_seed(1)


'''

export CUDA_VISIBLE_DEVICES=4
cd illcond
conda activate mae5
python mae/maeConditionAnalysis.py

'''


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlatImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.paths = sorted(
            glob.glob(os.path.join(root_dir, "**", "*.*"), recursive=True)
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # if you have labels somewhere, return them too; for now, dummy -1
        return img, -1





sys.path.append('..')
import models_mae


data_dir = "mae/imagenetDataSubset"


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



dataset = FlatImageDataset(data_dir, transform=transform)
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)  #**********

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True if device.type == "cuda" else False,  # good for GPU
)


print(f"Total images: {len(dataset)}")

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    #print(msg)
    # move model to GPU
    model = model.to(device)
    model.eval()  # inference mode is usually what you want here

    return model



def run_one_batch_firstImagePlot(img, model):
    # img is already a torch tensor from DataLoader: [N, 3, 224, 224]
    x = img.to(device, non_blocking=True)

    with torch.no_grad():  # no need for gradients when just visualizing
        loss, y, mask = model(x.float(), mask_ratio=0.75)

    # y is [N, L, p^2 * 3] -> unpatchify, then move to CPU for plotting
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    # x is still on GPU; for plotting we need it on CPU and in NHWC:
    x_cpu = torch.einsum('nchw->nhwc', x.detach().cpu())

    im_masked = x_cpu * (1 - mask)
    im_paste = x_cpu * (1 - mask) + y * mask

    whichImageINTheBatch = 0

    plt.rcParams['figure.figsize'] = [24, 24]
    plt.subplot(1, 4, 1)
    show_image(x_cpu[whichImageINTheBatch], "original")
    plt.subplot(1, 4, 2)
    show_image(im_masked[whichImageINTheBatch], "masked")
    plt.subplot(1, 4, 3)
    show_image(y[whichImageINTheBatch], "reconstruction")
    plt.subplot(1, 4, 4)
    show_image(im_paste[whichImageINTheBatch], "reconstruction + visible")
    plt.savefig('mae/sampleImageOutputs.png')
    plt.close()


for batch_idx, (images, labels) in enumerate(dataloader):
    images = images.to(device, non_blocking=True)
    break


# MAE with extra GAN loss
chkpt_dir = 'mae/mae_visualize_vit_large_ganloss.pth'
#chkpt_dir = 'mae_visualize_vit_large.pth'
model_mae_gan = prepare_model('mae/mae_visualize_vit_large_ganloss.pth', 'mae_vit_large_patch16')




print('Model loaded.')
print('MAE with extra GAN loss:')
#run_one_batch_firstImagePlot(images, model_mae_gan)



count = 0


print("\n=== ENCODER BLOCK PARAMETERS ===")
blockCount =0
TwoDblockCount = 0
conditionNumberList = []
minSingularValueList = []
maxSingularValueList = []
for i, blk in enumerate(model_mae_gan.blocks):
    print(f"\n----- BLOCK {i} -----")
    for name, param in blk.named_parameters():
        #count += 1
        # prefix with blocks.{i}. to match global naming
        print(f"blocks.{i}.{name:40} {list(param.shape)}")
        #print("count", count)
        if(len(param.shape)==2 and 'weight' in name and 'attn' not in name and 'mlp' in name):
            TwoDblockCount+=1
            U, S, Vt = torch.linalg.svd(param, full_matrices=False)
            condition_number = (S.max() / S.min())
            condition_number = condition_number.item()
            conditionNumberList.append(condition_number)
            minSingularValueList.append(S.min().item())
            maxSingularValueList.append(S.max().item())
            print("condition_number", condition_number)
    blockCount+=1
    print("blockCount", blockCount)


# 4) (Optional) Decoder blocks, if your MAE has them
if hasattr(model_mae_gan, "decoder_blocks"):
    print("\n=== DECODER BLOCK PARAMETERS ===")
    decoderBlockCount = 0
    for i, blk in enumerate(model_mae_gan.decoder_blocks):
        print(f"\n----- DECODER BLOCK {i} -----")
        for name, param in blk.named_parameters():
            #count += 1
            print(f"decoder_blocks.{i}.{name:30} {list(param.shape)}")
            #print("count", count)
            #if(len(param.shape)==2 and 'weight' in name and 'attn' not in name and 'mlp' in name):
            if(len(param.shape)==2 and 'weight' in name and 'mlp' in name):
                TwoDblockCount+=1

                U, S, Vt = torch.linalg.svd(param, full_matrices=False)
                condition_number = (S.max() / S.min())
                condition_number = condition_number.item()
                conditionNumberList.append(condition_number)
                minSingularValueList.append(S.min().item())
                maxSingularValueList.append(S.max().item())
                print("condition_number", condition_number)

        decoderBlockCount+=1
        print("decoderBlockCount", decoderBlockCount)

print("len(conditionNumberList)", len(conditionNumberList))


print(conditionNumberList[1:])


# Why ignore block zero : Following He et al. (2022), the MAE encoder first embeds image patches via a linear projection with positional embeddings, after which the token sequence is processed by a stack of Transformer blocks. Consequently, the first Transformer block operates directly on embedding-level representations and is often treated as part of the embedding/stem in representation analyses.
# Ref : https://arxiv.org/abs/2111.06377 : Masked Autoencoders Are Scalable Vision Learners

import matplotlib.ticker as ticker

####################################### actual condition numbers

filtered_g_cond_nums = conditionNumberList[1:] # [value for value in actual_cond_nums_array if value != 1.0]
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
step = 10
yticks = list(range(1, len(filtered_g_cond_nums), step))

plt.yticks(yticks, yticks, fontsize=28)

plt.tight_layout()  # Adjust layout to prevent cutoff of labels
plt.savefig("mae/conditionAnalysis/mae_conditioning_chart_actual_k.png")
plt.show()
plt.close()

####################################### minimum condition numbers

filtered_g_cond_nums = minSingularValueList[1:] #[value for value in min_sing_vals_array if value != 1e10]
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
step = 10
yticks = list(range(1, len(filtered_g_cond_nums), step))
plt.yticks(yticks, yticks, fontsize=28)
plt.tight_layout()  # Adjust layout to prevent cutoff of labels
plt.savefig("mae/conditionAnalysis/mae_min_sing_vals.png")
plt.show()
plt.close()

print()

filtered_g_cond_nums = maxSingularValueList[1:] #[value for value in max_sing_vals_array if value != 1e-10]
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
step = 10
yticks = list(range(1, len(filtered_g_cond_nums), step))
plt.yticks(yticks, yticks, fontsize=28)
plt.tight_layout()  # Adjust layout to prevent cutoff of labels
plt.savefig("mae/conditionAnalysis/mae_max_sing_vals.png")
plt.show()
plt.close()



vals = np.array(minSingularValueList[1:], dtype=float)

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

step = 10
yticks = list(range(1, len(vals), step))
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=28)

fig.tight_layout()
fig.savefig("mae/conditionAnalysis/MAE_min_sing_valsCC.png", dpi=300)
plt.show()
plt.close()


#-----------------#

vals = np.array(maxSingularValueList[1:], dtype=float)

fig, ax = plt.subplots(figsize=(4, 6))  # wider helps a lot
ax.barh(np.arange(len(vals)), vals, color="blue", alpha=0.7)

linthresh = 1e0
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

step = 10
yticks = list(range(1, len(vals), step))
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=28)

fig.tight_layout()
fig.savefig("mae/conditionAnalysis/MAE_max_sing_valsCC.png", dpi=300)
plt.show()
plt.close()




#-----------------#


vals = np.array(conditionNumberList[1:], dtype=float)

'''# Keep only finite values for axis scaling / ticks
finite = np.isfinite(vals)
vals_finite = vals[finite]

# If you want to *plot* inf/nan as a capped value instead of dropping them:
cap = np.nanmax(vals_finite) * 1.05
vals_plot = vals.copy()
vals_plot[~finite] = cap  # put inf/nan at the right edge'''

fig, ax = plt.subplots(figsize=(4, 6))
#ax.barh(np.arange(len(allCondNums)), allCondNums, color="blue", alpha=0.7)

ax.barh(range(len(vals)),
    vals,
    color='blue',
    edgecolor='blue',
    linewidth=2)


linthresh = 1e1
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

step = 10
yticks = list(range(1, len(vals), step))
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=28)

fig.tight_layout()
fig.savefig("mae/conditionAnalysis/MAECondNumCC.png", dpi=300)
plt.show()
plt.close()
