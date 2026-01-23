import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from datasets import load_dataset

import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib.ticker import FuncFormatter


import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

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


export CUDA_VISIBLE_DEVICES=7
cd illcond
conda activate mae5
python mae/maeAttackQuantitativeMeanStdVarPlotsRunningTestMCMC.py --set_mask_ratio 0.75 --learningRate 0.01


'''

import argparse

parser = argparse.ArgumentParser(description='Adversarial attack on VAE')
#parser.add_argument('--attck_type', type=str, default="lip", help='Segment index')
#parser.add_argument('--desired_norm_l_inf', type=float, default="lip", help='Segment index')
parser.add_argument('--set_mask_ratio', type=float, default="mask ratio", help='Segment index')
parser.add_argument('--learningRate', type=float, default="0.01", help='Segment index')


args = parser.parse_args()

#attck_type = args.attck_type
#desired_norm_l_inf = args.desired_norm_l_inf
set_mask_ratio = args.set_mask_ratio
learningRate = args.learningRate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#print("attck_type", attck_type)

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
    batch_size=100,
    shuffle=False,
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



def getReconstructions(img, model, name):

    x = img.to(device, non_blocking=True)
    with torch.no_grad():  # no need for gradients when just visualizing
        loss, y, mask = model(x.float(), mask_ratio=set_mask_ratio)

    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    return y




for batch_idx, (images, labels) in enumerate(dataloader):
    images = images.to(device, non_blocking=True)
    break


# MAE with extra GAN loss
chkpt_dir = 'mae/mae_visualize_vit_large_ganloss.pth'
model_mae_gan = prepare_model('mae/mae_visualize_vit_large_ganloss.pth', 'mae_vit_large_patch16')


print('Model loaded.')
print('MAE with extra GAN loss:')

criterion = nn.MSELoss()

for batch_idx, (images, labels) in enumerate(dataloader):
    break
source_im = images[0].unsqueeze(0)
mi, ma = source_im.min().item(), source_im.max().item()


#attck_types = [ "la_cos_kf", "la_l2_kf", "la_wass_kf", "oa_cos_kf", "oa_l2_kf", "oa_wass_kf", "grill_cos_kf_only_decodingsMasked", "grill_cos_kf_only_decodings", "grill_l2_kf_only_decodingsMasked", "grill_l2_kf_only_decodingsMasked", "grill_wass_kf_only_decodings" ]

#allAttacktypes = ["la_l2_kf", "la_wass_kf", "la_cos_kf", "oa_l2_kf", "oa_wass_kf", "oa_cos_kf", "lgr_l2_kf", "lgr_wass_kf", "lgr_cos_kf", "grill_l2_kf_only_decodings", "grill_wass_kf_only_decodings", "grill_cos_kf_only_decodings"]


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


def log_posterior(z_latent, x_input, model, AdvIds_restore):
    # Decoder output
    x_mean = model.forward_decoder(z_latent, AdvIds_restore)
    del model
    # Flatten images
    x_flat      = x_input.view(x_input.size(0), -1)
    x_mean_flat = x_mean.view(x_mean.size(0), -1)
    del x_mean, x_input

    # Likelihood term p(x | z)
    log_p_x = -((x_flat - x_mean_flat) ** 2).sum(dim=1) / 2.0  # [N]
    del x_flat, x_mean_flat
    # Prior term p(z) -- standard normal
    z_flat = z_latent.view(z_latent.size(0), -1)
    del z_latent
    log_p_z = -0.5 * (z_flat ** 2).sum(dim=1)  # [N]
    del z_flat

    return (log_p_x + log_p_z).sum()


def get_hmc_lat1ij(z1, normalized_attacked, model, AdvIds_restore):
    z = z1  # keep connection to encoder / noise_addition
    x = normalized_attacked
    step_size = 0.008
    n_steps = 2
    leapfrog_steps = 2

    for i in range(n_steps):
        p = torch.randn_like(z)
        z_new = z.clone().requires_grad_(True)  # no detach

        p_new = p.clone()

        log_post = log_posterior(z_new, x, model, AdvIds_restore)
        grad = torch.autograd.grad(log_post, z_new, create_graph=True)[0]

        p_new = p_new + 0.5 * step_size * grad

        for _ in range(leapfrog_steps):
            z_new = (z_new + step_size * p_new)
            z_new.requires_grad_(True)

            log_post = log_posterior(z_new, x, model, AdvIds_restore)
            grad = torch.autograd.grad(log_post, z_new, create_graph=False)[0]

            p_new = p_new + step_size * grad

        p_new = (p_new + 0.5 * step_size * grad).detach()
        p_new = -p_new
        del grad

        # IMPORTANT: no torch.no_grad() here if you want gradients
        z_flat = z.view(z.size(0), -1)
        logp_current = -0.5 * (z_flat ** 2).sum(dim=1) - (
            (x.view(x.size(0), -1) - model.forward_decoder(z, AdvIds_restore).view(x.size(0), -1)) ** 2
        ).sum(dim=1) / 2.0

        z_new_flat = z_new.view(z_new.size(0), -1)
        logp_new = -0.5 * (z_new_flat ** 2).sum(dim=1) - (
            (x.view(x.size(0), -1) - model.forward_decoder(z_new, AdvIds_restore).view(x.size(0), -1)) ** 2
        ).sum(dim=1) / 2.0

        accept_ratio = torch.exp(logp_new - logp_current).clamp(max=1.0)
        mask = (torch.rand_like(accept_ratio) < accept_ratio).float()
        mask = mask.view(-1, *([1] * (z.dim() - 1)))
        z = mask * z_new + (1 - mask) * z
        del mask, z_new

    return z


allAttacktypes = ["oa_l2_kf_mcmc", "grill_l2_kf_only_decodings_mcmc"]

allDesired_norm_l_inf = [0.05, 0.07, 0.09]

#allDesired_norm_l_inf = [0.08, 0.09]


all_method_means = []
all_method_stds = []

for attck_type in allAttacktypes:

    mean_per_per_accum = []
    std_per_per_accum = []

    for desired_norm_l_inf in allDesired_norm_l_inf:
        
        print("attck_type", attck_type)
        print("desired_norm_l_inf", desired_norm_l_inf)
        noise_addition = torch.load("mae/univ_attack_storage/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")



        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
        #mi, ma = images.min().item(), images.max().item()

        normalized_attacked = torch.clamp(images.float() + noise_addition, mi, ma)

        with torch.no_grad():
            NormLatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            NormY = model_mae_gan.forward_decoder(NormLatent, NormIds_restore)  # [N, L, p*p*3]
            NormY = model_mae_gan.unpatchify(NormY)
            normalRecon = torch.einsum('nchw->nhwc', NormY).detach().cpu()
        #print("NormY.shape", NormY.shape)
        #torch.manual_seed(1234)
        #torch.cuda.manual_seed_all(1234)
        #normalRecon = getReconstructions(images, model_mae_gan, "normal")
        #print("normalRecon.shape", normalRecon.shape)

        #torch.manual_seed(1234)
        #torch.cuda.manual_seed_all(1234)

        AdvLatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked.float(), mask_ratio=set_mask_ratio)
        #AdvLatent = get_hmc_lat1ij(AdvLatent, normalized_attacked, model_mae_gan, AdvIds_restore)
        AdvLatent = get_hmc_lat1ij(AdvLatent, normalized_attacked, model_mae_gan, AdvIds_restore)

        with torch.no_grad():
            AdvY = model_mae_gan.forward_decoder(AdvLatent, AdvIds_restore)  # [N, L, p*p*3]
            AdvY = model_mae_gan.unpatchify(AdvY)
            attackedRecon = torch.einsum('nchw->nhwc', AdvY).detach().cpu()
            #print("AdvY.shape", AdvY.shape)

            #attackedRecon = getReconstructions(normalized_attacked, model_mae_gan, "attacked")
            #print("attackedRecon.shape", attackedRecon.shape)

            l2Loss = torch.norm(normalRecon-attackedRecon, 2)

            l2_per_image = torch.norm(normalRecon - attackedRecon, p=2, dim=(1, 2, 3))

            L2LossMean = torch.mean(l2_per_image)
            L2LossesStd = torch.std(l2_per_image)
            #print("totalBatchL2", l2Loss)
            print("L2LossMean", L2LossMean)
            print("L2LossesStd", L2LossesStd)
            #print(l2_per_image.shape)
            print()
            #print("l2_per_image", l2_per_image)

            mean_per_per_accum.append(L2LossMean)
            std_per_per_accum.append(L2LossesStd)
        
    mean_per_per_accum = np.array(mean_per_per_accum)
    std_per_per_accum = np.array(std_per_per_accum)

    all_method_means.append(mean_per_per_accum)
    all_method_stds.append(std_per_per_accum)




import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Simulated data: epsilon values
#epsilon = np.linspace(0.1, 1.0, 4)

epsilon = allDesired_norm_l_inf



#objective_names = ["LA,l-2", "LA, wasserst.", "LA, cosine", "OA,l-2", "OA, wasserst.", "OA, cosine","LGR, l-2", "LGR, wasserst.", "LGR, cosine", "GRILL, l-2", "GRILL, wasserst.", "GRILL, cosine"]

objective_names = ["OA,l-2", "GRILL, l-2"]


#objective_names = ["LA,l-2", "LA, wasserst.", "LA, SKL", "LA, cosine", "OA, l-2", "OA, wasserst.", "OA, SKL", "OA, cosine", "LMA, l-2", "LMA, wasserst.", "LMA, SKL", "LMA, cosine", "GRILL, l-2", "GRILL, wasserst.", "GRILL, SKL", "GRILL, cosine"]


# Simulated distributions (mean and standard deviation)
#mean_values = np.sin(2 * np.pi * epsilon)  # Some function to represent the mean
#std_dev = 0.2 + 0.1 * np.cos(2 * np.pi * epsilon)  # Changing spread

'''color_list = [
    'blue', 'orange', 'green', 'red', 
    'purple', 'cyan', 'magenta', 'yellow',
    'brown', 'pink', 'gray', 'olive', 
    'lime', 'teal', 'indigo', 'gold'
]'''

# lgr l2 brown

'''color_list = [
    'blue', 'orange', 'red', 
    'purple', 'cyan', 'yellow', 
    'brown', 'pink', 'olive', 
    'lime', 'teal', 'gold'
]'''


color_list = [
    'purple', 
    'lime'
]

#color_list = ['blue', 'orange', 'green', 'red', 'lime', 'teal', 'indigo', 'gold']

plt.figure(figsize=(6, 7))


for i in range(len(all_method_means)):
#for i in [12, 13, 14, 15]:#, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15]:
    mean_values = all_method_means[i]
    std_dev = all_method_stds[i]


    # Compute upper and lower bounds for the shaded region
    upper_bound = mean_values + std_dev
    lower_bound = mean_values - std_dev

    # Plot the mean curve
    plt.plot(epsilon, mean_values, label=objective_names[i], color=color_list[i])

    # Plot the shaded region (± std deviation)
    plt.fill_between(epsilon, lower_bound, upper_bound, color=color_list[i], alpha=0.2)

# Labels and legend


plt.xlabel(r'$c$', fontsize=28)
plt.ylabel('L-2 distance', fontsize=28)


formatter = FuncFormatter(lambda x, _: f'{x:.2f}')
#plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)


plt.xticks(rotation=45, fontsize=28)
plt.yticks(fontsize=28)
#plt.title("Distribution Change with Epsilon")
plt.grid(True)
#plt.legend()
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=24)
# Adjust layout to fit the legend
handles, labels = plt.gca().get_legend_handles_labels()

# Increase line thickness in the legend
for handle in handles:
    handle.set_linewidth(4)
plt.tight_layout()


plt.show()

plt.savefig("mae/damage_distributions_variation/MAE_MCMC_RunningTestLeg.png")

