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




--------->> Also seems to be doing great

export CUDA_VISIBLE_DEVICES=3
cd illcond
conda activate mae5
python mae/maeAttackQualitativeImagePlotting.py --attck_type "grill_cos_kf_only_decodings" --desired_norm_l_inf 0.09 --set_mask_ratio 0.75 --learningRate 0.01
python mae/maeAttackQualitativeImagePlotting.py --attck_type "grill_cos_kf_only_decodings_mcmc" --desired_norm_l_inf 0.09 --set_mask_ratio 0.75 --learningRate 0.01

python mae/maeAttackQualitativeImagePlotting.py --attck_type "grill_l2_kf_only_decodings" --desired_norm_l_inf 0.09 --set_mask_ratio 0.75 --learningRate 0.01
python maeAttackQualitativeImagePlotting.py --attck_type "grill_l2_kf_only_decodings_mcmc" --desired_norm_l_inf 0.09 --set_mask_ratio 0.75 --learningRate 0.01

python mae/maeAttackQualitativeImagePlotting.py --attck_type "oa_l2_kf" --desired_norm_l_inf 0.09 --set_mask_ratio 0.75 --learningRate 0.01
python mae/maeAttackQualitativeImagePlotting.py --attck_type "oa_l2_kf_mcmc" --desired_norm_l_inf 0.09 --set_mask_ratio 0.75 --learningRate 0.01


'''

import argparse

parser = argparse.ArgumentParser(description='Adversarial attack on VAE')
parser.add_argument('--attck_type', type=str, default="lip", help='Segment index')
parser.add_argument('--desired_norm_l_inf', type=float, default="lip", help='Segment index')
parser.add_argument('--set_mask_ratio', type=float, default="mask ratio", help='Segment index')
parser.add_argument('--learningRate', type=float, default="0.01", help='Segment index')


args = parser.parse_args()

attck_type = args.attck_type
desired_norm_l_inf = args.desired_norm_l_inf
set_mask_ratio = args.set_mask_ratio
learningRate = args.learningRate


defend = False



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
    batch_size=10,
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
    #step_size = 0.008 # default
    step_size = 0.00008
    n_steps = 50
    leapfrog_steps = 100

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


def run_one_batch_firstImagePlot(img, model):
    # img is already a torch tensor from DataLoader: [N, 3, 224, 224]
    x = img.to(device, non_blocking=True)

    with torch.no_grad():  # no need for gradients when just visualizing
        loss, y, mask = model(x.float(), mask_ratio=set_mask_ratio)

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
    plt.savefig('mae/demoImage/sampleImageOutputs_attack_type'+str(attck_type)+'_norm_bound_'+str(desired_norm_l_inf)+'_set_mask_ratio_'+str(set_mask_ratio)+'learningRate_'+str(learningRate)+'.png')
    plt.close()

    return y


def getReconstructions(img, model, name):
    # img is already a torch tensor from DataLoader: [N, 3, 224, 224]
    x = img.to(device, non_blocking=True)

    #with torch.no_grad():  # no need for gradients when just visualizing
        #loss, y, mask = model(x.float(), mask_ratio=set_mask_ratio)

    Lat, mask, Ids_restore = model_mae_gan.forward_encoder(x.float(), mask_ratio=set_mask_ratio)
    if name=="attacked" and defend:
        print("checkedHere ?")
        Lat = get_hmc_lat1ij(Lat, x, model_mae_gan, Ids_restore)
        #kkk =2
    y = model_mae_gan.forward_decoder(Lat, Ids_restore)  # [N, L, p*p*3]


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

    if name=="attacked":
        for whichImageINTheBatch in range(len(y)):
            plt.rcParams['figure.figsize'] = [24, 24]
            plt.subplot(1, 4, 1)
            show_image(x_cpu[whichImageINTheBatch], "original")
            plt.subplot(1, 4, 2)
            show_image(im_masked[whichImageINTheBatch], "masked")
            plt.subplot(1, 4, 3)
            show_image(y[whichImageINTheBatch], "reconstruction")
            plt.subplot(1, 4, 4)
            show_image(im_paste[whichImageINTheBatch], "reconstruction + visible")
            plt.savefig('mae/qualitativeTest/'+attck_type+'/l_inf_'+str(desired_norm_l_inf)+'/input_'+name+'_sampleNum_'+str(whichImageINTheBatch)+'_attack_type'+str(attck_type)+'_norm_bound_'+str(desired_norm_l_inf)+'_set_mask_ratio_'+str(set_mask_ratio)+'learningRate_'+str(learningRate)+'.png')
            plt.close()
    else:
        for whichImageINTheBatch in range(len(y)):
            plt.rcParams['figure.figsize'] = [24, 24]
            plt.subplot(1, 4, 1)
            show_image(x_cpu[whichImageINTheBatch], "original")
            plt.subplot(1, 4, 2)
            show_image(im_masked[whichImageINTheBatch], "masked")
            plt.subplot(1, 4, 3)
            show_image(y[whichImageINTheBatch], "reconstruction")
            plt.subplot(1, 4, 4)
            show_image(im_paste[whichImageINTheBatch], "reconstruction + visible")
            plt.savefig('mae/qualitativeTest/unattacked/input_'+name+'_sampleNum_'+str(whichImageINTheBatch)+'_set_mask_ratio_'+str(set_mask_ratio)+'learningRate_'+str(learningRate)+'.png')
            plt.close()

    

    return y



def run_one_batch_firstImagePlotAdv(img, model):
    # img is already a torch tensor from DataLoader: [N, 3, 224, 224]
    x = img.to(device, non_blocking=True)

    with torch.no_grad():  # no need for gradients when just visualizing
        loss, y, mask = model(x.float(), mask_ratio=set_mask_ratio)

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
    plt.savefig('mae/demoImage/AdvsampleImageOutputs_attack_type'+str(attck_type)+'_norm_bound_'+str(desired_norm_l_inf)+'_set_mask_ratio_'+str(set_mask_ratio)+'learningRate_'+str(learningRate)+'.png')
    plt.close()

    return y

for batch_idx, (images, labels) in enumerate(dataloader):
    images = images.to(device, non_blocking=True)
    break


# MAE with extra GAN loss
chkpt_dir = 'mae/mae_visualize_vit_large_ganloss.pth'
model_mae_gan = prepare_model('mae/mae_visualize_vit_large_ganloss.pth', 'mae_vit_large_patch16')

intermediate_acts = {}


layerwise_outputs = {}
def make_store_hook(store_dict, name):
    def hook(module, input, output):
        store_dict[name] = output #.detach()  # keep on GPU; add .cpu() if you want CPU
    return hook


print('Model loaded.')
print('MAE with extra GAN loss:')
#run_one_batch_firstImagePlot(images, model_mae_gan)




#########################################################################################

# ENCODER BLOCK HOOKS
for i, blk in enumerate(model_mae_gan.blocks):
    #blk.attn.qkv.register_forward_hook(make_store_hook(layerwise_outputs, f"enc_block_{i}_attn_qkv"))
    #blk.attn.proj.register_forward_hook(make_store_hook(layerwise_outputs, f"enc_block_{i}_attn_proj"))
    blk.mlp.fc1.register_forward_hook(make_store_hook(layerwise_outputs, f"enc_block_{i}_mlp_fc1"))
    blk.mlp.fc2.register_forward_hook(make_store_hook(layerwise_outputs, f"enc_block_{i}_mlp_fc2"))

# DECODER BLOCK HOOKS
for i, blk in enumerate(model_mae_gan.decoder_blocks):
    #blk.attn.qkv.register_forward_hook(make_store_hook(layerwise_outputs, f"dec_block_{i}_attn_qkv"))
    #blk.attn.proj.register_forward_hook(make_store_hook(layerwise_outputs, f"dec_block_{i}_attn_proj"))
    blk.mlp.fc1.register_forward_hook(make_store_hook(layerwise_outputs, f"dec_block_{i}_mlp_fc1"))
    blk.mlp.fc2.register_forward_hook(make_store_hook(layerwise_outputs, f"dec_block_{i}_mlp_fc2"))

#########################################################################################

criterion = nn.MSELoss()

'''for batch_idx, (images, labels) in enumerate(dataloader):
    break
source_im = images[0].unsqueeze(0)
mi, ma = source_im.min().item(), source_im.max().item()
print("source_im.shape", source_im.shape)'''

noise_addition = torch.load("mae/univ_attack_storage/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")

#with torch.no_grad():

for batch_idx, (images, labels) in enumerate(dataloader):
    images = images[6:8].to(device, non_blocking=True)
    mi, ma = images.min().item(), images.max().item()

    normalized_attacked = torch.clamp(images.float() + noise_addition, mi, ma)

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    with torch.no_grad():

        normalRecon = getReconstructions(images, model_mae_gan, "normal")
        print("normalRecon.shape", normalRecon.shape)

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    attackedRecon = getReconstructions(normalized_attacked, model_mae_gan, "attacked")
    print("attackedRecon.shape", attackedRecon.shape)

    l2Loss = torch.norm(normalRecon-attackedRecon, 2)

    print("l2Loss", l2Loss)

    break
