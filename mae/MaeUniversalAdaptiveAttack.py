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





################################################################### ADaptive attacks ################################################################### ADaptive attacks 



export CUDA_VISIBLE_DEVICES=4
cd illcond
conda activate mae5
python mae/MaeUniversalAdaptiveAttack.py --attck_type "oa_l2_kf_mcmc" --desired_norm_l_inf 0.09 --set_mask_ratio 0.75 --learningRate 0.01
python mae/MaeUniversalAdaptiveAttack.py --attck_type "grill_l2_kf_only_decodings_mcmc" --desired_norm_l_inf 0.05 --set_mask_ratio 0.75 --learningRate 0.01


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


def log_posterior(z_latent, x_input, model, AdvIds_restore):
    # Decoder output
    x_mean = model.forward_decoder(z_latent, AdvIds_restore)

    # Flatten images
    x_flat      = x_input.view(x_input.size(0), -1)
    x_mean_flat = x_mean.view(x_mean.size(0), -1)

    # Likelihood term p(x | z)
    log_p_x = -((x_flat - x_mean_flat) ** 2).sum(dim=1) / 2.0  # [N]

    # Prior term p(z) -- standard normal
    z_flat = z_latent.view(z_latent.size(0), -1)
    log_p_z = -0.5 * (z_flat ** 2).sum(dim=1)  # [N]

    return (log_p_x + log_p_z).sum()


def get_hmc_lat1i(z1, normalized_attacked, model, AdvIds_restore):
    z = z1  # [N, L, D] typically
    x = normalized_attacked
    step_size = 0.008
    n_steps = 2
    leapfrog_steps = 2

    for i in range(n_steps):
        p = torch.randn_like(z)  # same shape as z
        z_new = z.clone().detach().requires_grad_(True)
        p_new = p.clone()

        # ----- initial grad -----
        log_post = log_posterior(z_new, x, model, AdvIds_restore)
        grad = torch.autograd.grad(log_post, z_new, create_graph=True)[0]

        # half-step momentum update
        p_new = p_new + 0.5 * step_size * grad

        # ----- leapfrog steps -----
        for _ in range(leapfrog_steps):
            # position update
            z_new = (z_new + step_size * p_new).detach().requires_grad_(True)

            # recompute grad at new z_new
            log_post = log_posterior(z_new, x, model, AdvIds_restore)
            grad = torch.autograd.grad(log_post, z_new, create_graph=True)[0]

            # full-step momentum update
            p_new = p_new + step_size * grad

        # final half-step momentum update
        p_new = p_new + 0.5 * step_size * grad
        p_new = -p_new  # make symmetric

        # ----- Metropolis–Hastings accept/reject -----
        with torch.no_grad():
            # current log posterior
            z_flat = z.view(z.size(0), -1)
            logp_current = -0.5 * (z_flat ** 2).sum(dim=1) - (
                (x.view(x.size(0), -1) - model.forward_decoder(z, AdvIds_restore).view(x.size(0), -1)) ** 2
            ).sum(dim=1) / 2.0  # [N]

            # proposed log posterior
            z_new_flat = z_new.view(z_new.size(0), -1)
            logp_new = -0.5 * (z_new_flat ** 2).sum(dim=1) - (
                (x.view(x.size(0), -1) - model.forward_decoder(z_new, AdvIds_restore).view(x.size(0), -1)) ** 2
            ).sum(dim=1) / 2.0  # [N]

            accept_ratio = torch.exp(logp_new - logp_current).clamp(max=1.0)
            mask = torch.rand_like(accept_ratio) < accept_ratio  # [N]
            #print("mask", mask)

            # broadcast mask to latent shape
            mask_latent = mask.view(-1, *([1] * (z.dim() - 1)))
            #print("mask_latent", mask_latent)
            z = torch.where(mask_latent, z_new, z)

    z_mcmc = z
    return z_mcmc




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
            grad = torch.autograd.grad(log_post, z_new, create_graph=True)[0]

            p_new = p_new + step_size * grad

        p_new = p_new + 0.5 * step_size * grad
        p_new = -p_new

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

    return z



def get_hmc_lat1(z1, normalized_attacked, model, AdvIds_restore):
    #     forward_decoder
    #       

    #NormLatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
    #NormY = model_mae_gan.forward_decoder(NormLatent, NormIds_restore)  # [N, L, p*p*3]

    z = z1#.clone().detach().requires_grad_(True)  # Start point for MCMC
    x = normalized_attacked#.detach()              # Adversarial input
    step_size = 0.008
    n_steps = 20
    leapfrog_steps = 10

    #samples = []
    for i in range(n_steps):
        p = torch.randn_like(z)  # Sample momentum
        z_new = z.clone()
        p_new = p.clone()
        #x_mean = model.decoder(model.fc3(z_new))
        x_mean = model.forward_decoder(z_new, AdvIds_restore)
        x_flat, x_mean_flat = x.view(x.size(0), -1), x_mean.view(x.size(0), -1)
        log_p_x = -((x_flat - x_mean_flat) ** 2).sum(dim=1) / 2  # assuming Gaussian decoder
        log_p_z = -0.5 * (z_new ** 2).sum(dim=1)                # standard normal prior
        log_post = (log_p_x + log_p_z).sum()
        grad = torch.autograd.grad(log_post, z_new)[0]

        # Leapfrog integration
        p_new = p_new + 0.5 * step_size * grad
        for _ in range(leapfrog_steps):
            z_new = z_new + step_size * p_new
            z_new = z_new#.detach().requires_grad_(True)
            #x_mean = model.decoder(model.fc3(z_new))
            #x_mean = model.decoder(model.fc3(z_new))
            x_mean = model.forward_decoder(z_new, AdvIds_restore)
            x_flat, x_mean_flat = x.view(x.size(0), -1), x_mean.view(x.size(0), -1)
            log_p_x = -((x_flat - x_mean_flat) ** 2).sum(dim=1) / 2
            log_p_z = -0.5 * (z_new ** 2).sum(dim=1)
            log_post = (log_p_x + log_p_z).sum()
            grad = torch.autograd.grad(log_post, z_new)[0]
            p_new = p_new + step_size * grad
        p_new = p_new + 0.5 * step_size * grad
        p_new = -p_new  # Make symmetric

        #with torch.no_grad():
        logp_current = -0.5 * (z ** 2).sum(dim=1) - ((x.view(x.size(0), -1) - model.forward_decoder(z, AdvIds_restore).view(x.size(0), -1)) ** 2).sum(dim=1) / 2
        logp_new = -0.5 * (z_new ** 2).sum(dim=1) - ((x.view(x.size(0), -1) - model.forward_decoder(z_new, AdvIds_restore).view(x.size(0), -1)) ** 2).sum(dim=1) / 2
        
        accept_ratio = torch.exp(logp_new - logp_current).clamp(max=1.0)
        #print("accept_ratio", accept_ratio)
        #print("torch.rand_like(accept_ratio) ", torch.rand_like(accept_ratio))
        mask = torch.rand_like(accept_ratio) < accept_ratio
        #print("mask", mask)
        z = torch.where(mask.unsqueeze(1), z_new, z)
        z = z#.detach().requires_grad_(True)  # Prepare for next iteration
        #samples.append(z)

    z_mcmc = z#.detach()  # Final robust latent sample
    #print("z_mcmc.shape", z_mcmc.shape)
    return z_mcmc

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
count = 0


print("\n=== BLOCK PARAMETERS ===")
conditionNumberList = []
for i, blk in enumerate(model_mae_gan.blocks):
    for name, param in blk.named_parameters():
        if(len(param.shape)==2):
            if "mlp" in name:
                print("name", name)
                U, S, Vt = torch.linalg.svd(param, full_matrices=False)
                condition_number = (S.max() / S.min())
                condition_number = condition_number.item()
                conditionNumberList.append(condition_number)

inter = len(conditionNumberList)
interconditionNumberList = conditionNumberList
if hasattr(model_mae_gan, "decoder_blocks"):
    for i, blk in enumerate(model_mae_gan.decoder_blocks):
        for name, param in blk.named_parameters():
            if(len(param.shape)==2):
                if "mlp" in name:
                    print("name", name)
                    U, S, Vt = torch.linalg.svd(param, full_matrices=False)
                    condition_number = (S.max() / S.min())
                    condition_number = condition_number.item()
                    conditionNumberList.append(condition_number)

print("len(conditionNumberList)", len(conditionNumberList))
total =  len(conditionNumberList)
laterconditionNumberList = conditionNumberList[inter:]

print("conditionNumberList", conditionNumberList)

print("total-inter", total-inter)

noise_addition = 0.1 * torch.rand(1, 3, 224, 224).cuda()

criterion = nn.MSELoss()


for batch_idx, (images, labels) in enumerate(dataloader):
    break
source_im = images[0].unsqueeze(0)
mi, ma = source_im.min().item(), source_im.max().item()
print("source_im.shape", source_im.shape)

noise_addition = (torch.randn_like(source_im) * 0.2).cuda()
noise_addition = noise_addition.clone().detach().requires_grad_(True)
print("noise_addition.shape", noise_addition.shape)
optimizer = optim.Adam([noise_addition], lr=0.001)






#attck_type = "grill_l2_kf"
#attck_type = "grill_cos_kf"
#desired_norm_l_inf = 0.07

def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()


def cosForOa(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()


def wasserstein_distance(tensor_a, tensor_b):
    tensor_a_flat = torch.flatten(tensor_a)
    tensor_b_flat = torch.flatten(tensor_b)
    tensor_a_sorted, _ = torch.sort(tensor_a_flat)
    tensor_b_sorted, _ = torch.sort(tensor_b_flat)    
    wasserstein_dist = torch.mean(torch.abs(tensor_a_sorted - tensor_b_sorted))
    return wasserstein_dist



if attck_type == "grill_cos_kf_only_decodings_mcmc":
    all_condition_nums = np.array(conditionNumberList)
    all_condition_nums[all_condition_nums>100.0]=100
    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)
    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = [0.0]
    for step in range(100):
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            #print("images.shape", images.shape)

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(images.float() + noise_addition, mi, ma)

            layerwise_outputs.clear()

            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)

            NormLatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)


            NormY = model_mae_gan.forward_decoder(NormLatent, NormIds_restore)  # [N, L, p*p*3]

            orig_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)

            AdvLatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked.float(), mask_ratio=set_mask_ratio)

            AdvLatent = get_hmc_lat1ij(AdvLatent, normalized_attacked, model_mae_gan, AdvIds_restore)

            AdvY = model_mae_gan.forward_decoder(AdvLatent, AdvIds_restore)  # [N, L, p*p*3]

            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            hookCount = 0
            loss = 0

            for layer, adv_output in list(adv_layerwise_outputs.items())[48:]:
                orig_output = orig_layerwise_outputs[layer]
                #print("orig_output.shape", orig_output.shape)
                loss +=  (1-cos(adv_output, orig_output))**2 #* cond_nums_normalized[hookCount]
                #hookCount+=1

            total_loss =  -1 * loss * (1- cosForOa(NormY, AdvY))**2
            #print("total_loss.item()", total_loss.item())
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        with torch.no_grad():
            print("total_loss", total_loss)
            normalRecon = run_one_batch_firstImagePlot(images, model_mae_gan)
            advRecon = run_one_batch_firstImagePlotAdv(normalized_attacked, model_mae_gan)
            deviation = torch.norm(normalRecon - advRecon, p=2)
            print("deviation.item()", deviation.item())
            #all_adv_div_list.append(deviation.item())
            updateQuest = deviation >= max(all_adv_div_list)
            print("step", step)
            print("attck_type", attck_type)
            if(updateQuest):
                all_adv_div_list.append(deviation.item())
                torch.save(noise_addition, "mae/univ_attack_storage/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")
                print("adv_div_list", all_adv_div_list)
                np.save("mae/deviation_store/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", all_adv_div_list)




if attck_type == "grill_l2_kf_only_decodings_mcmc":
    all_condition_nums = np.array(conditionNumberList)
    all_condition_nums[all_condition_nums>100.0]=100
    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)
    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = [0.0]
    for step in range(100):
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            #print("images.shape", images.shape)

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(images.float() + noise_addition, mi, ma)

            layerwise_outputs.clear()

            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)

            NormLatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)


            NormY = model_mae_gan.forward_decoder(NormLatent, NormIds_restore)  # [N, L, p*p*3]

            orig_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)

            AdvLatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked.float(), mask_ratio=set_mask_ratio)

            AdvLatent = get_hmc_lat1ij(AdvLatent, normalized_attacked, model_mae_gan, AdvIds_restore)

            AdvY = model_mae_gan.forward_decoder(AdvLatent, AdvIds_restore)  # [N, L, p*p*3]

            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            hookCount = 0
            loss = 0

            for layer, adv_output in list(adv_layerwise_outputs.items())[48:]:
                orig_output = orig_layerwise_outputs[layer]
                #print("orig_output.shape", orig_output.shape)
                loss +=  criterion(adv_output, orig_output) #* cond_nums_normalized[hookCount]
                #hookCount+=1

            total_loss =  -1 * loss * criterion(adv_output, orig_output)
            #print("total_loss.item()", total_loss.item())
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        with torch.no_grad():
            print("total_loss", total_loss)
            normalRecon = run_one_batch_firstImagePlot(images, model_mae_gan)
            advRecon = run_one_batch_firstImagePlotAdv(normalized_attacked, model_mae_gan)
            deviation = torch.norm(normalRecon - advRecon, p=2)
            print("deviation.item()", deviation.item())
            #all_adv_div_list.append(deviation.item())
            updateQuest = deviation >= max(all_adv_div_list)
            print("step", step)
            print("attck_type", attck_type)
            if(updateQuest):
                all_adv_div_list.append(deviation.item())
                torch.save(noise_addition, "mae/univ_attack_storage/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")
                print("adv_div_list", all_adv_div_list)
                np.save("mae/deviation_store/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", all_adv_div_list)





if attck_type == "oa_l2_kf_mcmc":
    all_condition_nums = np.array(conditionNumberList)
    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    all_condition_nums[all_condition_nums>100.0]=100
    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)
    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = [0.0]
    for step in range(100):
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            #print("images.shape", images.shape)

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(images.float() + noise_addition, mi, ma)

            layerwise_outputs.clear()

            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)

            NormLatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            NormY = model_mae_gan.forward_decoder(NormLatent, NormIds_restore)  # [N, L, p*p*3]

            orig_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)

            AdvLatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked.float(), mask_ratio=set_mask_ratio)
            AdvLatent = get_hmc_lat1ij(AdvLatent, normalized_attacked, model_mae_gan, AdvIds_restore)

            AdvY = model_mae_gan.forward_decoder(AdvLatent, AdvIds_restore)  # [N, L, p*p*3]

            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()


            total_loss =  -1 * criterion(NormY, AdvY)
            #print("total_loss.item()", total_loss.item())
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        with torch.no_grad():
            print("total_loss", total_loss)
            normalRecon = run_one_batch_firstImagePlot(images, model_mae_gan)
            advRecon = run_one_batch_firstImagePlotAdv(normalized_attacked, model_mae_gan)
            deviation = torch.norm(normalRecon - advRecon, p=2)
            print("deviation.item()", deviation.item())
            #all_adv_div_list.append(deviation.item())
            updateQuest = deviation >= max(all_adv_div_list)
            print("step", step)
            print("attck_type", attck_type)
            if(updateQuest):
                all_adv_div_list.append(deviation.item())
                torch.save(noise_addition, "mae/univ_attack_storage/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")
                print("adv_div_list", all_adv_div_list)
                np.save("mae/deviation_store/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", all_adv_div_list)



