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


export CUDA_VISIBLE_DEVICES=2
cd illcond
conda activate mae5
python mae/MaeSampleSpecificAttacks.py --attck_type "la_cos_kf_SS" --desired_norm_l_inf 0.9 --set_mask_ratio 0.75 --learningRate 0.01
python mae/MaeSampleSpecificAttacks.py --attck_type "la_l2_kf_SS" --desired_norm_l_inf 0.06 --set_mask_ratio 0.75 --learningRate 0.01

python mae/MaeSampleSpecificAttacks.py --attck_type "lgr_l2_kf" --desired_norm_l_inf 0.05 --set_mask_ratio 0.75 --learningRate 0.01
python mae/MaeSampleSpecificAttacks.py --attck_type "lgr_cos_kf" --desired_norm_l_inf 0.05 --set_mask_ratio 0.75 --learningRate 0.01
python mae/MaeSampleSpecificAttacks.py --attck_type "lgr_wass_kf" --desired_norm_l_inf 0.05 --set_mask_ratio 0.75 --learningRate 0.01


python mae/MaeSampleSpecificAttacks.py --attck_type "oa_cos_kf_SS" --desired_norm_l_inf 0.01 --set_mask_ratio 0.75 --learningRate 0.01
python mae/MaeSampleSpecificAttacks.py --attck_type "oa_wass_kf_SS" --desired_norm_l_inf 0.01 --set_mask_ratio 0.75 --learningRate 0.01
python mae/MaeSampleSpecificAttacks.py --attck_type "oa_l2_kf_SS" --desired_norm_l_inf 0.01 --set_mask_ratio 0.75 --learningRate 0.01

python mae/MaeSampleSpecificAttacks.py --attck_type "grill_cos_kf_only_decodings_SS" --desired_norm_l_inf 0.06 --set_mask_ratio 0.75 --learningRate 0.01
python mae/MaeSampleSpecificAttacks.py --attck_type "grill_wass_kf_only_decodings_SS" --desired_norm_l_inf 0.06 --set_mask_ratio 0.75 --learningRate 0.01
python mae/MaeSampleSpecificAttacks.py --attck_type "grill_l2_kf_only_decodings_SS" --desired_norm_l_inf 0.06 --set_mask_ratio 0.75 --learningRate 0.01

'''

import argparse

parser = argparse.ArgumentParser(description='Adversarial attack on VAE')
parser.add_argument('--attck_type', type=str, default="lip", help='Segment index')
parser.add_argument('--desired_norm_l_inf', type=float, default="lip", help='Segment index')
parser.add_argument('--set_mask_ratio', type=float, default="mask ratio", help='Segment index')
parser.add_argument('--learningRate', type=float, default="0.01", help='Segment index')

# Actual allAttacktypes = ["la_l2_kf_SS", "la_wass_kf_SS", "la_cos_kf_SS", "oa_l2_kf_SS", "oa_wass_kf_SS", "oa_cos_kf_SS", "lgr_l2_kf", "lgr_wass_kf", "lgr_cos_kf", "grill_l2_kf_only_decodings_SS", "grill_wass_kf_only_decodings_SS", "grill_cos_kf_only_decodings_SS"]


# Actual allAttacktypes = ["la_l2_kf_SS", "la_wass_kf_SS", "la_cos_kf_SS", "oa_l2_kf_SS", "oa_wass_kf_SS", "oa_cos_kf_SS", "lgr_l2_kf", "lgr_wass_kf", "lgr_cos_kf", "grill_l2_kf_only_decodings_SS", "grill_wass_kf_only_decodings_SS", "grill_cos_kf_only_decodings_SS"]

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





sys.path.append('mae')
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
#source_im = images[0].unsqueeze(0)
mi, ma = images.min().item(), images.max().item()
print("source_im.shape", images.shape)

noise_addition = (torch.randn_like(images) * 0.2).cuda()
noise_addition = noise_addition.clone().detach().requires_grad_(True)
print("noise_addition.shape", noise_addition.shape)
optimizer = optim.Adam([noise_addition], lr=0.001)





def run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, scaled_noise, adv_gen):
    with torch.no_grad():
        print(f"Step {step}, Loss: {total_loss.item()}, distortion L-2: {l2_distortion}, distortion L-inf: {l_inf_distortion}, deviation: {deviation}, recon mse: {mase_dev}")
        print()
        print("attack type", attck_type)    
        adv_div_list.append(deviation.item())
        adv_mse_list.append(mase_dev.item())
        fig, ax = plt.subplots(1, 3, figsize=(10, 10))
        ax[0].imshow(normalized_attacked[0].permute(1, 2, 0).cpu().numpy())
        ax[0].set_title('Attacked Image')
        ax[0].axis('off')

        ax[1].imshow(scaled_noise[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        ax[1].set_title('Noise')
        ax[1].axis('off')

        ax[2].imshow(adv_gen[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        ax[2].set_title('Attack reconstruction')
        ax[2].axis('off')
        plt.show()
        plt.savefig("mae/optimization_time_plots/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_learningRate_"+str(learningRate)+".png")

    optimized_noise = scaled_noise
    torch.save(optimized_noise, "mae/univ_attack_storage/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_learningRate"+str(learningRate)+".pt")
    print("adv_div_list", adv_div_list)
    np.save("mae/deviation_store/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", adv_div_list)
    plt.figure(figsize=(8, 5))
    plt.plot(adv_div_list, marker='o')
    plt.xlabel('Step')
    plt.ylabel('Deviation')
    plt.title(f'Deviation over Steps: {attck_type}, L∞={desired_norm_l_inf}')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("mae/optimization_time_plots/div_MAE_attack_type"+str(attck_type)+"_step_"+str(step)+"_norm_bound_"+str(desired_norm_l_inf)+"_learningRate"+str(learningRate)+".png")
    plt.show()


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

if attck_type == "grill_l2_kf":
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
            NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)
            orig_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()


            hookCount = 0
            loss = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                orig_output = orig_layerwise_outputs[layer]
                loss += criterion(adv_output, orig_output) * cond_nums_normalized[hookCount]
                hookCount+=1
                #print("loss", loss)
                #print()
            #latent, mask, ids_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)

            Normlatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            Advlatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)

            total_loss = -1 * loss * criterion(Normlatent, Advlatent)

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



if attck_type == "grill_cos_kf":
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
            #NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            Normlatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            orig_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            #AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            Advlatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            hookCount = 0
            loss = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                orig_output = orig_layerwise_outputs[layer]
                #orig_output=nn.LayerNorm(orig_output)
                #adv_output=nn.LayerNorm(adv_output)
                #loss += criterion(adv_output, orig_output) * conditionNumberList[hookCount]
                loss +=  (1-cos(adv_output, orig_output))**2 * cond_nums_normalized[hookCount]

                hookCount+=1
                #print("hookCount", hookCount)
                #print()
            #Normlatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            #Advlatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)

            total_loss =  -1 * loss * (1- cos(Normlatent, Advlatent))**2

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




if attck_type == "grill_cos_kf_full":
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
            NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)
            #Normlatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            orig_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            #AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)
            #Advlatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            hookCount = 0
            loss = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                orig_output = orig_layerwise_outputs[layer]
                #orig_output=nn.LayerNorm(orig_output)
                #adv_output=nn.LayerNorm(adv_output)
                #loss += criterion(adv_output, orig_output) * conditionNumberList[hookCount]
                loss +=  (1-cos(adv_output, orig_output))**2 * cond_nums_normalized[hookCount]

                hookCount+=1
                #print("hookCount", hookCount)
                #print()
            #Normlatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            #Advlatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)

            total_loss =  -1 * loss * (1- cosForOa(NormY, AdvY))**2

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







if attck_type == "la_cos_kf_SS":
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
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        break
    for step in range(30000):
        #for batch_idx, (images, labels) in enumerate(dataloader):
            #images = images.to(device, non_blocking=True)
            #print("images.shape", images.shape)

        optimizer.zero_grad()

        normalized_attacked = torch.clamp(images.float() + noise_addition, mi, ma)

        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        NormLatent, _, _ = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)

        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        AdvLatent, _, _ = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)

        total_loss = -1 * (cos(NormLatent, AdvLatent)-1)**2

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        if step%100==0:
            with torch.no_grad():
                print("total_loss", total_loss)
                normalRecon = run_one_batch_firstImagePlot(images, model_mae_gan)
                advRecon = run_one_batch_firstImagePlotAdv(normalized_attacked, model_mae_gan)
                deviation = torch.norm(normalRecon - advRecon, p=2)
                print("deviation.item()", deviation.item())
                #all_adv_div_list.append(deviation.item())
                #updateQuest = deviation >= max(all_adv_div_list)
                print("step", step)
                print("attck_type", attck_type)
                #if(updateQuest):
                all_adv_div_list.append(deviation.item())
                torch.save(noise_addition, "mae/univ_attack_storage/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")
                print("adv_div_list", all_adv_div_list)
                np.save("mae/deviation_store/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", all_adv_div_list)




if attck_type == "la_l2_kf_SS":
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

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        break

    for step in range(30000):
        #for batch_idx, (images, labels) in enumerate(dataloader):
            #images = images.to(device, non_blocking=True)
            #print("images.shape", images.shape)

        optimizer.zero_grad()

        normalized_attacked = torch.clamp(images.float() + noise_addition, mi, ma)

        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        NormLatent, _, _ = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)

        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        AdvLatent, _, _ = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)

        total_loss = -1 * criterion(NormLatent, AdvLatent)

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        if step%100==0:
            with torch.no_grad():
                print("total_loss", total_loss)
                normalRecon = run_one_batch_firstImagePlot(images, model_mae_gan)
                advRecon = run_one_batch_firstImagePlotAdv(normalized_attacked, model_mae_gan)
                deviation = torch.norm(normalRecon - advRecon, p=2)
                print("deviation.item()", deviation.item())
                #all_adv_div_list.append(deviation.item())
                #updateQuest = deviation >= max(all_adv_div_list)
                print("step", step)
                print("attck_type", attck_type)
                #if(updateQuest):
                all_adv_div_list.append(deviation.item())
                torch.save(noise_addition, "mae/univ_attack_storage/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")
                print("adv_div_list", all_adv_div_list)
                np.save("mae/deviation_store/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", all_adv_div_list)




if attck_type == "lgr_l2_kf":
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

            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            NormLatent, _, _ = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)


            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            AdvLatent, _, _ = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)
            AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)

            total_loss = -1 * criterion(NormLatent, AdvLatent) * criterion(NormY, AdvY)

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



if attck_type == "lgr_wass_kf":
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

            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            NormLatent, _, _ = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)


            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            AdvLatent, _, _ = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)
            AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)

            total_loss = -1 * wasserstein_distance(NormLatent, AdvLatent) * wasserstein_distance(NormY, AdvY)

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




if attck_type == "lgr_cos_kf":
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

            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            NormLatent, _, _ = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)


            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            AdvLatent, _, _ = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)
            AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)

            total_loss = -1 * (cos(NormLatent, AdvLatent)-1)**2 * (cosForOa(NormY, AdvY)-1)**2

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




if attck_type == "la_wass_kf_SS":
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
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        break
    for step in range(30000):
        #for batch_idx, (images, labels) in enumerate(dataloader):
            #images = images.to(device, non_blocking=True)
            #print("images.shape", images.shape)

        optimizer.zero_grad()

        normalized_attacked = torch.clamp(images.float() + noise_addition, mi, ma)

        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        NormLatent, _, _ = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)

        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        AdvLatent, _, _ = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)

        total_loss = -1 * wasserstein_distance(NormLatent, AdvLatent)

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        if step %100 == 0:
            with torch.no_grad():
                print("total_loss", total_loss)
                normalRecon = run_one_batch_firstImagePlot(images, model_mae_gan)
                advRecon = run_one_batch_firstImagePlotAdv(normalized_attacked, model_mae_gan)
                deviation = torch.norm(normalRecon - advRecon, p=2)
                print("deviation.item()", deviation.item())
                #all_adv_div_list.append(deviation.item())
                #updateQuest = deviation >= max(all_adv_div_list)
                print("step", step)
                print("attck_type", attck_type)
                #if(updateQuest):
                all_adv_div_list.append(deviation.item())
                torch.save(noise_addition, "mae/univ_attack_storage/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")
                print("adv_div_list", all_adv_div_list)
                np.save("mae/deviation_store/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", all_adv_div_list)




if attck_type == "oa_cos_kf_SS":
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
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        break
    for step in range(30000):
        #for batch_idx, (images, labels) in enumerate(dataloader):
            #images = images.to(device, non_blocking=True)
        #print("images.shape", images.shape)

        optimizer.zero_grad()

        normalized_attacked = torch.clamp(images.float() + noise_addition, mi, ma)


        #NormLatent, _, _ = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)

        #AdvLatent, _, _ = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)

        #torch.manual_seed(1234)
        #torch.cuda.manual_seed_all(1234)
        NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)
        #NormY = model_mae_gan.unpatchify(NormY)
        #NormY = torch.einsum('nchw->nhwc', NormY).detach().cpu()

        #torch.manual_seed(1234)
        #torch.cuda.manual_seed_all(1234)
        AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)
        #AdvY = model_mae_gan.unpatchify(AdvY)
        #AdvY = torch.einsum('nchw->nhwc', AdvY).detach().cpu()

        #print("NormY.shape", NormY.shape)
        #print("AdvY.shape", AdvY.shape)

        total_loss = -1 * (cosForOa(NormY, AdvY)-1)**2

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)
        if step%100==0:
            with torch.no_grad():
                print("total_loss", total_loss)
                normalRecon = run_one_batch_firstImagePlot(images, model_mae_gan)
                advRecon = run_one_batch_firstImagePlotAdv(normalized_attacked, model_mae_gan)
                deviation = torch.norm(normalRecon - advRecon, p=2)
                print("deviation.item()", deviation.item())
                #all_adv_div_list.append(deviation.item())
                #updateQuest = deviation >= max(all_adv_div_list)
                print("step", step)
                print("attck_type", attck_type)
                #if(updateQuest):
                all_adv_div_list.append(deviation.item())
                torch.save(noise_addition, "mae/univ_attack_storage/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")
                print("adv_div_list", all_adv_div_list)
                np.save("mae/deviation_store/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", all_adv_div_list)


if attck_type == "oa_l2_kf_SS":
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

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        break
    for step in range(30000):
        #for batch_idx, (images, labels) in enumerate(dataloader):
            #images = images.to(device, non_blocking=True)
        #print("images.shape", images.shape)

        optimizer.zero_grad()

        normalized_attacked = torch.clamp(images.float() + noise_addition, mi, ma)


        #NormLatent, _, _ = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)

        #AdvLatent, _, _ = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)

        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)
        #NormY = model_mae_gan.unpatchify(NormY)
        #NormY = torch.einsum('nchw->nhwc', NormY).detach().cpu()

        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)
        #AdvY = model_mae_gan.unpatchify(AdvY)
        #AdvY = torch.einsum('nchw->nhwc', AdvY).detach().cpu()

        #print("NormY.shape", NormY.shape)
        #print("AdvY.shape", AdvY.shape)

        total_loss = -1 * criterion(NormY, AdvY)

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)
        if step%100==0:
            with torch.no_grad():
                print("total_loss", total_loss)
                normalRecon = run_one_batch_firstImagePlot(images, model_mae_gan)
                advRecon = run_one_batch_firstImagePlotAdv(normalized_attacked, model_mae_gan)
                deviation = torch.norm(normalRecon - advRecon, p=2)
                print("deviation.item()", deviation.item())
                #all_adv_div_list.append(deviation.item())
                #updateQuest = deviation >= max(all_adv_div_list)
                print("step", step)
                print("attck_type", attck_type)
                #if(updateQuest):
                all_adv_div_list.append(deviation.item())
                torch.save(noise_addition, "mae/univ_attack_storage/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")
                print("adv_div_list", all_adv_div_list)
                np.save("mae/deviation_store/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", all_adv_div_list)



if attck_type == "oa_wass_kf_SS":
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
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        break
    for step in range(30000):
        #for batch_idx, (images, labels) in enumerate(dataloader):
        #images = images.to(device, non_blocking=True)
        #print("images.shape", images.shape)

        optimizer.zero_grad()

        normalized_attacked = torch.clamp(images.float() + noise_addition, mi, ma)


        #NormLatent, _, _ = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)

        #AdvLatent, _, _ = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)

        #torch.manual_seed(1234)
        #torch.cuda.manual_seed_all(1234)
        NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)
        #NormY = model_mae_gan.unpatchify(NormY)
        #NormY = torch.einsum('nchw->nhwc', NormY).detach().cpu()

        #torch.manual_seed(1234)
        #torch.cuda.manual_seed_all(1234)
        AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)
        #AdvY = model_mae_gan.unpatchify(AdvY)
        #AdvY = torch.einsum('nchw->nhwc', AdvY).detach().cpu()

        #print("NormY.shape", NormY.shape)
        #print("AdvY.shape", AdvY.shape)

        total_loss = -1 * wasserstein_distance(NormY, AdvY)

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        if step%100==0:
            with torch.no_grad():
                print("total_loss", total_loss)
                normalRecon = run_one_batch_firstImagePlot(images, model_mae_gan)
                advRecon = run_one_batch_firstImagePlotAdv(normalized_attacked, model_mae_gan)
                deviation = torch.norm(normalRecon - advRecon, p=2)
                print("deviation.item()", deviation.item())
                #all_adv_div_list.append(deviation.item())
                #updateQuest = deviation >= max(all_adv_div_list)
                print("step", step)
                print("attck_type", attck_type)
                #if(updateQuest):
                all_adv_div_list.append(deviation.item())
                torch.save(noise_addition, "mae/univ_attack_storage/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")
                print("adv_div_list", all_adv_div_list)
                np.save("mae/deviation_store/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", all_adv_div_list)




if attck_type == "grill_cos_kf_only_decodings_SS":
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
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        break
    for step in range(30000):

        optimizer.zero_grad()

        normalized_attacked = torch.clamp(images.float() + noise_addition, mi, ma)

        layerwise_outputs.clear()

        #torch.manual_seed(1234)
        #torch.cuda.manual_seed_all(1234)
        NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)

        orig_layerwise_outputs = layerwise_outputs.copy()
        layerwise_outputs.clear()

        #torch.manual_seed(1234)
        #torch.cuda.manual_seed_all(1234)
        AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)

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

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)
        if step%100==0:
            with torch.no_grad():
                print("total_loss", total_loss)
                normalRecon = run_one_batch_firstImagePlot(images, model_mae_gan)
                advRecon = run_one_batch_firstImagePlotAdv(normalized_attacked, model_mae_gan)
                deviation = torch.norm(normalRecon - advRecon, p=2)
                print("deviation.item()", deviation.item())
                #all_adv_div_list.append(deviation.item())
                #updateQuest = deviation >= max(all_adv_div_list)
                print("step", step)
                print("attck_type", attck_type)
                #if(updateQuest):
                all_adv_div_list.append(deviation.item())
                torch.save(noise_addition, "mae/univ_attack_storage/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")
                print("adv_div_list", all_adv_div_list)
                np.save("mae/deviation_store/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", all_adv_div_list)



if attck_type == "grill_l2_kf_only_decodings_SS":
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
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        break
    for step in range(30000):
        optimizer.zero_grad()
        normalized_attacked = torch.clamp(images.float() + noise_addition, mi, ma)
        layerwise_outputs.clear()
        #torch.manual_seed(1234)
        #torch.cuda.manual_seed_all(1234)
        NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)

        orig_layerwise_outputs = layerwise_outputs.copy()
        layerwise_outputs.clear()

        #torch.manual_seed(1234)
        #torch.cuda.manual_seed_all(1234)
        AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)

        adv_layerwise_outputs = layerwise_outputs.copy()
        layerwise_outputs.clear()

        hookCount = 0
        loss = 0

        for layer, adv_output in list(adv_layerwise_outputs.items())[48:]:
            orig_output = orig_layerwise_outputs[layer]
            loss +=  criterion(adv_output, orig_output)  #* cond_nums_normalized[hookCount]

        total_loss =  -1 * loss * criterion(NormY, AdvY)

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)
        if step%100==0:
            with torch.no_grad():
                print("total_loss", total_loss)
                normalRecon = run_one_batch_firstImagePlot(images, model_mae_gan)
                advRecon = run_one_batch_firstImagePlotAdv(normalized_attacked, model_mae_gan)
                deviation = torch.norm(normalRecon - advRecon, p=2)
                print("deviation.item()", deviation.item())
                #all_adv_div_list.append(deviation.item())
                #updateQuest = deviation >= max(all_adv_div_list)
                print("step", step)
                print("attck_type", attck_type)
                #if(updateQuest):
                all_adv_div_list.append(deviation.item())
                torch.save(noise_addition, "mae/univ_attack_storage/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")
                print("adv_div_list", all_adv_div_list)
                np.save("mae/deviation_store/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", all_adv_div_list)





if attck_type == "grill_l2_kf_only_decodings_mcmc":
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
            NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)
            #print("NormY.shape", NormY.shape)
            #print("NormMask.shape", NormMask.shape)
            #print("NormMask", NormMask)
            #visible_mask = (NormMask == 0) 
            #invisible_mask = (NormMask == 1) 
            #visibleNormY = NormY[visible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            #invisibleNormY = NormY[invisible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            #print("visibleNormY.shape", visibleNormY.shape)
            #print("invisibleNormY.shape", invisibleNormY.shape)

            #print()
            #Normlatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            orig_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)
            #Advlatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            #print("NormMask==AdvMask", NormMask==AdvMask)


            hookCount = 0
            loss = 0
            #for layer, adv_output in list(adv_layerwise_outputs.items()):
                #orig_output = orig_layerwise_outputs[layer]
                #print("orig_output.shape", orig_output.shape)
                #loss +=  (1-cos(adv_output, orig_output))**2 #* cond_nums_normalized[hookCount]
                #hookCount+=1
            #print("it should change here ")
            #print()
            for layer, adv_output in list(adv_layerwise_outputs.items())[48:]:
                orig_output = orig_layerwise_outputs[layer]
                #print("orig_output.shape", orig_output.shape)
                loss +=  criterion(adv_output, orig_output)  #* cond_nums_normalized[hookCount]
                #hookCount+=1

            total_loss =  -1 * loss * criterion(NormY, AdvY)

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






if attck_type == "grill_wass_kf_only_decodings_SS":
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
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        break
    for step in range(30000):

        optimizer.zero_grad()

        normalized_attacked = torch.clamp(images.float() + noise_addition, mi, ma)

        layerwise_outputs.clear()

        #torch.manual_seed(1234)
        #torch.cuda.manual_seed_all(1234)
        NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)

        orig_layerwise_outputs = layerwise_outputs.copy()
        layerwise_outputs.clear()

        #torch.manual_seed(1234)
        #torch.cuda.manual_seed_all(1234)
        AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)
        #Advlatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)
        adv_layerwise_outputs = layerwise_outputs.copy()
        layerwise_outputs.clear()

        #print("NormMask==AdvMask", NormMask==AdvMask)


        hookCount = 0
        loss = 0
        #for layer, adv_output in list(adv_layerwise_outputs.items()):
            #orig_output = orig_layerwise_outputs[layer]
            #print("orig_output.shape", orig_output.shape)
            #loss +=  (1-cos(adv_output, orig_output))**2 #* cond_nums_normalized[hookCount]
            #hookCount+=1
        #print("it should change here ")
        #print()
        for layer, adv_output in list(adv_layerwise_outputs.items())[48:]:
            orig_output = orig_layerwise_outputs[layer]
            #print("orig_output.shape", orig_output.shape)
            loss +=  wasserstein_distance(adv_output, orig_output) #* cond_nums_normalized[hookCount]
            #hookCount+=1

        total_loss =  -1 * loss * wasserstein_distance(NormY, AdvY)

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        if step%100==0:
            with torch.no_grad():
                print("total_loss", total_loss)
                normalRecon = run_one_batch_firstImagePlot(images, model_mae_gan)
                advRecon = run_one_batch_firstImagePlotAdv(normalized_attacked, model_mae_gan)
                deviation = torch.norm(normalRecon - advRecon, p=2)
                print("deviation.item()", deviation.item())
                #all_adv_div_list.append(deviation.item())
                #updateQuest = deviation >= max(all_adv_div_list)
                print("step", step)
                print("attck_type", attck_type)
                #if(updateQuest):
                all_adv_div_list.append(deviation.item())
                torch.save(noise_addition, "mae/univ_attack_storage/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")
                print("adv_div_list", all_adv_div_list)
                np.save("mae/deviation_store/MAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", all_adv_div_list)



if attck_type == "grill_cos_kf_only_decodingsMasked":
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
            NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)
            #print("NormY.shape", NormY.shape)
            #print("NormMask.shape", NormMask.shape)
            #print("NormMask", NormMask)
            visible_mask = (NormMask == 0) 
            invisible_mask = (NormMask == 1) 
            visibleNormY = NormY[visible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            invisibleNormY = NormY[invisible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            #print("visibleNormY.shape", visibleNormY.shape)
            #print("invisibleNormY.shape", invisibleNormY.shape)

            #print()
            #Normlatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            orig_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)
            #Advlatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)
            visibleAdvY = AdvY[visible_mask].reshape(AdvY.shape[0], -1, AdvY.shape[-1])
            invisibleAdvY = AdvY[invisible_mask].reshape(AdvY.shape[0], -1, AdvY.shape[-1])
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            #print("NormMask==AdvMask", NormMask==AdvMask)


            hookCount = 0
            loss = 0
            #for layer, adv_output in list(adv_layerwise_outputs.items()):
                #orig_output = orig_layerwise_outputs[layer]
                #print("orig_output.shape", orig_output.shape)
                #loss +=  (1-cos(adv_output, orig_output))**2 #* cond_nums_normalized[hookCount]
                #hookCount+=1
            #print("it should change here ")
            #print()
            for layer, adv_output in list(adv_layerwise_outputs.items())[48:]:
                orig_output = orig_layerwise_outputs[layer][:,1:,:]
                adv_output = adv_output[:,1:,:]
                invisible_orig_output = orig_output[invisible_mask].reshape(orig_output.shape[0], -1, orig_output.shape[-1])
                invisible_adv_output = adv_output[invisible_mask].reshape(adv_output.shape[0], -1, adv_output.shape[-1])

                #print("orig_output.shape", orig_output.shape)
                #print("invisible_orig_output.shape", invisible_orig_output.shape)
                #print("adv_output.shape", adv_output.shape)
                #print("invisible_adv_output.shape", invisible_adv_output.shape)

                loss +=  (1-cosForOa(invisible_orig_output, invisible_adv_output))**2 #* cond_nums_normalized[hookCount]
                #hookCount+=1

            total_loss =  -1 * loss * (1- cosForOa(invisibleNormY, invisibleAdvY))**2

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



if attck_type == "grill_l2_kf_only_decodingsMasked":
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
            NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)
            #print("NormY.shape", NormY.shape)
            #print("NormMask.shape", NormMask.shape)
            #print("NormMask", NormMask)
            visible_mask = (NormMask == 0) 
            invisible_mask = (NormMask == 1) 
            visibleNormY = NormY[visible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            invisibleNormY = NormY[invisible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            #print("visibleNormY.shape", visibleNormY.shape)
            #print("invisibleNormY.shape", invisibleNormY.shape)

            #print()
            #Normlatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            orig_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)
            #Advlatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)
            visibleAdvY = AdvY[visible_mask].reshape(AdvY.shape[0], -1, AdvY.shape[-1])
            invisibleAdvY = AdvY[invisible_mask].reshape(AdvY.shape[0], -1, AdvY.shape[-1])
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            #print("NormMask==AdvMask", NormMask==AdvMask)


            hookCount = 0
            loss = 0
            #for layer, adv_output in list(adv_layerwise_outputs.items()):
                #orig_output = orig_layerwise_outputs[layer]
                #print("orig_output.shape", orig_output.shape)
                #loss +=  (1-cos(adv_output, orig_output))**2 #* cond_nums_normalized[hookCount]
                #hookCount+=1
            #print("it should change here ")
            #print()
            for layer, adv_output in list(adv_layerwise_outputs.items())[48:]:
                orig_output = orig_layerwise_outputs[layer][:,1:,:]
                adv_output = adv_output[:,1:,:]
                invisible_orig_output = orig_output[invisible_mask].reshape(orig_output.shape[0], -1, orig_output.shape[-1])
                invisible_adv_output = adv_output[invisible_mask].reshape(adv_output.shape[0], -1, adv_output.shape[-1])

                #print("orig_output.shape", orig_output.shape)
                #print("invisible_orig_output.shape", invisible_orig_output.shape)
                #print("adv_output.shape", adv_output.shape)
                #print("invisible_adv_output.shape", invisible_adv_output.shape)

                loss +=  criterion(invisible_orig_output, invisible_adv_output) #* cond_nums_normalized[hookCount]
                #hookCount+=1

            total_loss =  -1 * loss * criterion(invisibleNormY, invisibleAdvY)

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



if attck_type == "grill_wass_kf_only_decodingsMasked":
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
            NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)
            #print("NormY.shape", NormY.shape)
            #print("NormMask.shape", NormMask.shape)
            #print("NormMask", NormMask)
            visible_mask = (NormMask == 0) 
            invisible_mask = (NormMask == 1) 
            visibleNormY = NormY[visible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            invisibleNormY = NormY[invisible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            #print("visibleNormY.shape", visibleNormY.shape)
            #print("invisibleNormY.shape", invisibleNormY.shape)

            #print()
            #Normlatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            orig_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)
            #Advlatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)
            visibleAdvY = AdvY[visible_mask].reshape(AdvY.shape[0], -1, AdvY.shape[-1])
            invisibleAdvY = AdvY[invisible_mask].reshape(AdvY.shape[0], -1, AdvY.shape[-1])
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            #print("NormMask==AdvMask", NormMask==AdvMask)


            hookCount = 0
            loss = 0
            #for layer, adv_output in list(adv_layerwise_outputs.items()):
                #orig_output = orig_layerwise_outputs[layer]
                #print("orig_output.shape", orig_output.shape)
                #loss +=  (1-cos(adv_output, orig_output))**2 #* cond_nums_normalized[hookCount]
                #hookCount+=1
            #print("it should change here ")
            #print()
            for layer, adv_output in list(adv_layerwise_outputs.items())[48:]:
                orig_output = orig_layerwise_outputs[layer][:,1:,:]
                adv_output = adv_output[:,1:,:]
                invisible_orig_output = orig_output[invisible_mask].reshape(orig_output.shape[0], -1, orig_output.shape[-1])
                invisible_adv_output = adv_output[invisible_mask].reshape(adv_output.shape[0], -1, adv_output.shape[-1])

                #print("orig_output.shape", orig_output.shape)
                #print("invisible_orig_output.shape", invisible_orig_output.shape)
                #print("adv_output.shape", adv_output.shape)
                #print("invisible_adv_output.shape", invisible_adv_output.shape)

                loss +=  wasserstein_distance(invisible_orig_output, invisible_adv_output) #* cond_nums_normalized[hookCount]
                #hookCount+=1

            total_loss =  -1 * loss * wasserstein_distance(invisibleNormY, invisibleAdvY)

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





if attck_type == "grill_cos_kf_only_decodingsMaskedWeights":
    all_condition_nums = np.array(laterconditionNumberList)
    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    #all_condition_nums[all_condition_nums>100.0]=100
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
            NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)
            #print("NormY.shape", NormY.shape)
            #print("NormMask.shape", NormMask.shape)
            #print("NormMask", NormMask)
            visible_mask = (NormMask == 0) 
            invisible_mask = (NormMask == 1) 
            visibleNormY = NormY[visible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            invisibleNormY = NormY[invisible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            #print("visibleNormY.shape", visibleNormY.shape)
            #print("invisibleNormY.shape", invisibleNormY.shape)

            #print()
            #Normlatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            orig_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)
            #Advlatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)
            visibleAdvY = AdvY[visible_mask].reshape(AdvY.shape[0], -1, AdvY.shape[-1])
            invisibleAdvY = AdvY[invisible_mask].reshape(AdvY.shape[0], -1, AdvY.shape[-1])
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            #print("NormMask==AdvMask", NormMask==AdvMask)


            hookCount = 0
            loss = 0
            '''for layer, adv_output in list(adv_layerwise_outputs.items())[:48]:
                #orig_output = orig_layerwise_outputs[layer]
                #print("orig_output.shape", orig_output.shape)
                #loss +=  (1-cos(adv_output, orig_output))**2 #* cond_nums_normalized[hookCount]
                hookCount+=1'''
            #print("it should change here ")
            #print()
            for layer, adv_output in list(adv_layerwise_outputs.items())[48:]:
                orig_output = orig_layerwise_outputs[layer][:,1:,:]
                adv_output = adv_output[:,1:,:]
                invisible_orig_output = orig_output[invisible_mask].reshape(orig_output.shape[0], -1, orig_output.shape[-1])
                invisible_adv_output = adv_output[invisible_mask].reshape(adv_output.shape[0], -1, adv_output.shape[-1])

                #print("orig_output.shape", orig_output.shape)
                #print("invisible_orig_output.shape", invisible_orig_output.shape)
                #print("adv_output.shape", adv_output.shape)
                #print("invisible_adv_output.shape", invisible_adv_output.shape)
                #print("hookCount", hookCount)
                loss +=  (1-cosForOa(invisible_orig_output, invisible_adv_output))**2 * cond_nums_normalized[hookCount]
                hookCount+=1

            total_loss =  -1 * loss * (1- cosForOa(invisibleNormY, invisibleAdvY))**2

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



if attck_type == "grill_cos_kf_only_decodingsMaskedSpotWeights":
    #all_condition_nums = np.array(conditionNumberList)
    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    #all_condition_nums[all_condition_nums>100.0]=100
    #cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_cmpli = np.random.rand(total-inter) 
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
            NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)
            #print("NormY.shape", NormY.shape)
            #print("NormMask.shape", NormMask.shape)
            #print("NormMask", NormMask)
            visible_mask = (NormMask == 0) 
            invisible_mask = (NormMask == 1) 
            visibleNormY = NormY[visible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            invisibleNormY = NormY[invisible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            #print("visibleNormY.shape", visibleNormY.shape)
            #print("invisibleNormY.shape", invisibleNormY.shape)

            #print()
            #Normlatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            orig_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)
            #Advlatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)
            visibleAdvY = AdvY[visible_mask].reshape(AdvY.shape[0], -1, AdvY.shape[-1])
            invisibleAdvY = AdvY[invisible_mask].reshape(AdvY.shape[0], -1, AdvY.shape[-1])
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            #print("NormMask==AdvMask", NormMask==AdvMask)


            hookCount = 0
            loss = 0
            '''for layer, adv_output in list(adv_layerwise_outputs.items())[:48]:
                #orig_output = orig_layerwise_outputs[layer]
                #print("orig_output.shape", orig_output.shape)
                #loss +=  (1-cos(adv_output, orig_output))**2 #* cond_nums_normalized[hookCount]
                hookCount+=1'''
            #print("it should change here ")
            #print()
            lossBasedWeights = []
            for layer, adv_output in list(adv_layerwise_outputs.items())[48:]:
                orig_output = orig_layerwise_outputs[layer][:,1:,:]
                adv_output = adv_output[:,1:,:]
                invisible_orig_output = orig_output[invisible_mask].reshape(orig_output.shape[0], -1, orig_output.shape[-1])
                invisible_adv_output = adv_output[invisible_mask].reshape(adv_output.shape[0], -1, adv_output.shape[-1])

                theLoss = (1-cosForOa(invisible_orig_output, invisible_adv_output))**2 
                lossBasedWeights.append(theLoss.item())
                loss +=  theLoss * cond_nums_normalized[hookCount]
                hookCount+=1
            lossBasedWeights = (np.array(lossBasedWeights))**1
            cond_nums_normalized = lossBasedWeights / lossBasedWeights.sum()
            #print("lossBasedWeights.sum()", lossBasedWeights.sum())
            #print("cond_nums_normalized", cond_nums_normalized)
            total_loss =  -1 * loss * (1- cosForOa(invisibleNormY, invisibleAdvY))**2

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




if attck_type == "grill_cos_kf_only_decodingsMaskedSpot2Weights":
    #all_condition_nums = np.array(conditionNumberList)
    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    #all_condition_nums[all_condition_nums>100.0]=100
    #cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_cmpli = np.random.rand(total-inter) 
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
            NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)
            #print("NormY.shape", NormY.shape)
            #print("NormMask.shape", NormMask.shape)
            #print("NormMask", NormMask)
            visible_mask = (NormMask == 0) 
            invisible_mask = (NormMask == 1) 
            visibleNormY = NormY[visible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            invisibleNormY = NormY[invisible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            #print("visibleNormY.shape", visibleNormY.shape)
            #print("invisibleNormY.shape", invisibleNormY.shape)

            #print()
            #Normlatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            orig_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)
            #Advlatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)
            visibleAdvY = AdvY[visible_mask].reshape(AdvY.shape[0], -1, AdvY.shape[-1])
            invisibleAdvY = AdvY[invisible_mask].reshape(AdvY.shape[0], -1, AdvY.shape[-1])
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            #print("NormMask==AdvMask", NormMask==AdvMask)


            hookCount = 0
            loss = 0
            '''for layer, adv_output in list(adv_layerwise_outputs.items())[:48]:
                #orig_output = orig_layerwise_outputs[layer]
                #print("orig_output.shape", orig_output.shape)
                #loss +=  (1-cos(adv_output, orig_output))**2 #* cond_nums_normalized[hookCount]
                hookCount+=1'''
            #print("it should change here ")
            #print()
            lossBasedWeights = []
            for layer, adv_output in list(adv_layerwise_outputs.items())[48:]:
                orig_output = orig_layerwise_outputs[layer][:,1:,:]
                adv_output = adv_output[:,1:,:]
                invisible_orig_output = orig_output[invisible_mask].reshape(orig_output.shape[0], -1, orig_output.shape[-1])
                invisible_adv_output = adv_output[invisible_mask].reshape(adv_output.shape[0], -1, adv_output.shape[-1])

                theLoss = (1-cosForOa(invisible_orig_output, invisible_adv_output))**2 
                lossBasedWeights.append(theLoss.item())
                loss +=  theLoss * cond_nums_normalized[hookCount]
                hookCount+=1
            lossBasedWeights = (np.array(lossBasedWeights))**2
            cond_nums_normalized = lossBasedWeights / lossBasedWeights.sum()
            #print("lossBasedWeights.sum()", lossBasedWeights.sum())
            #print("cond_nums_normalized", cond_nums_normalized)
            total_loss =  -1 * loss * (1- cosForOa(invisibleNormY, invisibleAdvY))**2

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





if attck_type == "grill_cos_kf_only_decodingsMaskedSpot4Weights":
    #all_condition_nums = np.array(conditionNumberList)
    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    #all_condition_nums[all_condition_nums>100.0]=100
    #cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_cmpli = np.random.rand(total-inter) 
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
            NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)
            #print("NormY.shape", NormY.shape)
            #print("NormMask.shape", NormMask.shape)
            #print("NormMask", NormMask)
            visible_mask = (NormMask == 0) 
            invisible_mask = (NormMask == 1) 
            visibleNormY = NormY[visible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            invisibleNormY = NormY[invisible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            #print("visibleNormY.shape", visibleNormY.shape)
            #print("invisibleNormY.shape", invisibleNormY.shape)

            #print()
            #Normlatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            orig_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)
            #Advlatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)
            visibleAdvY = AdvY[visible_mask].reshape(AdvY.shape[0], -1, AdvY.shape[-1])
            invisibleAdvY = AdvY[invisible_mask].reshape(AdvY.shape[0], -1, AdvY.shape[-1])
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            #print("NormMask==AdvMask", NormMask==AdvMask)


            hookCount = 0
            loss = 0
            '''for layer, adv_output in list(adv_layerwise_outputs.items())[:48]:
                #orig_output = orig_layerwise_outputs[layer]
                #print("orig_output.shape", orig_output.shape)
                #loss +=  (1-cos(adv_output, orig_output))**2 #* cond_nums_normalized[hookCount]
                hookCount+=1'''
            #print("it should change here ")
            #print()
            lossBasedWeights = []
            for layer, adv_output in list(adv_layerwise_outputs.items())[48:]:
                orig_output = orig_layerwise_outputs[layer][:,1:,:]
                adv_output = adv_output[:,1:,:]
                invisible_orig_output = orig_output[invisible_mask].reshape(orig_output.shape[0], -1, orig_output.shape[-1])
                invisible_adv_output = adv_output[invisible_mask].reshape(adv_output.shape[0], -1, adv_output.shape[-1])

                theLoss = (1-cosForOa(invisible_orig_output, invisible_adv_output))**2 
                lossBasedWeights.append(theLoss.item())
                loss +=  theLoss * cond_nums_normalized[hookCount]
                hookCount+=1
            lossBasedWeights = (np.array(lossBasedWeights))**4
            cond_nums_normalized = lossBasedWeights / lossBasedWeights.sum()
            #print("lossBasedWeights.sum()", lossBasedWeights.sum())
            #print("cond_nums_normalized", cond_nums_normalized)
            total_loss =  -1 * loss * (1- cosForOa(invisibleNormY, invisibleAdvY))**2

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



if attck_type == "grill_cos_kf_FullMaskedSpot2Weights":
    #all_condition_nums = np.array(conditionNumberList)
    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    #all_condition_nums[all_condition_nums>100.0]=100
    #cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_cmpli = np.random.rand(total) 
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
            NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)
            #print("NormY.shape", NormY.shape)
            #print("NormMask.shape", NormMask.shape)
            #print("NormMask", NormMask)
            visible_mask = (NormMask == 0) 
            invisible_mask = (NormMask == 1) 
            visibleNormY = NormY[visible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            invisibleNormY = NormY[invisible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            #print("visibleNormY.shape", visibleNormY.shape)
            #print("invisibleNormY.shape", invisibleNormY.shape)

            #print()
            #Normlatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            orig_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)
            #Advlatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)
            visibleAdvY = AdvY[visible_mask].reshape(AdvY.shape[0], -1, AdvY.shape[-1])
            invisibleAdvY = AdvY[invisible_mask].reshape(AdvY.shape[0], -1, AdvY.shape[-1])
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            #print("NormMask==AdvMask", NormMask==AdvMask)


            hookCount = 0
            loss = 0
            lossBasedWeights = []
            for layer, adv_output in list(adv_layerwise_outputs.items())[:48]:
                orig_output = orig_layerwise_outputs[layer]
                #print("orig_output.shape", orig_output.shape)
                theLoss = (1-cos(adv_output, orig_output))**2 
                lossBasedWeights.append(theLoss.item())
                loss +=  theLoss * cond_nums_normalized[hookCount]
                hookCount+=1
            #print("it should change here ")
            #print()

            for layer, adv_output in list(adv_layerwise_outputs.items())[48:]:
                orig_output = orig_layerwise_outputs[layer][:,1:,:]
                adv_output = adv_output[:,1:,:]
                invisible_orig_output = orig_output[invisible_mask].reshape(orig_output.shape[0], -1, orig_output.shape[-1])
                invisible_adv_output = adv_output[invisible_mask].reshape(adv_output.shape[0], -1, adv_output.shape[-1])

                theLoss = (1-cosForOa(invisible_orig_output, invisible_adv_output))**2 
                lossBasedWeights.append(theLoss.item())
                loss +=  theLoss * cond_nums_normalized[hookCount]
                hookCount+=1
            lossBasedWeights = (np.array(lossBasedWeights))**2
            cond_nums_normalized = lossBasedWeights / lossBasedWeights.sum()
            #print("cond_nums_normalized", cond_nums_normalized)
            total_loss =  -1 * loss * (1- cosForOa(invisibleNormY, invisibleAdvY))**2

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






if attck_type == "grill_cos_kf_FullMaskedSpot4Weights":
    #all_condition_nums = np.array(conditionNumberList)
    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    #all_condition_nums[all_condition_nums>100.0]=100
    #cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_cmpli = np.random.rand(total) 
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
            NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)
            #print("NormY.shape", NormY.shape)
            #print("NormMask.shape", NormMask.shape)
            #print("NormMask", NormMask)
            visible_mask = (NormMask == 0) 
            invisible_mask = (NormMask == 1) 
            visibleNormY = NormY[visible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            invisibleNormY = NormY[invisible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            #print("visibleNormY.shape", visibleNormY.shape)
            #print("invisibleNormY.shape", invisibleNormY.shape)

            #print()
            #Normlatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            orig_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)
            #Advlatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)
            visibleAdvY = AdvY[visible_mask].reshape(AdvY.shape[0], -1, AdvY.shape[-1])
            invisibleAdvY = AdvY[invisible_mask].reshape(AdvY.shape[0], -1, AdvY.shape[-1])
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            #print("NormMask==AdvMask", NormMask==AdvMask)


            hookCount = 0
            loss = 0
            lossBasedWeights = []
            for layer, adv_output in list(adv_layerwise_outputs.items())[:48]:
                orig_output = orig_layerwise_outputs[layer]
                #print("orig_output.shape", orig_output.shape)
                theLoss = (1-cos(adv_output, orig_output))**2 
                lossBasedWeights.append(theLoss.item())
                loss +=  theLoss * cond_nums_normalized[hookCount]
                hookCount+=1
            #print("it should change here ")
            #print()

            for layer, adv_output in list(adv_layerwise_outputs.items())[48:]:
                orig_output = orig_layerwise_outputs[layer][:,1:,:]
                adv_output = adv_output[:,1:,:]
                invisible_orig_output = orig_output[invisible_mask].reshape(orig_output.shape[0], -1, orig_output.shape[-1])
                invisible_adv_output = adv_output[invisible_mask].reshape(adv_output.shape[0], -1, adv_output.shape[-1])

                theLoss = (1-cosForOa(invisible_orig_output, invisible_adv_output))**2 
                lossBasedWeights.append(theLoss.item())
                loss +=  theLoss * cond_nums_normalized[hookCount]
                hookCount+=1
            lossBasedWeights = (np.array(lossBasedWeights))**4
            cond_nums_normalized = lossBasedWeights / lossBasedWeights.sum()
            #print("cond_nums_normalized", cond_nums_normalized)
            total_loss =  -1 * loss * (1- cosForOa(invisibleNormY, invisibleAdvY))**2

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








if attck_type == "grill_cos_kf_mask":
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
            NormLoss, NormY, NormMask = model_mae_gan(images.float(), mask_ratio=set_mask_ratio)
            print("NormY.shape", NormY.shape)
            print("NormMask.shape", NormMask.shape)
            print("NormMask", NormMask)
            visible_mask = (NormMask == 0) 
            invisible_mask = (NormMask == 1) 
            visibleNormY = NormY[visible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            invisibleNormY = NormY[invisible_mask].reshape(NormY.shape[0], -1, NormY.shape[-1])
            print("visibleNormY.shape", visibleNormY.shape)
            print("invisibleNormY.shape", invisibleNormY.shape)

            print()
            #Normlatent, NormMask, NormIds_restore = model_mae_gan.forward_encoder(images.float(), mask_ratio=set_mask_ratio)
            orig_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            AdvLoss, AdvY, AdvMask = model_mae_gan(normalized_attacked, mask_ratio=set_mask_ratio)
            #Advlatent, AdvMask, AdvIds_restore = model_mae_gan.forward_encoder(normalized_attacked, mask_ratio=set_mask_ratio)
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            print("NormMask==AdvMask", NormMask==AdvMask)


            hookCount = 0
            loss = 0
            for layer, adv_output in list(adv_layerwise_outputs.items()):
                orig_output = orig_layerwise_outputs[layer]
                print("orig_output.shape", orig_output.shape)
                loss +=  (1-cos(adv_output, orig_output))**2 #* cond_nums_normalized[hookCount]
                hookCount+=1
            print("it should change here ")
            print()
            for layer, adv_output in list(adv_layerwise_outputs.items())[48:]:
                orig_output = orig_layerwise_outputs[layer]
                print("orig_output.shape", orig_output.shape)
                loss +=  (1-cos(adv_output, orig_output))**2 #* cond_nums_normalized[hookCount]
                hookCount+=1

            total_loss =  -1 * loss * (1- cosForOa(NormY, AdvY))**2

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
