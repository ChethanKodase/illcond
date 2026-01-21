import torch
import torch.nn as nn
from model import AutoEncoder
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import random
import matplotlib.ticker as ticker

from torchvision import datasets, transforms
import os
import pandas as pd
import torch.optim as optim

'''

conda deactivate
conda deactivate
conda deactivate
export CUDA_VISIBLE_DEVICES=6
cd NVAE/
source nvaeenv1/bin/activate
cd ..
cd illcond/
python nvae/NvaeSampleSpecificQualitativePlotsForCLassicAttacks.py --desired_norm_l_inf 0.03 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeSampleSpecificQualitativePlotsForCLassicAttacks.py --desired_norm_l_inf 0.02 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeSampleSpecificQualitativePlotsForCLassicAttacks.py --desired_norm_l_inf 0.01 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint

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


import argparse

parser = argparse.ArgumentParser(description='Adversarial attack on VAE')
parser.add_argument('--desired_norm_l_inf', type=float, default="lip", help='Segment index')
parser.add_argument('--data_directory', type=str, default=0, help='data directory')
parser.add_argument('--nvae_checkpoint_path', type=str, default=0, help='nvae checkpoint directory')


args = parser.parse_args()

desired_norm_l_inf = args.desired_norm_l_inf
data_directory = args.data_directory
nvae_checkpoint_path = args.nvae_checkpoint_path


# Replace the placeholder values with your actual checkpoint path and parameters
checkpoint_path = ''+nvae_checkpoint_path+'/checkpoint.pt'
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


batch_size = 15
img_list = os.listdir(''+data_directory+'/smile/')
img_list.extend(os.listdir(''+data_directory+'/no_smile/'))

transform = transforms.Compose([
          transforms.Resize((64, 64)),
          transforms.ToTensor()
          ])
celeba_data = datasets.ImageFolder(data_directory, transform=transform)
split_train_frac = 0.95
train_set, test_set = torch.utils.data.random_split(celeba_data, [int(len(img_list) * split_train_frac), len(img_list) - int(len(img_list) * split_train_frac)])
train_data_size = len(train_set)
test_data_size = len(test_set)

print('train_data_size', train_data_size)
print('test_data_size', test_data_size)

trainLoader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True, drop_last=True)
testLoader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

########delete some stuff to clear memory
del celeba_data
del train_set
del test_set
del trainLoader


# Initialize a list to store batch tensors
batch_list = []

for idx, (source_im, _) in enumerate(testLoader):
    source_im, _ = source_im.cuda(), _
    batch_list.append(source_im)  # Store batch in a list
    #if(len(batch_list)==5):
    break


big_tensor = torch.stack(batch_list)  # Shape: (num_batches, batch_size, C, H, W)
#noise_addition = 2.0 * torch.rand(1, 3, 64, 64).cuda() - 1.0

mi, ma = big_tensor.min(), big_tensor.max()



#noise_addition = torch.zeros(1, 3, 64, 64).cuda()

def hook_fn(module, input, output):
    print(f"Layer: {module.__class__.__name__}")
    print(f"Output shape: {output.shape}")

print("what is going in", source_im.shape)
#source_im = source_im[0].unsqueeze(0)

noise_addition = (torch.randn_like(source_im) * 0.2).cuda()
noise_addition = noise_addition.clone().detach().requires_grad_(True)
optimizer = optim.Adam([noise_addition], lr=0.0001)


'''else:
    print("did it come here too ? ")
    noise_addition = 2.0 * torch.rand(1, 3, 64, 64).cuda() - 1.0
    noise_addition.requires_grad = True
    optimizer = optim.Adam([noise_addition], lr=0.0001)'''

adv_alpha = 0.5
criterion = nn.MSELoss()
num_steps = 40000
prev_loss = 0.0



print("what is going out again", noise_addition.shape)


# Dictionary to store layerwise outputs
layerwise_outputs = {}




def encoder_hook_fn(module, input, output):
    layerwise_outputs[module] = output

# Register hooks for encoder layers
encoder_hook_handles = []

for name, layer in model.enc_tower.named_modules():
    handle = layer.register_forward_hook(encoder_hook_fn)
    encoder_hook_handles.append(handle)




attck_types = ["la_l2_kf_SS", "la_wass_kf_cr_SS", "la_cos_kf_cr_SS", "grill_l2_kf_SS", "grill_wass_kf_SS", "grill_cos_kf_SS", ]

with torch.no_grad():

    row_one_ims = []
    row_two_ims = []
    #source_im = torch.load("/home/luser/NVAE/a_mixed_data/"+str(select_feature)+"_d/images.pt")[:50].unsqueeze(0).cuda()  
    #select_feature_dd = all_features[0]


    rec_logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps = model(source_im)
    reconstructed_output = model.decoder_output(rec_logits)
    rec_gen = reconstructed_output.sample()

    row_one_ims.append(source_im)
    row_two_ims.append(rec_gen)

    for i in range(len(attck_types)):

        optimized_noise = torch.load("nvae/univ_attack_storage/NVAE_attack_type"+str(attck_types[i])+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")

        with torch.no_grad():
            optimized_noise.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        normalized_attacked = torch.clamp(source_im + optimized_noise, mi, ma)

        print("normalized_attacked.shape", normalized_attacked.shape)
        adv_logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps = model(normalized_attacked)
        reconstructed_output = model.decoder_output(adv_logits)
        adv_gen = reconstructed_output.sample()

        row_one_ims.append(normalized_attacked)
        row_two_ims.append(adv_gen)


        del optimized_noise, normalized_attacked, adv_logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps, reconstructed_output, adv_gen

    all_row_ims = row_one_ims + row_two_ims
    num_ims = len(all_row_ims)


column_labels = ["Original \nImage", "LA \nL-2", "LA \nWass", "LA \nCos", "GRILL \nL-2", "GRILL \nWass", "GRILL \nCosine"]



for ch in range(15):
    with torch.no_grad():

        fig, axes = plt.subplots(
            2, num_ims // 2,
            figsize=(45, 10),
            gridspec_kw={'wspace': 0.02, 'hspace': 0.02},  # small gap
            constrained_layout=False
        )

        for idx, (ax, img) in enumerate(zip(axes.flat, all_row_ims)):
            ax.imshow(img[ch].permute(1, 2, 0).cpu().numpy())
            ax.set_aspect('equal')   # 🔑 preserve image shape
            ax.axis('off')

            # Titles only for top row
            if idx < num_ims // 2:
                col_index = idx % (num_ims // 2)
                ax.set_title(
                    column_labels[col_index],
                    fontsize=40,
                    pad=4      # small padding
                )

        # Fine-tune margins (not aggressive)
        plt.subplots_adjust(
            left=0.04, right=0.65,
            top=0.8, bottom=0.02,
            wspace=0.02, hspace=0.02
        )

        plt.savefig(
            f"SSqualitative/paperInd_{ch}_universal_NVAE_attacks_norm_bound_{desired_norm_l_inf}.png",
            bbox_inches='tight',
            pad_inches=0.01
        )

        plt.close(fig)