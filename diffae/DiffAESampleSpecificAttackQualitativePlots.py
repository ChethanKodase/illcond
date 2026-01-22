


#%load_ext autoreload
#%autoreload 2

'''

####################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################




cd alma
conda activate dt2
python diffae/DiffAESampleSpecificAttackQualitativePlots.py --desired_norm_l_inf 0.18 --which_gpu 1 --diffae_checkpoint diffae/checkpoints --ffhq_images_directory diffae/imgs_align_uni_ad

####################################################################################################################################################################################################################################################################################



'''


from templates import *
import matplotlib.pyplot as plt
import torch.optim as optim
import matplotlib.ticker as ticker

from torch.nn import DataParallel
import torch.nn.functional as F

from torch.utils.data import DataLoader
from conditioning import get_layer_pert_recon

import argparse
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

parser = argparse.ArgumentParser(description='DiffAE celebA training')

parser.add_argument('--which_gpu', type=int, default=0, help='Index of the GPU to use (0-N)')
parser.add_argument('--desired_norm_l_inf', type=float, default=0.08, help='Type of attack')
parser.add_argument('--diffae_checkpoint', type=str, default=5, help='Type of attack')
parser.add_argument('--ffhq_images_directory', type=str, default=5, help='images directory')


args = parser.parse_args()

which_gpu = args.which_gpu
desired_norm_l_inf = args.desired_norm_l_inf
diffae_checkpoint = args.diffae_checkpoint
ffhq_images_directory = args.ffhq_images_directory

device = 'cuda:'+str(which_gpu)+''


conf = ffhq256_autoenc()

#conf = ffhq256_autoenc_latent()
print(conf.name)
model = LitModel(conf)
print("diffae_checkpoint", diffae_checkpoint)
#state = torch.load(f'diffae/checkpoints/{conf.name}/last.ckpt', map_location='cpu')
state = torch.load(f"{diffae_checkpoint}/{conf.name}/last.ckpt", map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device);


total_params = sum(p.numel() for p in model.ema_model.parameters())
trainable_params = sum(p.numel() for p in model.ema_model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

data = ImageDataset(ffhq_images_directory, image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)

print("{len(data)}", len(data))

batch_size = 15
train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

batch_list = []
for source_im in train_loader:
    source_im = source_im['img'].to(device)
    #batch_list.append(batch)  # Store batch in a list
    #print("len(batch_list)", len(batch_list))
    #if(len(batch_list)==3):
    break

mi, ma = source_im.min().item(), source_im.max().item()


#big_tensor = torch.stack(batch_list)  # This we do to put all the images into the GPU so that there is no latency due to communication between CPU and GPU during optimization
print("big_tensor.shape", source_im.shape)
#del batch_list
del train_loader


#source_im = data[0]['img'][None].to(device)


import matplotlib.pyplot as plt
import os
# Construct the file path
#file_path = f"diffae/noise_storage/DiffAE_attack_type{attck_type}_norm_bound_{desired_norm_l_inf}_.pt"

source_segment = 0

file_path = f"diffae/attack_run_time_univ/attack_noise/DiffAE_attack_typelatent_cosine_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(source_segment)+".pt"

adv_alpha = 0.5

criterion = nn.MSELoss()

num_steps = 1000000


with torch.no_grad():
    cond_nums_normalized = get_layer_pert_recon(model)
    print("cond_nums_normalized", cond_nums_normalized)

print("cond_nums_normalized.shape", cond_nums_normalized.shape)
print("cond_nums_normalized.max()", cond_nums_normalized.max())
print("cond_nums_normalized.min()", cond_nums_normalized.min())

cond_nums_rand = np.random.rand(29)

print("cond_nums_rand", cond_nums_rand)

cond_nums_unif = np.full((29,), 1 / 29)

print(cond_nums_unif)
print("Sum:", np.sum(cond_nums_unif))

#################################################################################################################




#attck_types = ["la_l2_kf_SS", "la_wass_kf_cr_SS", "la_cos_kf_cr_SS", "grill_l2_kf_SS", "grill_wass_kf_SS", "grill_cos_kf_SS", ]
attck_types = ["la_l2_kfAdamNoScheduler1_SS", "la_wass_kfAdamNoScheduler1_SS", "la_cos_kfAdamNoScheduler1_SS", "grill_l2_kfAdamNoScheduler1_SS", "grill_wass_kfAdamNoScheduler1_SS", "grill_cos_kfAdamNoScheduler1_SS", ]

with torch.no_grad():

    row_one_ims = []
    row_two_ims = []


    Nembed = model.encode(source_im)
    NxT = model.encode_stochastic(source_im, Nembed, T=250)
    rec_gen = model.render(NxT, Nembed, T=20)


    row_one_ims.append((source_im+1)/2)
    row_two_ims.append(rec_gen)

    for i in range(len(attck_types)):

        #optimized_noise = torch.load(""+uni_noise_path+"/NVAE_attack_type"+str(attck_types[i])+"_norm_bound_"+str(desired_norm_l_inf)+"feature_"+str(select_feature)+"_source_segment_"+str(source_segment)+"_.pt")

        #optimized_noise = torch.load("nvae/univ_attack_storage/NVAE_attack_type"+str(attck_types[i])+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")
        optimized_noise = torch.load("diffae/noise_storage/DiffAE_attack_type"+str(attck_types[i])+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt").to(device)


        #with torch.no_grad():
        optimized_noise.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        normalized_attacked = torch.clamp(source_im + optimized_noise, mi, ma)

        print("normalized_attacked.shape", normalized_attacked.shape)
        '''adv_logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps = model(normalized_attacked)
        reconstructed_output = model.decoder_output(adv_logits)
        adv_gen = reconstructed_output.sample()'''

        attacked_embed = model.encode(normalized_attacked.to(device))
        xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
        adv_gen = model.render(xT_ad, attacked_embed, T=20)

        row_one_ims.append((normalized_attacked+1)/2)
        row_two_ims.append(adv_gen)


        del optimized_noise, normalized_attacked, adv_gen

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
            f"diffae/SSqualitative/paperInd_{ch}_universal_NVAE_attacks_norm_bound_{desired_norm_l_inf}.png",
            bbox_inches='tight',
            pad_inches=0.01
        )

        plt.close(fig)
