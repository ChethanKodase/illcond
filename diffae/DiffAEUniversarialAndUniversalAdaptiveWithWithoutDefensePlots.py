
'''



cd diffae

cd alma
conda activate dt2
python diffae/DiffAEUniversarialAndUniversalAdaptiveWithWithoutDefensePlots.py --desired_norm_l_inf 0.21 --which_gpu 7 --diffae_checkpoint diffae/checkpoints --ffhq_images_directory diffae/imgs_align_uni_ad --noise_directory diffae/noise_storage


python diffae/DiffAEUniversarialAndUniversalAdaptiveWithWithoutDefensePlots.py --desired_norm_l_inf 0.25 --which_gpu 6 --diffae_checkpoint diffae/checkpoints --ffhq_images_directory diffae/imgs_align_uni_ad --noise_directory diffae/noise_storage

python diffae/DiffAEUniversarialAndUniversalAdaptiveWithWithoutDefensePlots.py --desired_norm_l_inf 0.17 --which_gpu 6 --diffae_checkpoint diffae/checkpoints --ffhq_images_directory diffae/imgs_align_uni_ad --noise_directory diffae/noise_storage



python diffae/DiffAEUniversarialAndUniversalAdaptiveWithWithoutDefensePlots.py --desired_norm_l_inf 0.33 --which_gpu 7 --diffae_checkpoint diffae/checkpoints --ffhq_images_directory diffae/imgs_align_uni_ad --noise_directory diffae/attack_run_time_univ/attack_noise    # to get classic attacks 



cd alma
conda activate dt2
python diffae/DiffAEUniversarialAndUniversalAdaptiveWithWithoutDefensePlots.py --desired_norm_l_inf 0.3 --which_gpu 1 --diffae_checkpoint diffae/checkpoints --ffhq_images_directory diffae/imgs_align_uni_ad --noise_directory diffae/noise_storage

cd alma
conda activate dt2
python diffae/DiffAEUniversarialAndUniversalAdaptiveWithWithoutDefensePlots.py --desired_norm_l_inf 0.33 --which_gpu 1 --diffae_checkpoint diffae/checkpoints --ffhq_images_directory diffae/imgs_align_uni_ad --noise_directory diffae/noise_storage




'''


import numpy as np
from matplotlib import pyplot as plt
import torch
from templates import *
import torch
import ot

from torch.cuda.amp import autocast, GradScaler



import argparse

parser = argparse.ArgumentParser(description='DiffAE celebA training')


parser.add_argument('--desired_norm_l_inf', type=float, default=0.08, help='Type of attack')
parser.add_argument('--which_gpu', type=int, default=0, help='Index of the GPU to use (0-N)')
parser.add_argument('--diffae_checkpoint', type=str, default=5, help='Type of attack')
parser.add_argument('--ffhq_images_directory', type=str, default=5, help='images directory')
parser.add_argument('--noise_directory', type=str, default=5, help='images directory')



args = parser.parse_args()

desired_norm_l_inf = args.desired_norm_l_inf
which_gpu = args.which_gpu
diffae_checkpoint = args.diffae_checkpoint
ffhq_images_directory = args.ffhq_images_directory
noise_directory = args.noise_directory

adaptive = False
defending = False


#which_gpu = 7
source_segment = 0


#l_inf_bound = 0.12

#desired_norm_l_inf = 0.3
#desired_norm_l_inf = 0.33
#desired_norm_l_inf = 0.33



#vae_beta_value = 5.0
device = 'cuda:'+str(which_gpu)+''

def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
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

def calculate_psnr(img1, img2, max_val=1.0):
    # Mean Squared Error
    mse = F.mse_loss(img1, img2)
    # PSNR
    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr

import torch.nn.functional as F

def calculate_ssim(img1, img2, max_val=1.0, window_size=11, k1=0.01, k2=0.03):
    C1 = (k1 * max_val) ** 2
    C2 = (k2 * max_val) ** 2

    # Ensure img1 and img2 are float tensors
    img1 = img1.float()
    img2 = img2.float()
    
    # Create Gaussian kernel
    channels = img1.shape[1]  # Number of channels (e.g., 3 for RGB)
    window = torch.ones((channels, 1, window_size, window_size), device=img1.device) / (window_size ** 2)

    # Apply convolution with grouping
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

xts = []
for i in range(100):
    xts.append(i*10000)

#attack_types = ["latent_l2", "latent_wasserstein", "latent_SKL", "latent_cosine", "combi_l2", "combi_wasserstein", "combi_SKL", "combi_cos", "combi_cos_lw"]
#attack_types = [ "latent_cosine", "combi_cos_cond_corr", "combi_cos_cond_corr_cap", "combi_cos_cond_dir", "combi_cos_cond_dir_cap" ]
#attack_types = ["latent_l2", "latent_wasserstein", "latent_SKL", "latent_cosine", "combi_l2", "combi_wasserstein", "combi_SKL", "combi_cos_cond_dir_cap"]

#attack_types = ["latent_l2", "latent_wasserstein", "latent_SKL", "latent_cosine", "combi_l2", "combi_wasserstein", "combi_SKL", "combi_cos_cond_dir_cap"]

#attack_types = ["la_cos_mcmc", "alma_cos_mcmc"]
#attack_types = ["la_cos1", "alma_cos1"]
if adaptive:
    #attack_types = ["la_cos_mcmc2", "alma_cos_mcmc2"]  # perfect monte carlo attacls
    attack_types = ["la_cos_kfAdamNoScheduler1_mcmc", "grill_cos_kfAdamNoScheduler1_mcmc"]  # perfect monte carlo attacls
else:
    #attack_types = ["latent_cosine", "combi_cos_cond_dir_cap"] # classic attacks
    attack_types = ["la_cos_kfAdamNoScheduler1", "grill_cos_kfAdamNoScheduler1"] # classic attacks



#attack_types = ["la_l2_kfAdamNoScheduler1", "la_wass_kfAdamNoScheduler1", "latent_SKL", "la_cos_kfAdamNoScheduler1_mcmc", "grill_l2_kfAdamNoScheduler1", "grill_wass_kfAdamNoScheduler1", "combi_SKL", "grill_cos_kfAdamNoScheduler1_mcmc"]


conf = ffhq256_autoenc()

#conf = ffhq256_autoenc_latent()
print(conf.name)
model = LitModel(conf)
state = torch.load(f"{diffae_checkpoint}/{conf.name}/last.ckpt", map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.ema_model.eval()
model.ema_model.to(device);

#attack_types = ["combi_cos" ]


#data for universal attacks
data = ImageDataset(ffhq_images_directory, image_size=conf.img_size, exts=['jpg', 'JPG', 'png'], do_augment=False)
print("{len(data)}", len(data))
batch_size = 1
train_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4)


with torch.no_grad():

    batch_list = []
    for source_im in train_loader:
        batch = source_im['img'].to(device)  # Move batch to GPU
        #print("batch.shape", batch.shape)
        batch_list.append(batch)  # Store batch in a list
        #print("len(batch_list)", len(batch_list))
        if(len(batch_list)==10):
            break
    big_tensor = torch.stack(batch_list)  # Shape: (num_batches, batch_size, C, H, W)
print("big_tensor.shape", big_tensor.shape)
del batch_list
del train_loader

desired_norm_l_inf_temp = 0.08
n_projections = 100  # Start with 100 projections
num_features = 3 * 256 * 256  # Flattened input size
projections = torch.randn(num_features, n_projections).to(device)


#attack_types = ["latent_l2", "latent_wasserstein", "latent_SKL", "combi_l2", "combi_wasserstein", "combi_SKL"]
#attack_types = ["la_l2", "la_wass", "la_skl", "la_cos", "alma_l2", "alma_wass", "alma_skl", "alma_cos"]


def get_pseudo_decoder(normal_x, source_x):

    for i, block in enumerate(model.ema_model.encoder.out):
        normal_x = block(normal_x)
        source_x = block(source_x)

    return normal_x, source_x

def get_hmc_lat1(normal_x, source_x):
    z = normal_x.requires_grad_(True)#.clone().detach().requires_grad_(True)  # Start point for MCMC
    #x = normalized_attacked#.detach()              # Adversarial input
    #source_x = source ####  0.2 was fine

    #normal
    step_size = 0.2
    n_steps = 30
    leapfrog_steps = 20

    #extra
    step_size = 0.2
    n_steps = 50
    leapfrog_steps = 50


    #samples = []
    for i in range(n_steps):
        p = torch.randn_like(z)  # Sample momentum
        z_new = z.clone()
        p_new = p.clone()

        x_mean, x = get_pseudo_decoder(normal_x, source_x)


        x_flat, x_mean_flat = x.view(x.size(0), -1), x_mean.view(x.size(0), -1)

        log_p_x = -((x_flat - x_mean_flat) ** 2).sum(dim=1) / 2  # assuming Gaussian decoder

        log_p_z = -0.5 * (z_new.view(z_new.size(0), -1) ** 2).sum(dim=1)                # standard normal prior
        log_post = (log_p_x + log_p_z).sum()
        grad = torch.autograd.grad(log_post, z_new)[0]

        #print("grad.max()", grad.max())
        #print("grad.min()", grad.min())
        # Leapfrog integration
        p_new = p_new + 0.5 * step_size * grad
        for _ in range(leapfrog_steps):
            z_new = z_new + step_size * p_new
            z_new = z_new#.detach().requires_grad_(True)

            x_mean, x = get_pseudo_decoder(normal_x, source_x)

            x_flat, x_mean_flat = x.view(x.size(0), -1), x_mean.view(x.size(0), -1)
            log_p_x = -((x_flat - x_mean_flat) ** 2).sum(dim=1) / 2
            log_p_z = -0.5 * (z_new.view(z_new.size(0), -1) ** 2).sum(dim=1)
            log_post = (log_p_x + log_p_z).sum()
            grad = torch.autograd.grad(log_post, z_new)[0]
            p_new = p_new + step_size * grad
        p_new = p_new + 0.5 * step_size * grad
        p_new = -p_new  # Make symmetric

        z_decode = get_pseudo_decoder(normal_x, source_x)[0]
        z_new_decode = get_pseudo_decoder(z_new, source_x)[0]

        logp_current = -0.5 * (z.view(x.size(0), -1) ** 2).sum(dim=1) - ((x.view(x.size(0), -1) - z_decode.view(x.size(0), -1)) ** 2).sum(dim=1) / 2
        logp_new = -0.5 * (z_new.view(x.size(0), -1) ** 2).sum(dim=1) - ((x.view(x.size(0), -1) - z_new_decode.view(x.size(0), -1)) ** 2).sum(dim=1) / 2

        accept_ratio = torch.exp(logp_new - logp_current).clamp(max=1.0)
        mask = torch.rand_like(accept_ratio) < accept_ratio
        #print("mask", mask)
        z = torch.where(mask.unsqueeze(1), z_new.view(x.size(0), -1), z.view(x.size(0), -1))
        z_new = z_new.view_as(source_x)
        z = z#.detach().requires_grad_(True)  # Prepare for next iteration
        z = z.view_as(source_x)
        #samples.append(z)

    z_mcmc = z#.detach()  # Final robust latent sample
    return z_mcmc

def get_source_and_attacked_embeds_mcmc(normal_x, source_x):

    #encoder_lip_sum = 0
    #block_count = 0
    normalized_attacked = normal_x
    #source = source_x
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        source_x = block(source_x)
        normal_x = block(normal_x)
        #encoder_lip_sum += (cos(source_x, normal_x)-1)**2 * (cond_nums_normalized[block_count])
        #block_count+=1

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        source_x = block(source_x)
        normal_x = block(normal_x)
        #encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_normalized[block_count])
        #block_count+=1

    #normal_x = get_hmc_lat1(normal_x, source_x)

    for i, block in enumerate(model.ema_model.encoder.out):
        source_x = block(source_x)
        normal_x = block(normal_x)
        #encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_normalized[block_count])
        #block_count+=1
    if defending:
        normal_x = get_hmc_lat2(normal_x, normalized_attacked)
        

    return normal_x, source_x



def get_pseudo_decoder2(attacked_embed):
    xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=2)
    adv_gen = model.render(xT_ad, attacked_embed, T=2)
    return adv_gen

def get_hmc_lat2(z, x):
    z = z.requires_grad_(True)#.clone().detach().requires_grad_(True)  # Start point for MCMC

    #z = z1#.clone().detach().requires_grad_(True)  # Start point for MCMC

    #x = normalized_attacked#.detach()              # Adversarial input
    x_flat = x.view(x.size(0), -1)

    # Trial 1: n  = 1 , l = 1
    # Trial 2: n  = 2 , l = 4
    #normal
    #step_size = 0.008
    #n_steps = 2
    #leapfrog_steps = 2

    #extra
    step_size = 0.008
    n_steps = 20
    leapfrog_steps = 50

    with autocast(): 
        #samples = []
        for i in range(n_steps):
            p = torch.randn_like(z)  # Sample momentum
            z_new = z.clone()
            p_new = p.clone()
            x_mean = get_pseudo_decoder2(z_new)

            x_mean_flat = x_mean.view(x.size(0), -1)

            log_p_x = -((x_flat - x_mean_flat) ** 2).sum(dim=1) / 2  # assuming Gaussian decoder
            log_p_z = -0.5 * (z_new ** 2).sum(dim=1)                # standard normal prior
            log_post = (log_p_x + log_p_z).sum()
            grad = torch.autograd.grad(log_post, z_new)[0]

            # Leapfrog integration
            p_new = p_new + 0.5 * step_size * grad
            for _ in range(leapfrog_steps):
                z_new = z_new + step_size * p_new
                z_new = z_new#.detach().requires_grad_(True)
                x_mean = get_pseudo_decoder2(z_new)
                x_mean_flat = x_mean.view(x.size(0), -1)
                #x_flat, x_mean_flat = x.view(x.size(0), -1), x_mean.view(x.size(0), -1)
                log_p_x = -((x_flat - x_mean_flat) ** 2).sum(dim=1) / 2
                log_p_z = -0.5 * (z_new ** 2).sum(dim=1)
                log_post = (log_p_x + log_p_z).sum()
                grad = torch.autograd.grad(log_post, z_new)[0]
                p_new = p_new + step_size * grad
            p_new = p_new + 0.5 * step_size * grad
            p_new = -p_new  # Make symmetric

            with torch.no_grad():
                logp_current = -0.5 * (z ** 2).sum(dim=1) - ((x.view(x.size(0), -1) - get_pseudo_decoder2(z).view(x.size(0), -1)) ** 2).sum(dim=1) / 2
                #logp_new = -0.5 * (z_new ** 2).sum(dim=1) - ((x.view(x.size(0), -1) - get_pseudo_decoder2(z_new).view(x.size(0), -1)) ** 2).sum(dim=1) / 2
                logp_new = -0.5 * (z_new ** 2).sum(dim=1) - ((x.view(x.size(0), -1) - x_mean.view(x.size(0), -1)) ** 2).sum(dim=1) / 2


            accept_ratio = torch.exp(logp_new - logp_current).clamp(max=1.0)
            #print("accept_ratio", accept_ratio)
            #print("torch.rand_like(accept_ratio) ", torch.rand_like(accept_ratio))
            mask = torch.rand_like(accept_ratio) < accept_ratio
            z = torch.where(mask.unsqueeze(1), z_new, z)
            #z = z#.detach().requires_grad_(True)  # Prepare for next iteration
            #samples.append(z)
        z_mcmc = z#.detach()  # Final robust latent sample
        #print("z_mcmc.shape", z_mcmc.shape)
    return z_mcmc


def get_la_cosine_loss_mcmc2(normal_x, source_x):

    #encoder_lip_sum = 0
    #block_count = 0
    normalized_attacked = normal_x
    #source = source_x

    with autocast():
        for i, block in enumerate(model.ema_model.encoder.input_blocks):
            source_x = block(source_x)
            normal_x = block(normal_x)
            #encoder_lip_sum += (cos(source_x, normal_x)-1)**2 * (cond_nums_normalized[block_count])
            #block_count+=1

        for i, block in enumerate(model.ema_model.encoder.middle_block):
            source_x = block(source_x)
            normal_x = block(normal_x)
            #encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_normalized[block_count])
            #block_count+=1


        for i, block in enumerate(model.ema_model.encoder.out):
            source_x = block(source_x)
            normal_x = block(normal_x)
            #encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_normalized[block_count])
            #block_count+=1

        #normal_x = get_hmc_lat2(normal_x, normalized_attacked)
        #normal_x = get_hmc_lat2(source_x, source)

    return (cos(source_x, normal_x)-1)**2 


#with torch.no_grad():

for j in range(len(attack_types)):
    
    if adaptive :
        #optimized_noise = torch.load(""+noise_directory+"/DiffAE_attack_type"+str(attack_types[j])+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt").to(device)   # this is for adaptive attacks noises
        optimized_noise = torch.load("diffae/noise_storage/DiffAE_attack_type"+str(attack_types[j])+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt").to(device)   # this is for adaptive attacks noises

    else:
        #optimized_noise = torch.load(""+noise_directory+"/DiffAE_attack_type"+str(attack_types[j])+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(source_segment)+".pt").to(device)    # this is for classic attacks noises

        optimized_noise = torch.load("diffae/noise_storage/DiffAE_attack_type"+str(attack_types[j])+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt").to(device)   # this is for adaptive attacks noises




    all_divs_for_attack = []
    all_recon_loss = []
    all_adv_div_wass = []
    all_adv_abs_deviation = []
    all_adv_wass_deviation = []
    all_ssim_value=[]
    all_psnr_value = []
    k=0
    for source_im in big_tensor[2:]:
        print("k", k)
        mi, ma = source_im.min(), source_im.max()
        #attacked = source_im + optimized_noise
        #normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 
        normalized_attacked = torch.clamp(source_im + optimized_noise, mi, ma)


        attacked_embed, embed = get_source_and_attacked_embeds_mcmc(normalized_attacked.to(device), source_im.to(device))

        #with torch.no_grad():

        #attacked_embed = model.encode(normalized_attacked.to(device))
        xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
        adv_gen = model.render(xT_ad, attacked_embed, T=20)

        #embed = model.encode(source_im.to(device))
        xT_unp = model.encode_stochastic(source_im.to(device), embed, T=250)
        recons_unp = model.render(xT_unp, embed, T=20)

        recon_loss = torch.mean((adv_gen - normalized_attacked) ** 2)
        adv_deviation = torch.norm(adv_gen - recons_unp, p=2)  # p=2 for L2 norm

        adv_abs_deviation = torch.sum(abs(adv_gen - recons_unp))

        adv_wass_deviation = wasserstein_distance(adv_gen, recons_unp)

        psnr_value = calculate_psnr(adv_gen, recons_unp, max_val=1.0)
        ssim_value = calculate_ssim(adv_gen, recons_unp, max_val=1.0)


        print("adv_gen.shape", adv_gen.shape)
        print("recons_unp.shape", recons_unp.shape)
        print("normalized_attacked.shape", normalized_attacked.shape)
        adv_gen_flat = adv_gen.view(adv_gen.shape[0], -1)  # Flatten to [1, 196608]
        recons_unp_flat = recons_unp.view(recons_unp.shape[0], -1)
        adv_div_wass = ot.sliced_wasserstein_distance(adv_gen_flat, recons_unp_flat, projections=projections)


        all_divs_for_attack.append(adv_deviation.item())
        all_recon_loss.append(recon_loss.item())
        all_adv_div_wass.append(adv_div_wass.item())
        all_adv_abs_deviation.append(adv_abs_deviation.item())
        all_adv_wass_deviation.append(adv_wass_deviation.item())
        all_psnr_value.append(psnr_value.item())
        all_ssim_value.append(ssim_value.item())
        print("attack_types[j]", attack_types[j])
        print("adv_deviation.item()", adv_deviation.item())
        print("recon_loss.item()", recon_loss.item())
        print("adv_div_wass.item()", adv_div_wass.item())
        print("adv_abs_deviation.item()", adv_abs_deviation.item())
        print("adv_wass_deviation.item()", adv_wass_deviation.item())
        print("psnr_value.item()", psnr_value.item())
        print("ssim_value.item()", ssim_value.item())


        print("len(all_divs_for_attack)", len(all_divs_for_attack))
    
        if(k%20==0):
            with torch.no_grad():
                plt.imshow(adv_gen[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
                plt.axis('off')
                plt.show()
                plt.savefig("diffae/defense_efficacy/adv_DiffAE_attack_type"+str(attack_types[j])+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(k)+".png", bbox_inches='tight')
                plt.close()

                plt.imshow(recons_unp[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
                plt.axis('off')
                plt.show()
                plt.savefig("diffae/defense_efficacy/recon_DiffAE_attack_type"+str(attack_types[j])+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(k)+".png", bbox_inches='tight')
                plt.close()

                plt.imshow(((normalized_attacked[0]+1)/2).cpu().detach().permute(1, 2, 0).cpu().numpy())
                plt.axis('off')
                plt.show()
                plt.savefig("diffae/defense_efficacy/normalized_attacked_DiffAE_attack_type"+str(attack_types[j])+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(k)+".png", bbox_inches='tight')
                plt.close()

                plt.imshow(((source_im[0]+1)/2).cpu().detach().permute(1, 2, 0).cpu().numpy())
                plt.axis('off')
                plt.show()
                plt.savefig("diffae/defense_efficacy/source_img_DiffAE_attack_type"+str(attack_types[j])+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(k)+".png", bbox_inches='tight')
                plt.close()

        k+=1
        #if(k==400):
        break


