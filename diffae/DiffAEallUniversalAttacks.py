


#%load_ext autoreload
#%autoreload 2

'''




####################################################################################################################################################################################################################################################################################




cd alma
conda activate dt2
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.18 --attck_type la_l2_kfAdamNoScheduler1 --which_gpu 5 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.19 --attck_type la_l2_kfAdamNoScheduler1 --which_gpu 5 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.20 --attck_type la_l2_kfAdamNoScheduler1 --which_gpu 5 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad



## pending
cd alma
conda activate dt2
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.18 --attck_type la_wass_kfAdamNoScheduler1 --which_gpu 3 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.19 --attck_type la_wass_kfAdamNoScheduler1 --which_gpu 3 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.20 --attck_type la_wass_kfAdamNoScheduler1 --which_gpu 3 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad


cd alma
conda activate dt2
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.18 --attck_type la_cos_kfAdamNoScheduler1 --which_gpu 4 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.19 --attck_type la_cos_kfAdamNoScheduler1 --which_gpu 4 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.20 --attck_type la_cos_kfAdamNoScheduler1 --which_gpu 4 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad



cd alma
conda activate dt2
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.18 --attck_type grill_l2_kfAdamNoScheduler1 --which_gpu 6 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.19 --attck_type grill_l2_kfAdamNoScheduler1 --which_gpu 6 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.20 --attck_type grill_l2_kfAdamNoScheduler1 --which_gpu 6 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad


## pending
cd alma
conda activate dt2
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.18 --attck_type grill_wass_kfAdamNoScheduler1 --which_gpu 2 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.19 --attck_type grill_wass_kfAdamNoScheduler1 --which_gpu 2 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.20 --attck_type grill_wass_kfAdamNoScheduler1 --which_gpu 2 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad


cd illcond
conda activate dt2
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.5 --attck_type grill_cos_kfAdamNoScheduler1 --which_gpu 7 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.19 --attck_type grill_cos_kfAdamNoScheduler1 --which_gpu 7 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.20 --attck_type grill_cos_kfAdamNoScheduler1 --which_gpu 7 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad




### universal attacks end
####################################################################################################################################################################################################################################################################################

####################################################################################################################################################################################################################################################################################
################################### weights abalation

python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.33 --attck_type grill_cos_pr1 --which_gpu 1 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.33 --attck_type grill_cos_pr_rnd1 --which_gpu 2 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.33 --attck_type grill_cos_pr_unif1 --which_gpu 3 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad

python diffae/review_plotting_abalation.py


################################### weights abalation ends
####################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################
##################################################################### to get histograms

python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.33 --attck_type la_cos_pr --which_gpu 4 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.33 --attck_type grill_cos_pr_rnd1 --which_gpu 2 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/review_plotting.py

##################################################################### to get histograms ends
####################################################################################################################################################################################################################################################################################





############# Adaptive attacks MCMC ################
Final adaptive 21 Nov



cd alma
conda activate dt2
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.20 --attck_type grill_cos_kfAdamNoScheduler1_mcmc --which_gpu 6 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.20 --attck_type la_cos_kfAdamNoScheduler1_mcmc --which_gpu 7 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad




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
parser.add_argument('--attck_type', type=str, default=5, help='Type of attack')
parser.add_argument('--desired_norm_l_inf', type=float, default=0.08, help='Type of attack')
parser.add_argument('--diffae_checkpoint', type=str, default=5, help='Type of attack')
parser.add_argument('--ffhq_images_directory', type=str, default=5, help='images directory')


args = parser.parse_args()

which_gpu = args.which_gpu
attck_type = args.attck_type
desired_norm_l_inf = args.desired_norm_l_inf
diffae_checkpoint = args.diffae_checkpoint
ffhq_images_directory = args.ffhq_images_directory

device = 'cuda:'+str(which_gpu)+''


conf = ffhq256_autoenc()

#conf = ffhq256_autoenc_latent()
print(conf.name)
model = LitModel(conf)
print("diffae_checkpoint", diffae_checkpoint)
#state = torch.load(f'../diffae/checkpoints/{conf.name}/last.ckpt', map_location='cpu')
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

batch_size = 25
train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

batch_list = []
for source_im in train_loader:
    batch = source_im['img'].to(device)
    batch_list.append(batch)  # Store batch in a list
    #print("len(batch_list)", len(batch_list))
    #if(len(batch_list)==3):
        #break

mi, ma = batch.min(), batch.max()


big_tensor = torch.stack(batch_list)  
print("big_tensor.shape", big_tensor.shape)
del batch_list
del train_loader


source_im = data[0]['img'][None].to(device)


import matplotlib.pyplot as plt
import os
# Construct the file path
#file_path = f"diffae/noise_storage/DiffAE_attack_type{attck_type}_norm_bound_{desired_norm_l_inf}_.pt"

#source_segment = 0

#file_path = f"../diffae/attack_run_time_univ/attack_noise/DiffAE_attack_typelatent_cosine_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(source_segment)+".pt"



noise_addition = (torch.randn_like(source_im) * 0.2).to(device)
noise_addition = noise_addition.clone().detach().requires_grad_(True)
optimizer = optim.Adam([noise_addition], lr=0.0001)

adv_alpha = 0.5

criterion = nn.MSELoss()

num_steps = 1000000
from geomloss import SamplesLoss


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

def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()
def poincare_cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    norm_a = torch.norm(a, p=2)
    norm_b = torch.norm(b, p=2)
    norm_a = torch.clamp(norm_a, max=1 - 1e-6)
    norm_b = torch.clamp(norm_b, max=1 - 1e-6)
    a = a / (1 - norm_a**2 + 1e-6)
    b = b / (1 - norm_b**2 + 1e-6)
    return (a * b).sum()
def lorentz_inner(x, y):
    time_like = -(x[0] * y[0])    
    space_like = (x[1:] * y[1:]).sum()
    return time_like + space_like
def hyperbolic_cos_hyperboloid(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = a / torch.sqrt(lorentz_inner(a, a) + 1e-6)
    b = b / torch.sqrt(lorentz_inner(b, b) + 1e-6)
    similarity = -lorentz_inner(a, b)
    return similarity
def poincare_distance(x, y):
    norm_x_sq = torch.sum(x**2, dim=-1, keepdim=True)
    norm_y_sq = torch.sum(y**2, dim=-1, keepdim=True)
    norm_x_sq = torch.clamp(norm_x_sq, max=1 - 1e-6)
    norm_y_sq = torch.clamp(norm_y_sq, max=1 - 1e-6)
    euclidean_sq = torch.sum((x - y)**2, dim=-1, keepdim=True)
    arg = 1 + 2 * euclidean_sq / ((1 - norm_x_sq) * (1 - norm_y_sq))
    arg = torch.clamp(arg, min=1 + 1e-6) 
    distance = -1 * torch.arccosh(arg)
    return distance.squeeze(-1)  
def angular_distance(x, y):
    dot_product = torch.sum(x * y, dim=-1, keepdim=True)
    norm_x = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) + 1e-6)
    norm_y = torch.sqrt(torch.sum(y**2, dim=-1, keepdim=True) + 1e-6)
    cosine_similarity = dot_product / (norm_x * norm_y)
    cosine_similarity = torch.clamp(cosine_similarity, min=-1.0, max=1.0)
    angular_dist = torch.arccos(cosine_similarity)
    return angular_dist.squeeze(-1)  # Remove singleton dimensions if any
def wasserstein_distance(tensor_a, tensor_b):
    tensor_a_flat = torch.flatten(tensor_a)
    tensor_b_flat = torch.flatten(tensor_b)
    tensor_a_sorted, _ = torch.sort(tensor_a_flat)
    tensor_b_sorted, _ = torch.sort(tensor_b_flat)    
    wasserstein_dist = torch.mean(torch.abs(tensor_a_sorted - tensor_b_sorted))
    return wasserstein_dist
def compute_mean_and_variance(tensor):
    flattened_tensor = torch.flatten(tensor)  
    mean = torch.mean(flattened_tensor) 
    variance = torch.var(flattened_tensor, unbiased=False)  
    return mean, variance
def get_symmetric_KLDivergence(input1, input2):
    mu1, var1 = compute_mean_and_variance(input1)
    mu2, var2 = compute_mean_and_variance(input2)    
    kl_1_to_2 = torch.log(var2 / var1) + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5
    kl_2_to_1 = torch.log(var1 / var2) + (var2 + (mu2 - mu1) ** 2) / (2 * var1) - 0.5
    symmetric_kl = (kl_1_to_2 + kl_2_to_1) / 2
    return symmetric_kl





def run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, optimized_noise, adv_gen):
    print(f"Step {step}, Loss: {total_loss.item()}, distortion L-2: {l2_distortion}, distortion L-inf: {l_inf_distortion}, deviation: {deviation}")
    print()
    print("attack type", attck_type)    
    adv_div_list.append(deviation.item())
    with torch.no_grad():
        fig, ax = plt.subplots(1, 3, figsize=(10, 10))
        ax[0].imshow(((normalized_attacked[0]+1)/2).permute(1, 2, 0).cpu().numpy())
        ax[0].set_title('Attacked Image')
        ax[0].axis('off')

        ax[1].imshow(optimized_noise[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        ax[1].set_title('Noise')
        ax[1].axis('off')

        ax[2].imshow(adv_gen[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        ax[2].set_title('Attack reconstruction')
        ax[2].axis('off')
        plt.show()
        plt.savefig("diffae/runtime_plots/DiffAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.png")   #####this
    torch.save(optimized_noise, "diffae/noise_storage/DiffAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")   #####this
    np.save("../diffae/attack_run_time_univ/adv_div_convergence/DiffAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", adv_div_list)

def get_latent_space_l2_loss(normalized_attacked, source_im):
    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))
    return F.mse_loss(embed, attacked_embed, reduction='sum')

def get_latent_space_cosine_loss(normalized_attacked, source_im):
    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))
    return cos(embed, attacked_embed)

def get_latent_space_cosine_loss_cr(normalized_attacked, source_im):
    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))
    cur_cos = 0
    for i in range(len(embed)):
        cur_cos += (cos(embed[i], attacked_embed[i]) - 1.0)**2 
    return cur_cos

def get_latent_space_l2_cosine_loss(normalized_attacked, source_im):
    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))
    return cos(embed, attacked_embed) + F.mse_loss(embed, attacked_embed, reduction='sum')

def get_latent_space_hyperbolic_loss1(normalized_attacked, source_im):
    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))
    return hyperbolic_cos_hyperboloid(embed, attacked_embed)

def get_latent_space_hyperbolic_loss2(normalized_attacked, source_im):
    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))
    return poincare_distance(embed, attacked_embed)

def get_latent_space_hyperbolic_loss3(normalized_attacked, source_im):
    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))
    return poincare_cos(embed, attacked_embed)

def get_latent_space_stat_l2_loss(normalized_attacked, source_im):
    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))
    xT_i = model.encode_stochastic(source_im.to(device), embed, T=2)
    xT_a = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=2)
    xT_i.requires_grad = True
    xT_a.requires_grad = True
    recon = model.render(xT_i, embed, T=2)
    adv_gen = model.render(xT_a, attacked_embed, T=2)
    return F.mse_loss(recon, adv_gen, reduction='sum') * F.mse_loss(embed, attacked_embed, reduction='sum')


def get_latent_space_wasserstein_loss(normalized_attacked, source_im):
    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))
    return wasserstein_distance(embed, attacked_embed)

def get_latent_space_wasserstein_loss_corrected(normalized_attacked, source_im):
    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))
    cur_was = 0
    for i in range(len(embed)):
        cur_was += wasserstein_distance(embed[i], attacked_embed[i])
    return cur_was

def get_latent_space_SKL_loss(normalized_attacked, source_im):
    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))
    return get_symmetric_KLDivergence(embed, attacked_embed)

def get_latent_space_mixed_sum_loss(normalized_attacked, source_im):
    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))
    return get_symmetric_KLDivergence(embed, attacked_embed) + F.mse_loss(embed, attacked_embed, reduction='sum') + get_symmetric_KLDivergence(embed, attacked_embed)

def get_latent_space_mixed_prod_loss(normalized_attacked, source_im):
    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))
    return get_symmetric_KLDivergence(embed, attacked_embed) * F.mse_loss(embed, attacked_embed, reduction='sum') * get_symmetric_KLDivergence(embed, attacked_embed)

def get_combined_l2_loss(normalized_attacked, source_im):
    x = source_im.to(device)  # Input batch
    x_p = normalized_attacked
    encoder_lip_sum = 0
    block_count = 0
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.out):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += F.mse_loss(x, x_p, reduction='sum') * (cond_nums_normalized[block_count])
        block_count+=1

    return encoder_lip_sum * F.mse_loss(x, x_p, reduction='sum') 

def get_combined_cosine_loss_backup(normalized_attacked_e, source_im):
    x = source_im.to(device)  # Input batch
    x_p = normalized_attacked
    encoder_lip_sum = 0
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += (cos(x, x_p)-1)**2 

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += (cos(x, x_p)-1)**2 

    for i, block in enumerate(model.ema_model.encoder.out):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += (cos(x, x_p)-1)**2 

    embed = model.encode(source_im.to(device))
    normalized_attacked_e = model.encode(normalized_attacked_e.to(device)) 
    return encoder_lip_sum * (cos(embed, attacked_embed)-1)**2 


def get_combined_cosine_loss(normal_x, source_x):

    encoder_lip_sum = 0
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2 

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2 

    for i, block in enumerate(model.ema_model.encoder.out):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2 

    return encoder_lip_sum * (cos(source_x, normal_x)-1)**2 



def get_combined_cosine_loss_cond(normal_x, source_x):

    encoder_lip_sum = 0
    block_count = 0
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2 * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.out):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_normalized[block_count])
        block_count+=1
        #print("block_count", block_count)

    return encoder_lip_sum * (cos(source_x, normal_x)-1)**2 



def get_combined_cosine_loss_cond_LaLo(normal_x, source_x):

    encoder_lip_sum = 0
    block_count = 0
    layerLossList = []
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        source_x = block(source_x)
        normal_x = block(normal_x)
        theLoss = (cos(source_x, normal_x)-1)**2 
        layerLossList.append(theLoss.item())
        encoder_lip_sum += theLoss * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        source_x = block(source_x)
        normal_x = block(normal_x)
        theLoss = (cos(source_x, normal_x)-1)**2 
        layerLossList.append(theLoss.item())
        encoder_lip_sum += theLoss * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.out):
        source_x = block(source_x)
        normal_x = block(normal_x)
        theLoss = (cos(source_x, normal_x)-1)**2 
        layerLossList.append(theLoss.item())
        encoder_lip_sum += theLoss  * (cond_nums_normalized[block_count])
        block_count+=1
        #print("block_count", block_count)

    return encoder_lip_sum * (cos(source_x, normal_x)-1)**2 , np.array(layerLossList)





def get_lat_cosine_loss_cond_LaLo(normal_x, source_x):

    encoder_lip_sum = 0
    block_count = 0
    layerLossList = []
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        source_x = block(source_x)
        normal_x = block(normal_x)
        theLoss = (cos(source_x, normal_x)-1)**2 
        layerLossList.append(theLoss.item())
        encoder_lip_sum += theLoss * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        source_x = block(source_x)
        normal_x = block(normal_x)
        theLoss = (cos(source_x, normal_x)-1)**2 
        layerLossList.append(theLoss.item())
        encoder_lip_sum += theLoss * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.out):
        source_x = block(source_x)
        normal_x = block(normal_x)
        theLoss = (cos(source_x, normal_x)-1)**2 
        layerLossList.append(theLoss.item())
        encoder_lip_sum += theLoss  * (cond_nums_normalized[block_count])
        block_count+=1
        #print("block_count", block_count)

    return (cos(source_x, normal_x)-1)**2 , np.array(layerLossList)




def get_combined_cosine_loss_cond_mcmc(normal_x, source_x):

    encoder_lip_sum = 0
    block_count = 0
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2 * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.out):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_normalized[block_count])
        block_count+=1
        #print("block_count", block_count)

        #normal_x = get_hmc_lat2(normal_x, normalized_attacked)
    normal_x = get_hmc_lat2(normal_x, normalized_attacked)


    return encoder_lip_sum * (cos(source_x, normal_x)-1)**2 




def get_latent_cosine_loss_cond_mcmc(normal_x, source_x):

    #encoder_lip_sum = 0
    #block_count = 0
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
        #print("block_count", block_count)

        #normal_x = get_hmc_lat2(normal_x, normalized_attacked)
    normal_x = get_hmc_lat2(normal_x, normalized_attacked)


    return (cos(source_x, normal_x)-1)**2 




def get_combined_cosine_loss_cond_cr(normal_x, source_x):

    encoder_lip_sum = 0
    block_count = 0
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        source_x = block(source_x)
        normal_x = block(normal_x)
        cur_cos = 0
        for i in range(len(normal_x)):
            cur_cos += (cos(source_x[i], normal_x[i]) - 1.0)**2 
        encoder_lip_sum += cur_cos * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        source_x = block(source_x)
        normal_x = block(normal_x)
        cur_cos = 0
        for i in range(len(normal_x)):
            cur_cos += (cos(source_x[i], normal_x[i]) - 1.0)**2 
        encoder_lip_sum += cur_cos * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.out):
        source_x = block(source_x)
        normal_x = block(normal_x)
        cur_cos = 0
        for i in range(len(normal_x)):
            cur_cos += (cos(source_x[i], normal_x[i]) - 1.0)**2 
        encoder_lip_sum += cur_cos * (cond_nums_normalized[block_count])
        block_count+=1
        #print("block_count", block_count)

        last_cos = 0
        for i in range(len(normal_x)):
            last_cos += (cos(source_x[i], normal_x[i]) - 1.0)**2 


    return encoder_lip_sum * last_cos


def get_pseudo_decoder(normal_x, source_x):

    for i, block in enumerate(model.ema_model.encoder.out):
        normal_x = block(normal_x)
        source_x = block(source_x)

    return normal_x, source_x

def get_hmc_lat1(normal_x, source_x):
    z = normal_x#.clone().detach().requires_grad_(True)  # Start point for MCMC
    #x = normalized_attacked#.detach()              # Adversarial input
    #source_x = source
    step_size = 0.2
    n_steps = 20
    leapfrog_steps = 15

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
        z = torch.where(mask.unsqueeze(1), z_new.view(x.size(0), -1), z.view(x.size(0), -1))
        z_new = z_new.view_as(source_x)
        z = z#.detach().requires_grad_(True)  # Prepare for next iteration
        z = z.view_as(source_x)
        #samples.append(z)

    z_mcmc = z #.detach()  # Final robust latent sample
    return z_mcmc

import time

def get_pseudo_decoder2(attacked_embed):
    xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=2)
    adv_gen = model.render(xT_ad, attacked_embed, T=2)
    return adv_gen

def get_hmc_lat2(z, x):
    #z = z1#.clone().detach().requires_grad_(True)  # Start point for MCMC

    #x = normalized_attacked#.detach()              # Adversarial input
    x_flat = x.view(x.size(0), -1)

    step_size = 0.008
    n_steps = 1
    leapfrog_steps = 1

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




def get_combined_cosine_loss_mcmc2(normal_x, source_x):

    encoder_lip_sum = 0
    block_count = 0
    normalized_attacked = normal_x
    #source = source_x

    with autocast():
        for i, block in enumerate(model.ema_model.encoder.input_blocks):
            source_x = block(source_x)
            normal_x = block(normal_x)
            encoder_lip_sum += (cos(source_x, normal_x)-1)**2 * (cond_nums_normalized[block_count])
            block_count+=1

        for i, block in enumerate(model.ema_model.encoder.middle_block):
            source_x = block(source_x)
            normal_x = block(normal_x)
            encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_normalized[block_count])
            block_count+=1


        for i, block in enumerate(model.ema_model.encoder.out):
            source_x = block(source_x)
            normal_x = block(normal_x)
            encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_normalized[block_count])
            block_count+=1

        normal_x = get_hmc_lat2(normal_x, normalized_attacked)
        #normal_x = get_hmc_lat2(source_x, source)

    return encoder_lip_sum * (cos(source_x, normal_x)-1)**2 



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

        normal_x = get_hmc_lat2(normal_x, normalized_attacked)
        #normal_x = get_hmc_lat2(source_x, source)

    return (cos(source_x, normal_x)-1)**2 



def get_combined_cosine_loss_mcmc(normal_x, source_x):

    encoder_lip_sum = 0
    block_count = 0
    #normalized_attacked = normal_x
    #source = source_x
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2 * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_normalized[block_count])
        block_count+=1

    normal_x = get_hmc_lat1(normal_x, source_x)

    for i, block in enumerate(model.ema_model.encoder.out):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_normalized[block_count])
        block_count+=1

    return encoder_lip_sum * (cos(source_x, normal_x)-1)**2 



def get_combined_cosine_loss1(normal_x, source_x):

    encoder_lip_sum = 0
    block_count = 0
    #normalized_attacked = normal_x
    #source = source_x
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2 * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_normalized[block_count])
        block_count+=1

    #normal_x = get_hmc_lat1(normal_x, source_x)

    for i, block in enumerate(model.ema_model.encoder.out):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_normalized[block_count])
        block_count+=1

    return encoder_lip_sum * (cos(source_x, normal_x)-1)**2 



def get_la_cosine_loss_mcmc(normal_x, source_x):

    #encoder_lip_sum = 0
    #block_count = 0
    #normalized_attacked = normal_x
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

    normal_x = get_hmc_lat1(normal_x, source_x)

    for i, block in enumerate(model.ema_model.encoder.out):
        source_x = block(source_x)
        normal_x = block(normal_x)
        #encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_normalized[block_count])
        #block_count+=1

    return (cos(source_x, normal_x)-1)**2 


def get_la_cosine_loss1(normal_x, source_x):


    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        source_x = block(source_x)
        normal_x = block(normal_x)

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        source_x = block(source_x)
        normal_x = block(normal_x)

    for i, block in enumerate(model.ema_model.encoder.out):
        source_x = block(source_x)
        normal_x = block(normal_x)

    return (cos(source_x, normal_x)-1)**2 


def get_combined_cosine_loss_cond_rnd(normal_x, source_x):

    encoder_lip_sum = 0
    block_count = 0
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2 * (cond_nums_rand[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_rand[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.out):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_rand[block_count])
        block_count+=1
        #print("block_count", block_count)

    return encoder_lip_sum * (cos(source_x, normal_x)-1)**2 


def get_combined_cosine_loss_cond_unif(normal_x, source_x):

    encoder_lip_sum = 0
    block_count = 0
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2 * (cond_nums_unif[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_unif[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.out):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum += (cos(source_x, normal_x)-1)**2  * (cond_nums_unif[block_count])
        block_count+=1
        #print("block_count", block_count)

    return encoder_lip_sum * (cos(source_x, normal_x)-1)**2 



def get_combined_cosine_loss_gcr(normal_x, source_x):

    encoder_lip_sum = 0
    block_count = 0
    layer_loss_list = []
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum = (cos(source_x, normal_x)-1)**2 #* (cond_nums_normalized[block_count])
        layer_loss_list.append(encoder_lip_sum)
        #block_count+=1

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum = (cos(source_x, normal_x)-1)**2  #* (cond_nums_normalized[block_count])
        layer_loss_list.append(encoder_lip_sum)
        #block_count+=1

    for i, block in enumerate(model.ema_model.encoder.out):
        source_x = block(source_x)
        normal_x = block(normal_x)
        encoder_lip_sum = (cos(source_x, normal_x)-1)**2  #* (cond_nums_normalized[block_count])
        layer_loss_list.append(encoder_lip_sum)
        #block_count+=1
        #print("block_count", block_count)
    #encoder_lip_sum = (cos(source_x, normal_x)-1)**2  #* (cond_nums_normalized[block_count])

    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))

    return (cos(embed, attacked_embed)-1)**2 , layer_loss_list





def get_combined_wasserstein_loss(normalized_attacked, source_im):

    x = source_im.to(device)  # Input batch
    x_p = normalized_attacked
    encoder_lip_sum = 0
    block_count = 0

    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += wasserstein_distance(x, x_p) * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += wasserstein_distance(x, x_p) * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.out):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += wasserstein_distance(x, x_p) * (cond_nums_normalized[block_count])
        block_count+=1

    return encoder_lip_sum * wasserstein_distance(x, x_p) 


def get_combined_wasserstein_loss_corrected(normalized_attacked, source_im):

    x = source_im.to(device)  # Input batch
    x_p = normalized_attacked
    encoder_lip_sum = 0
    block_count = 0

    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        x = block(x)
        x_p = block(x_p)
        cur_was = 0
        for i in range(len(x)):
            cur_was += wasserstein_distance(x[i], x_p[i])
        return cur_was
        encoder_lip_sum += cur_was * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        x = block(x)
        x_p = block(x_p)
        cur_was = 0
        for i in range(len(x)):
            cur_was += wasserstein_distance(x[i], x_p[i])
        return cur_was
        encoder_lip_sum += cur_was * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.out):
        x = block(x)
        x_p = block(x_p)
        cur_was = 0
        for i in range(len(x)):
            cur_was += wasserstein_distance(x[i], x_p[i])
        return cur_was
        encoder_lip_sum += cur_was * (cond_nums_normalized[block_count])
        block_count+=1

        last_was = 0
        for i in range(len(x)):
            last_was += wasserstein_distance(x[i], x_p[i])

    return encoder_lip_sum * last_was


def get_combined_SKL_loss(normalized_attacked, source_im):

    x = source_im.to(device)  # Input batch
    x_p = normalized_attacked
    encoder_lip_sum = 0
    block_count = 0

    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += get_symmetric_KLDivergence(x, x_p) * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += get_symmetric_KLDivergence(x, x_p) * (cond_nums_normalized[block_count])
        block_count+=1

    for i, block in enumerate(model.ema_model.encoder.out):
        x = block(x)
        x_p = block(x_p)
        encoder_lip_sum += get_symmetric_KLDivergence(x, x_p) * (cond_nums_normalized[block_count])
        block_count+=1

    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))

    return encoder_lip_sum * get_symmetric_KLDivergence(embed, attacked_embed) 


if(attck_type == "la_l2"):
    adv_div_list = []
    for step in range(155):
        batch_step = 0
        for source_im in big_tensor:
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

            loss_to_maximize = get_latent_space_l2_loss(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                normalized_attacked = normalized_attacked[0].unsqueeze(0)
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)






if(attck_type == "la_l2_kfAdamNoScheduler1"):
    adv_div_list = []
    for step in range(155):
        batch_step = 0
        for source_im in big_tensor:
            # Clamp perturbed image into valid image range, then update noise_addition in-place

            optimizer.zero_grad()
            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            # Compute cosine losses across encoder layers
            #last_layer_loss, layer_loss_list = get_combined_cosine_loss_gcr(normalized_attacked, source_im)

            # Loss to maximize: cosine difference between embeddings
            loss_to_maximize = get_latent_space_l2_loss(normalized_attacked, source_im)
            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()

            # Clamp noise within L∞-norm ball
            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        print("step", step)
        if(step%10==0  and step!=0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
                l2_distortion = torch.norm(noise_addition, p=2)
                l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, noise_addition, adv_gen)



if(attck_type == "la_wass_kfAdamNoScheduler1"):
    adv_div_list = []
    for step in range(155):
        batch_step = 0
        for source_im in big_tensor:
            # Clamp perturbed image into valid image range, then update noise_addition in-place

            optimizer.zero_grad()
            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            # Compute cosine losses across encoder layers
            #last_layer_loss, layer_loss_list = get_combined_cosine_loss_gcr(normalized_attacked, source_im)

            # Loss to maximize: cosine difference between embeddings
            loss_to_maximize = get_latent_space_wasserstein_loss_corrected(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()

            # Clamp noise within L∞-norm ball
            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        print("step", step)
        if(step%10==0  and step!=0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
                l2_distortion = torch.norm(noise_addition, p=2)
                l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, noise_addition, adv_gen)


if(attck_type == "la_cos_kfAdamNoScheduler1"):
    adv_div_list = []
    for step in range(155):
        batch_step = 0
        for source_im in big_tensor:
            # Clamp perturbed image into valid image range, then update noise_addition in-place

            optimizer.zero_grad()
            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            # Compute cosine losses across encoder layers
            #last_layer_loss, layer_loss_list = get_combined_cosine_loss_gcr(normalized_attacked, source_im)

            # Loss to maximize: cosine difference between embeddings
            loss_to_maximize = (get_latent_space_cosine_loss(normalized_attacked, source_im) - 1.0) ** 2
            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()

            # Clamp noise within L∞-norm ball
            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        print("step", step)
        if(step%10==0  and step!=0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
                l2_distortion = torch.norm(noise_addition, p=2)
                l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, noise_addition, adv_gen)



if(attck_type == "la_cos_kfAdamNoScheduler1_mcmc"):
    adv_div_list = []
    for step in range(100):
        batch_step = 0
        for source_im in big_tensor:
            # Clamp perturbed image into valid image range, then update noise_addition in-place

            optimizer.zero_grad()
            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            # Compute cosine losses across encoder layers
            #last_layer_loss, layer_loss_list = get_combined_cosine_loss_gcr(normalized_attacked, source_im)

            # Loss to maximize: cosine difference between embeddings
            #loss_to_maximize = (get_latent_space_cosine_loss(normalized_attacked, source_im) - 1.0) ** 2

            total_loss = -1 * get_latent_cosine_loss_cond_mcmc(normalized_attacked, source_im)
                            

            #total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()

            # Clamp noise within L∞-norm ball
            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        print("step", step)
        if(step%10==0  and step!=0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
                l2_distortion = torch.norm(noise_addition, p=2)
                l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, noise_addition, adv_gen)



if(attck_type == "grill_cos_kfAdamNoScheduler1"):
    adv_div_list = []
    for step in range(155):
        batch_step = 0
        for source_im in big_tensor:
            # Clamp perturbed image into valid image range, then update noise_addition in-place

            optimizer.zero_grad()
            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            # Compute cosine losses across encoder layers
            #last_layer_loss, layer_loss_list = get_combined_cosine_loss_gcr(normalized_attacked, source_im)

            # Loss to maximize: cosine difference between embeddings
            #loss_to_maximize = (get_latent_space_cosine_loss(normalized_attacked, source_im) - 1.0) ** 2

            total_loss = -1 * get_combined_cosine_loss_cond(normalized_attacked, source_im)


            #total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()

            # Clamp noise within L∞-norm ball
            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        print("step", step)
        if(step%10==0  and step!=0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
                l2_distortion = torch.norm(noise_addition, p=2)
                l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, noise_addition, adv_gen)




if(attck_type == "grill_cos_kfAdamNoScheduler1_mcmc"):
    adv_div_list = []
    for step in range(100):
        batch_step = 0
        for source_im in big_tensor:
            # Clamp perturbed image into valid image range, then update noise_addition in-place

            optimizer.zero_grad()
            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            # Compute cosine losses across encoder layers
            #last_layer_loss, layer_loss_list = get_combined_cosine_loss_gcr(normalized_attacked, source_im)

            # Loss to maximize: cosine difference between embeddings
            #loss_to_maximize = (get_latent_space_cosine_loss(normalized_attacked, source_im) - 1.0) ** 2

            total_loss = -1 * get_combined_cosine_loss_cond_mcmc(normalized_attacked, source_im)

            #total_loss = -1 * get_combined_cosine_loss_mcmc2(normalized_attacked, source_im)

            #total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()

            # Clamp noise within L∞-norm ball
            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        print("step", step)
        if(step%10==0  and step!=0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
                l2_distortion = torch.norm(noise_addition, p=2)
                l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, noise_addition, adv_gen)





if(attck_type == "grill_l2_kfAdamNoScheduler1"):
    adv_div_list = []
    for step in range(155):
        batch_step = 0
        for source_im in big_tensor:
            # Clamp perturbed image into valid image range, then update noise_addition in-place

            optimizer.zero_grad()
            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            # Compute cosine losses across encoder layers
            #last_layer_loss, layer_loss_list = get_combined_cosine_loss_gcr(normalized_attacked, source_im)

            # Loss to maximize: cosine difference between embeddings
            #loss_to_maximize = (get_latent_space_cosine_loss(normalized_attacked, source_im) - 1.0) ** 2

            total_loss = -1 * get_combined_l2_loss(normalized_attacked, source_im)

            #total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()

            # Clamp noise within L∞-norm ball
            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        print("step", step)
        if(step%10==0  and step!=0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
                l2_distortion = torch.norm(noise_addition, p=2)
                l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, noise_addition, adv_gen)



if(attck_type == "grill_wass_kfAdamNoScheduler1"):
    adv_div_list = []
    for step in range(155):
        batch_step = 0
        for source_im in big_tensor:
            # Clamp perturbed image into valid image range, then update noise_addition in-place

            optimizer.zero_grad()
            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            # Compute cosine losses across encoder layers
            #last_layer_loss, layer_loss_list = get_combined_cosine_loss_gcr(normalized_attacked, source_im)

            # Loss to maximize: cosine difference between embeddings
            #loss_to_maximize = (get_latent_space_cosine_loss(normalized_attacked, source_im) - 1.0) ** 2

            total_loss =  -1 * get_combined_wasserstein_loss_corrected(normalized_attacked, source_im)

            #total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()

            # Clamp noise within L∞-norm ball
            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        print("step", step)
        if(step%10==0  and step!=0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
                l2_distortion = torch.norm(noise_addition, p=2)
                l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, noise_addition, adv_gen)

