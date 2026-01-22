


#%load_ext autoreload
#%autoreload 2

'''

cd alma
conda activate dt2
python diffae/autoencoding_attack_grad_image.py --desired_norm_l_inf 0.35 --attck_type la_l2 --which_gpu 7 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad



conda activate dt2
python diffae/autoencoding_attack_grad_image.py --desired_norm_l_inf 0.31 --attck_type la_l2 --which_gpu 3 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/autoencoding_attack_grad_image.py --desired_norm_l_inf 0.31 --attck_type la_wass --which_gpu 5 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/autoencoding_attack_grad_image.py --desired_norm_l_inf 0.31 --attck_type la_skl --which_gpu 6 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/autoencoding_attack_grad_image.py --desired_norm_l_inf 0.21 --attck_type la_cos --which_gpu 7 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad

python diffae/autoencoding_attack_grad_image.py --desired_norm_l_inf 0.31 --attck_type alma_l2 --which_gpu 7 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/autoencoding_attack_grad_image.py --desired_norm_l_inf 0.27 --attck_type alma_wass --which_gpu 1 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/autoencoding_attack_grad_image.py --desired_norm_l_inf 0.27 --attck_type alma_skl --which_gpu 2 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/autoencoding_attack_grad_image.py --desired_norm_l_inf 0.21 --attck_type alma_cos --which_gpu 7 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad



cd alma
conda activate dt2
python diffae/autoencoding_attack_grad_image.py --desired_norm_l_inf 0.36 --attck_type gcr_cos --which_gpu 5 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad --chosen_space_ind 17

cd alma
conda activate dt2
python diffae/autoencoding_attack_grad_image.py --desired_norm_l_inf 0.27 --attck_type gcr_cos --which_gpu 4 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad --chosen_space_ind 16



cd alma
conda activate dt2
python diffae/autoencoding_attack_grad_image.py --desired_norm_l_inf 0.27 --attck_type gcr_cos --which_gpu 6 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad --chosen_space_ind 15



cd alma
conda activate dt2
python diffae/autoencoding_attack_grad_image.py --desired_norm_l_inf 0.27 --attck_type gcr_cos_simp2 --which_gpu 1 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad --chosen_space_ind 14

'''


from templates import *
import matplotlib.pyplot as plt
import torch.optim as optim

from torch.nn import DataParallel
import torch.nn.functional as F

from torch.utils.data import DataLoader
from conditioning import get_layer_pert_recon


#seeding code begins
import torch
import random
import numpy as np
import os

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#seeding code ends


import argparse

parser = argparse.ArgumentParser(description='DiffAE celebA training')

parser.add_argument('--which_gpu', type=int, default=0, help='Index of the GPU to use (0-N)')
parser.add_argument('--attck_type', type=str, default=5, help='Type of attack')
parser.add_argument('--desired_norm_l_inf', type=float, default=0.08, help='Type of attack')
parser.add_argument('--diffae_checkpoint', type=str, default=5, help='Type of attack')
parser.add_argument('--ffhq_images_directory', type=str, default=5, help='images directory')

parser.add_argument('--chosen_space_ind', type=int, default=5, help='images directory')


args = parser.parse_args()

which_gpu = args.which_gpu
attck_type = args.attck_type
desired_norm_l_inf = args.desired_norm_l_inf
diffae_checkpoint = args.diffae_checkpoint
ffhq_images_directory = args.ffhq_images_directory
chosen_space_ind = args.chosen_space_ind

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
    print("len(batch_list)", len(batch_list))
    if(len(batch_list)==3):
        break
big_tensor = torch.stack(batch_list)  # This we do to put all the images into the GPU so that there is no latency due to communication between CPU and GPU during optimization
print("big_tensor.shape", big_tensor.shape)
del batch_list
del train_loader


source_im = data[0]['img'][None].to(device)


import matplotlib.pyplot as plt
import os
# Construct the file path
#file_path = f"diffae/noise_storage/DiffAE_attack_type{attck_type}_norm_bound_{desired_norm_l_inf}_.pt"

source_segment = 0
file_path = f"/data1/chethan/diffae/attack_run_time_univ/attack_noise/DiffAE_attack_typelatent_cosine_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(source_segment)+".pt"

# Check if the file exists
if (attck_type=="gcr_cos_use_converged" and os.path.exists(file_path)):
    noise_addition = torch.load(file_path).to(device)
    noise_addition = noise_addition.clone().detach().requires_grad_(True)

    print("file exists")

'''if (attck_type=="gcr_cos" and os.path.exists(file_path)):
    noise_addition = torch.rand(source_im.shape).to(device)
    print("no file, so random initialization")

    noise_addition = noise_addition * (source_im.max() - source_im.min()) + source_im.min()  
    noise_addition = noise_addition.clone().detach().requires_grad_(True)'''

if (attck_type=="gcr_cos" and os.path.exists(file_path)):
    noise_addition = torch.rand(source_im.shape).to(device)
    print("no file, so random initialization")

    noise_addition = noise_addition * (source_im.max() - source_im.min()) + source_im.min()  
    noise_addition = noise_addition.clone().detach().requires_grad_(True)

if (attck_type=="gcr_cos"):
    noise_addition = torch.rand(source_im.shape).to(device)
    print("no file, so random initialization")

    noise_addition = noise_addition * (source_im.max() - source_im.min()) + source_im.min()  
    noise_addition = noise_addition.clone().detach().requires_grad_(True)


if (attck_type=="gcr_cos_simp" and os.path.exists(file_path)):
    noise_addition = torch.rand(source_im.shape).to(device)
    print("no file, so random initialization")

    noise_addition = noise_addition * (source_im.max() - source_im.min()) + source_im.min()  
    noise_addition = noise_addition.clone().detach().requires_grad_(True)

if (attck_type=="gcr_cos_simp2" and os.path.exists(file_path)):
    noise_addition = torch.rand(source_im.shape).to(device)
    print("no file, so random initialization")

    noise_addition = noise_addition * (source_im.max() - source_im.min()) + source_im.min()  
    noise_addition = noise_addition.clone().detach().requires_grad_(True)


if (attck_type=="gcr_cos_all_layers" and os.path.exists(file_path)):
    noise_addition = torch.rand(source_im.shape).to(device)
    print("no file, so random initialization")

    noise_addition = noise_addition * (source_im.max() - source_im.min()) + source_im.min()  
    noise_addition = noise_addition.clone().detach().requires_grad_(True)

if(attck_type=="random_init_la_cos"):
    noise_addition = torch.rand(source_im.shape).to(device)
    print("no file, so random initialization")

    noise_addition = noise_addition * (source_im.max() - source_im.min()) + source_im.min()  
    noise_addition = noise_addition.clone().detach().requires_grad_(True)
'''else:
    noise_addition = torch.rand(source_im.shape).to(device)
    print("no file, so random initialization")

    noise_addition = noise_addition * (source_im.max() - source_im.min()) + source_im.min()  
    noise_addition = noise_addition.clone().detach().requires_grad_(True)'''

#noise_addition.requires_grad = True
optimizer = optim.Adam([noise_addition], lr=0.0001)
source_im.requires_grad = True

adv_alpha = 0.5

criterion = nn.MSELoss()

num_steps = 1000000
from geomloss import SamplesLoss


with torch.no_grad():
    cond_nums_normalized = get_layer_pert_recon(model)
    print("cond_nums_normalized", cond_nums_normalized)
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
        plt.savefig("diffae/runtime_plots/DiffAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.png")
    torch.save(optimized_noise, "diffae/noise_storage/DiffAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")
    #np.save("../diffae/attack_run_time_univ/adv_div_convergence/DiffAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", adv_div_list)

def get_latent_space_l2_loss(normalized_attacked, source_im):
    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))
    return F.mse_loss(embed, attacked_embed, reduction='sum')

def get_latent_space_cosine_loss(normalized_attacked, source_im):
    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))
    return cos(embed, attacked_embed)

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



def get_combined_cosine_loss_gcr_simp(normal_x, source_x):
    encoder_lip_sum = 0
    layer_loss_list = []

    # Store intermediate tensors for later gradient inspection
    retained_outputs = []

    # Input blocks
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        source_x = block(source_x)
        normal_x = block(normal_x)

        # Retain gradients of intermediate outputs
        #source_x.retain_grad()
        normal_x.retain_grad()
        retained_outputs.append((f"input_block_{i}", source_x, normal_x))

        loss = ((F.cosine_similarity(source_x, normal_x, dim=1) - 1) ** 2).mean()
        encoder_lip_sum += loss
        layer_loss_list.append(loss)

    # Middle block
    for i, block in enumerate(model.ema_model.encoder.middle_block):
        source_x = block(source_x)
        normal_x = block(normal_x)

        #source_x.retain_grad()
        normal_x.retain_grad()
        retained_outputs.append((f"middle_block_{i}", source_x, normal_x))

        loss = ((F.cosine_similarity(source_x, normal_x, dim=1) - 1) ** 2).mean()
        encoder_lip_sum += loss
        layer_loss_list.append(loss)

    # Output block
    for i, block in enumerate(model.ema_model.encoder.out):
        source_x = block(source_x)
        normal_x = block(normal_x)

        #source_x.retain_grad()
        normal_x.retain_grad()
        retained_outputs.append((f"out_block_{i}", source_x, normal_x))

        loss = ((F.cosine_similarity(source_x, normal_x, dim=1) - 1) ** 2).mean()
        encoder_lip_sum += loss
        layer_loss_list.append(loss)

    # Final output embedding
    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))
    final_loss = ((F.cosine_similarity(embed, attacked_embed, dim=1) - 1) ** 2).mean()

    total_loss = final_loss + encoder_lip_sum

    # Backward pass
    total_loss.backward()

    # Print gradients of retained outputs
    for name, sx, nx in retained_outputs:
        #print(f"Gradient for {name} (source_x): {sx.grad.shape} - mean grad: {sx.grad.abs().mean().item():.6f}")
        print(f"Gradient for {name} (normal_x): {nx.grad.shape} - mean grad: {nx.grad.abs().mean().item():.6f}")

    return final_loss, layer_loss_list




def get_combined_cosine_loss_gcr_simp2(normal_x, source_x):

    retained_outputs = []

    # Input blocks
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        source_x = block(source_x)
        normal_x = block(normal_x)

        normal_x.retain_grad()
        retained_outputs.append(normal_x)


    for i, block in enumerate(model.ema_model.encoder.middle_block):
        source_x = block(source_x)
        normal_x = block(normal_x)

        normal_x.retain_grad()
        retained_outputs.append(normal_x)

    for i, block in enumerate(model.ema_model.encoder.out):
        source_x = block(source_x)
        normal_x = block(normal_x)

        normal_x.retain_grad()
        retained_outputs.append(normal_x)

    final_loss = -1 * ((F.cosine_similarity(source_x, normal_x, dim=1) - 1) ** 2).mean()

    return final_loss, retained_outputs


def get_intermediate_grads_from_final_loss(source_x, normal_x):
    intermediate_outputs = []

    source_x.requires_grad_(True)
    normal_x.requires_grad_(True)

    # Go through encoder blocks
    for i, block in enumerate(model.ema_model.encoder.input_blocks):
        source_x = block(source_x)
        normal_x = block(normal_x)

        # Retain grads for analysis
        source_x.retain_grad()
        normal_x.retain_grad()
        intermediate_outputs.append((f"input_block_{i}", source_x, normal_x))

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        source_x = block(source_x)
        normal_x = block(normal_x)
        source_x.retain_grad()
        normal_x.retain_grad()
        intermediate_outputs.append((f"middle_block_{i}", source_x, normal_x))

    for i, block in enumerate(model.ema_model.encoder.out):
        source_x = block(source_x)
        normal_x = block(normal_x)
        source_x.retain_grad()
        normal_x.retain_grad()
        intermediate_outputs.append((f"out_block_{i}", source_x, normal_x))

    # Only use final embeddings for loss
    embed = model.encode(source_im.to(device))
    attacked_embed = model.encode(normalized_attacked.to(device))
    final_loss = ((F.cosine_similarity(embed, attacked_embed, dim=1) - 1) ** 2).mean()

    # Backward from final loss only
    final_loss.backward()

    # Inspect intermediate gradients
    for name, sx, nx in intermediate_outputs:
        if sx.grad is not None and nx.grad is not None:
            print(f"[{name}] source_x grad mean: {sx.grad.abs().mean().item():.6f}")
            print(f"[{name}] normal_x grad mean: {nx.grad.abs().mean().item():.6f}")
        else:
            print(f"[{name}] Gradient not available (possibly disconnected from loss)")

    return final_loss


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


if(attck_type == "la_wass"):
    adv_div_list = []
    for step in range(155):
        batch_step = 0
        for source_im in big_tensor:
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

            loss_to_maximize = get_latent_space_wasserstein_loss(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)

if(attck_type == "la_skl"):
    adv_div_list = []
    for step in range(155):
        batch_step = 0
        for source_im in big_tensor:
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

            loss_to_maximize = get_latent_space_SKL_loss(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)

if(attck_type == "la_cos"):
    adv_div_list = []
    for step in range(155):
        batch_step = 0
        for source_im in big_tensor:
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

            loss_to_maximize = (get_latent_space_cosine_loss(normalized_attacked, source_im)-1.0)**2 

            total_loss = -1 * loss_to_maximize
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "alma_cos"):
    adv_div_list = []
    for step in range(155):
        for source_im in big_tensor:
            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) ))
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min()))  + source_im.min() 
            total_loss = -1 * get_combined_cosine_loss_cond(normalized_attacked, source_im)
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                scaled_noise = noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)


balance = 0.3
if(attck_type == "gcr_cos"):
    adv_div_list = []
    epoch_list = []
    loss_list = []
    sel_layer_loss_list = []
    for step in range(155):
        batch_ali = []
        loss_ali = []
        layer_step_loss = []
        for source_im in big_tensor:
            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) ))
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min()))  + source_im.min() 
            last_layer_loss, layer_loss_list = get_combined_cosine_loss_gcr(normalized_attacked, source_im)
            #print("len(layer_loss_list)", len(layer_loss_list))
            ret_noise_addition = noise_addition.clone().detach().requires_grad_(True)

            #chosen_space_ind = 25
            #print("layer_loss_list[chosen_space_ind]", layer_loss_list[chosen_space_ind])
            grads = torch.autograd.grad(outputs=layer_loss_list[chosen_space_ind]*-1, inputs=noise_addition, retain_graph=True, create_graph=False)[0]

            total_loss = last_layer_loss * -1 #* (1-balance) + alignment * balance
            total_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            fut_normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) ))
            fut_normalized_attacked = ( source_im.max() - source_im.min() ) * ((fut_normalized_attacked-fut_normalized_attacked.min())/(fut_normalized_attacked.max()-fut_normalized_attacked.min()))  + source_im.min()

            ret_normalized_attacked = (source_im + (ret_noise_addition * (desired_norm_l_inf / (torch.norm(ret_noise_addition, p=float('inf')))) ))
            ret_normalized_attacked = ( source_im.max() - source_im.min() ) * ((ret_normalized_attacked-ret_normalized_attacked.min())/(ret_normalized_attacked.max()-ret_normalized_attacked.min()))  + source_im.min()

            fut_normalized_attacked.clone().detach().requires_grad_(False) 

            fut_last_layer_loss, fut_layer_loss_list = get_combined_cosine_loss_gcr(ret_normalized_attacked, fut_normalized_attacked)
            sus_grads = torch.autograd.grad(outputs=fut_layer_loss_list[chosen_space_ind], inputs=ret_noise_addition, retain_graph=False, create_graph=False)[0]

            alignment = cos(sus_grads, grads)
            batch_ali.append(alignment.item())
            loss_ali.append(total_loss.item())
            layer_step_loss.append(layer_loss_list[chosen_space_ind].item())

            #print("batch_ali", batch_ali)
            #print("alignment", alignment)
            #print()
            del fut_last_layer_loss
            del fut_layer_loss_list
            torch.cuda.empty_cache()
        print("step", step)
        batch_ali_ar = np.array(batch_ali)
        loss_ali_ar = np.array(loss_ali)
        layer_loss_ali_ar = np.array(layer_step_loss)

        #print("batch_ali_ar.shape", batch_ali_ar.shape)
        #print("batch_ali_ar", batch_ali_ar)
        epoch_list.append(batch_ali_ar)
        loss_list.append(loss_ali_ar)
        sel_layer_loss_list.append(loss_ali_ar)

        epoch_array = np.array(epoch_list)
        loss_array = np.array(loss_list)
        sel_layer_loss_array = np.array(sel_layer_loss_list)

        print("epoch_array.shape", epoch_array.shape)
        print("loss_array.shape", loss_array.shape)
        print("sel_layer_loss_array.shape", sel_layer_loss_array.shape)
        #print("sel_layer_loss_array", sel_layer_loss_array)

        np.save("/data1/chethan/alma/diffae/opti_saves/"+attck_type+"_space_ind"+str(chosen_space_ind)+"_epoch_batch_alignment_array.npy", epoch_array)
        np.save("/data1/chethan/alma/diffae/opti_saves/"+attck_type+"_space_ind"+str(chosen_space_ind)+"loss_epoch_batch_array.npy", loss_array)
        np.save("/data1/chethan/alma/diffae/opti_saves/"+attck_type+"_space_ind"+str(chosen_space_ind)+"layer_loss_epoch_batch_array.npy", sel_layer_loss_array)

        if(step%50==0  and step!=0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                scaled_noise = noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)


                print("epoch_array.shape", epoch_array.shape)
                print("loss_array.shape", loss_array.shape)
                #print("loss_array", loss_array)
                '''np.save("/data1/chethan/alma/diffae/opti_saves/space_ind"+str(chosen_space_ind)+"_epoch_batch_alignment_array.npy", epoch_array)
                np.save("/data1/chethan/alma/diffae/opti_saves/space_ind"+str(chosen_space_ind)+"loss_epoch_batch_array.npy", loss_array)'''
    



balance = 0.3
if(attck_type == "gcr_cos_simp"):
    adv_div_list = []
    epoch_list = []
    loss_list = []
    sel_layer_loss_list = []
    for step in range(155):
        batch_ali = []
        loss_ali = []
        layer_step_loss = []
        for source_im in big_tensor:
            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) ))
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min()))  + source_im.min() 
            last_layer_loss, layer_loss_list = get_combined_cosine_loss_gcr_simp(normalized_attacked, source_im)
            #print("len(layer_loss_list)", len(layer_loss_list))
            ret_noise_addition = noise_addition.clone().detach().requires_grad_(True)

            #chosen_space_ind = 25
            #print("layer_loss_list[chosen_space_ind]", layer_loss_list[chosen_space_ind])
            grads = torch.autograd.grad(outputs=layer_loss_list[chosen_space_ind]*-1, inputs=noise_addition, retain_graph=True, create_graph=False)[0]

            total_loss = last_layer_loss * -1 #* (1-balance) + alignment * balance
            total_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            fut_normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) ))
            fut_normalized_attacked = ( source_im.max() - source_im.min() ) * ((fut_normalized_attacked-fut_normalized_attacked.min())/(fut_normalized_attacked.max()-fut_normalized_attacked.min()))  + source_im.min()

            ret_normalized_attacked = (source_im + (ret_noise_addition * (desired_norm_l_inf / (torch.norm(ret_noise_addition, p=float('inf')))) ))
            ret_normalized_attacked = ( source_im.max() - source_im.min() ) * ((ret_normalized_attacked-ret_normalized_attacked.min())/(ret_normalized_attacked.max()-ret_normalized_attacked.min()))  + source_im.min()

            fut_normalized_attacked.clone().detach().requires_grad_(False) 

            fut_last_layer_loss, fut_layer_loss_list = get_combined_cosine_loss_gcr(ret_normalized_attacked, fut_normalized_attacked)
            sus_grads = torch.autograd.grad(outputs=fut_layer_loss_list[chosen_space_ind], inputs=ret_noise_addition, retain_graph=False, create_graph=False)[0]

            alignment = cos(sus_grads, grads)
            batch_ali.append(alignment.item())
            loss_ali.append(total_loss.item())
            layer_step_loss.append(layer_loss_list[chosen_space_ind].item())

            #print("batch_ali", batch_ali)
            #print("alignment", alignment)
            #print()
            del fut_last_layer_loss
            del fut_layer_loss_list
            torch.cuda.empty_cache()
        print("step", step)
        batch_ali_ar = np.array(batch_ali)
        loss_ali_ar = np.array(loss_ali)
        layer_loss_ali_ar = np.array(layer_step_loss)

        #print("batch_ali_ar.shape", batch_ali_ar.shape)
        #print("batch_ali_ar", batch_ali_ar)
        epoch_list.append(batch_ali_ar)
        loss_list.append(loss_ali_ar)
        sel_layer_loss_list.append(loss_ali_ar)

        epoch_array = np.array(epoch_list)
        loss_array = np.array(loss_list)
        sel_layer_loss_array = np.array(sel_layer_loss_list)

        print("epoch_array.shape", epoch_array.shape)
        print("loss_array.shape", loss_array.shape)
        print("sel_layer_loss_array.shape", sel_layer_loss_array.shape)
        #print("sel_layer_loss_array", sel_layer_loss_array)

        np.save("/data1/chethan/alma/diffae/opti_saves/"+attck_type+"_space_ind"+str(chosen_space_ind)+"_epoch_batch_alignment_array.npy", epoch_array)
        np.save("/data1/chethan/alma/diffae/opti_saves/"+attck_type+"_space_ind"+str(chosen_space_ind)+"loss_epoch_batch_array.npy", loss_array)
        np.save("/data1/chethan/alma/diffae/opti_saves/"+attck_type+"_space_ind"+str(chosen_space_ind)+"layer_loss_epoch_batch_array.npy", sel_layer_loss_array)

        if(step%50==0  and step!=0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                scaled_noise = noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)


                print("epoch_array.shape", epoch_array.shape)
                print("loss_array.shape", loss_array.shape)
                #print("loss_array", loss_array)
                '''np.save("/data1/chethan/alma/diffae/opti_saves/space_ind"+str(chosen_space_ind)+"_epoch_batch_alignment_array.npy", epoch_array)
                np.save("/data1/chethan/alma/diffae/opti_saves/space_ind"+str(chosen_space_ind)+"loss_epoch_batch_array.npy", loss_array)'''





balance = 0.3
if(attck_type == "gcr_cos_simp2"):
    adv_div_list = []
    epoch_list = []
    loss_list = []
    sel_layer_loss_list = []
    for step in range(155):
        batch_ali = []
        loss_ali = []
        layer_step_loss = []
        for source_im in big_tensor:
            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) ))
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min()))  + source_im.min() 
            final_loss, retained_outputs = get_combined_cosine_loss_gcr_simp2(normalized_attacked, source_im)

            final_loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            for nx in retained_outputs:
                print(f"Gradient for layer : {nx.grad.shape} - mean grad: {nx.grad.abs().mean().item():.6f}")
                print("l2 norm : ", torch.norm(nx, 2))
                print("retained output shape : ", nx.shape)
                print()
        print("step", step)
        if(step%50==0 and not step==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)







balance = 0.3
if(attck_type == "gcr_cos_all_layers"):
    adv_div_list = []
    epoch_list = []
    loss_list = []
    for step in range(155):
        batch_ali = []
        loss_ali = []
        for source_im in big_tensor:
            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) ))
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min()))  + source_im.min() 
            last_layer_loss, layer_loss_list = get_combined_cosine_loss_gcr(normalized_attacked, source_im)
            ret_noise_addition = noise_addition.clone().detach().requires_grad_(True)

            chosen_space_ind = 0
            for chosen_space_ind in range(len(layer_loss_list)):
                grads = torch.autograd.grad(outputs=layer_loss_list[chosen_space_ind]*-1, inputs=noise_addition, retain_graph=True, create_graph=False)[0]

                total_loss = last_layer_loss * -1 #* (1-balance) + alignment * balance
                total_loss.backward(retain_graph=False)

                optimizer.step()
                optimizer.zero_grad()

                fut_normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) ))
                fut_normalized_attacked = ( source_im.max() - source_im.min() ) * ((fut_normalized_attacked-fut_normalized_attacked.min())/(fut_normalized_attacked.max()-fut_normalized_attacked.min()))  + source_im.min()

                ret_normalized_attacked = (source_im + (ret_noise_addition * (desired_norm_l_inf / (torch.norm(ret_noise_addition, p=float('inf')))) ))
                ret_normalized_attacked = ( source_im.max() - source_im.min() ) * ((ret_normalized_attacked-ret_normalized_attacked.min())/(ret_normalized_attacked.max()-ret_normalized_attacked.min()))  + source_im.min()

                fut_normalized_attacked.clone().detach().requires_grad_(False) 

                fut_last_layer_loss, fut_layer_loss_list = get_combined_cosine_loss_gcr(ret_normalized_attacked, fut_normalized_attacked)
                sus_grads = torch.autograd.grad(outputs=fut_layer_loss_list[chosen_space_ind], inputs=ret_noise_addition, retain_graph=False, create_graph=False)[0]

                alignment = cos(sus_grads, grads)
                batch_ali.append(alignment.item())
                loss_ali.append(total_loss.item())
                #print("batch_ali", batch_ali)
                print("alignment", alignment)
                #print()
                # del sus_grads
                # del grads
                del fut_last_layer_loss
                del fut_layer_loss_list
                torch.cuda.empty_cache()
        print("step", step)
        batch_ali_ar = np.array(batch_ali)
        loss_ali_ar = np.array(loss_ali)

        #print("batch_ali_ar.shape", batch_ali_ar.shape)
        #print("batch_ali_ar", batch_ali_ar)
        epoch_list.append(batch_ali_ar)
        loss_list.append(loss_ali_ar)

        epoch_array = np.array(epoch_list)
        loss_array = np.array(loss_list)

        print("epoch_array.shape", epoch_array.shape)
        print("loss_array.shape", loss_array.shape)

        if(step%20==0 and step!=0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                scaled_noise = noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)

                print("epoch_array.shape", epoch_array.shape)
                print("loss_array.shape", loss_array.shape)
                print("loss_array", loss_array)
                np.save("/data1/chethan/alma/diffae/opti_saves/"+attck_type+"space_ind"+str(chosen_space_ind)+"_epoch_batch_alignment_array.npy", epoch_array)
                np.save("/data1/chethan/alma/diffae/opti_saves/"+attck_type+"space_ind"+str(chosen_space_ind)+"loss_epoch_batch_array.npy", loss_array)





balance = 0.3
if(attck_type == "gcr_cos_use_converged"):
    adv_div_list = []
    epoch_list = []
    loss_list = []
    for step in range(155):
        batch_ali = []
        loss_ali = []
        for source_im in big_tensor:
            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) ))
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min()))  + source_im.min() 
            last_layer_loss, layer_loss_list = get_combined_cosine_loss_gcr(normalized_attacked, source_im)
            ret_noise_addition = noise_addition.clone().detach().requires_grad_(True)

            chosen_space_ind = 25
            grads = torch.autograd.grad(outputs=layer_loss_list[chosen_space_ind]*-1, inputs=noise_addition, retain_graph=True, create_graph=False)[0]

            total_loss = last_layer_loss * -1 #* (1-balance) + alignment * balance
            total_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            fut_normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) ))
            fut_normalized_attacked = ( source_im.max() - source_im.min() ) * ((fut_normalized_attacked-fut_normalized_attacked.min())/(fut_normalized_attacked.max()-fut_normalized_attacked.min()))  + source_im.min()

            ret_normalized_attacked = (source_im + (ret_noise_addition * (desired_norm_l_inf / (torch.norm(ret_noise_addition, p=float('inf')))) ))
            ret_normalized_attacked = ( source_im.max() - source_im.min() ) * ((ret_normalized_attacked-ret_normalized_attacked.min())/(ret_normalized_attacked.max()-ret_normalized_attacked.min()))  + source_im.min()

            fut_normalized_attacked.clone().detach().requires_grad_(False) 

            fut_last_layer_loss, fut_layer_loss_list = get_combined_cosine_loss_gcr(ret_normalized_attacked, fut_normalized_attacked)
            sus_grads = torch.autograd.grad(outputs=fut_layer_loss_list[chosen_space_ind], inputs=ret_noise_addition, retain_graph=False, create_graph=False)[0]

            alignment = cos(sus_grads, grads)
            batch_ali.append(alignment.item())
            loss_ali.append(total_loss.item())
            #print("batch_ali", batch_ali)
            #print("alignment", alignment)
            #print()
            del fut_last_layer_loss
            del fut_layer_loss_list
            torch.cuda.empty_cache()
        print("step", step)
        batch_ali_ar = np.array(batch_ali)
        loss_ali_ar = np.array(loss_ali)

        #print("batch_ali_ar.shape", batch_ali_ar.shape)
        #print("batch_ali_ar", batch_ali_ar)
        epoch_list.append(batch_ali_ar)
        loss_list.append(loss_ali_ar)

        epoch_array = np.array(epoch_list)
        loss_array = np.array(loss_list)

        print("epoch_array.shape", epoch_array.shape)
        print("loss_array.shape", loss_array.shape)

        if(step%20==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                scaled_noise = noise_addition * (desired_norm_l_inf / (torch.norm(noise_addition, p=float('inf')))) 
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)


                print("epoch_array.shape", epoch_array.shape)
                print("loss_array.shape", loss_array.shape)
                print("loss_array", loss_array)
                np.save("/data1/chethan/alma/diffae/opti_saves/"+attck_type+"space_ind"+str(chosen_space_ind)+"_epoch_batch_alignment_array.npy", epoch_array)
                np.save("/data1/chethan/alma/diffae/opti_saves/"+attck_type+"space_ind"+str(chosen_space_ind)+"loss_epoch_batch_array.npy", loss_array)


if(attck_type == "alma_l2"):
    adv_div_list = []
    for step in range(155):
        batch_step = 0
        for source_im in big_tensor:
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 

            loss_to_maximize = get_combined_l2_loss(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)

if(attck_type == "alma_wass"):
    adv_div_list = []
    for step in range(155):
        batch_step = 0
        for source_im in big_tensor:
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 
            loss_to_maximize = get_combined_wasserstein_loss(normalized_attacked, source_im)
            total_loss = -1 * loss_to_maximize
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "alma_skl"):
    adv_div_list = []
    for step in range(155):
        batch_step = 0
        for source_im in big_tensor:
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = ( source_im.max() - source_im.min() ) * ((attacked-attacked.min())/(attacked.max()-attacked.min()))  + source_im.min() 
            loss_to_maximize = get_combined_SKL_loss(normalized_attacked, source_im)
            total_loss = -1 * loss_to_maximize
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("step", step)
        if(step%50==0):
            with torch.no_grad():
                attacked_embed = model.encode(normalized_attacked.to(device))
                xT_ad = model.encode_stochastic(normalized_attacked.to(device), attacked_embed, T=250)
                adv_gen = model.render(xT_ad, attacked_embed, T=20)
                l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                l2_distortion = torch.norm(scaled_noise, p=2)
                deviation = torch.norm(adv_gen - source_im, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, normalized_attacked, scaled_noise, adv_gen)
