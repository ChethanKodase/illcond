
'''

conda deactivate
conda deactivate
cd illcond
conda activate /home/luser/anaconda3/envs/inn
python betaVAEandTcVAE/BetaVAEandTCVAEqualitativePlotsForUniversalClassicAndAdaptiveAttacks.py


pending things : Output attack 3 metrics : Output attack for VQ-VAE is not feasible because of problems with discrete latent space and gradient calculation issues
SKL combibations

'''


import numpy as np
from matplotlib import pyplot as plt
import torch
from vae import VAE_big, VAE_big_b
import random
import pandas as pd
import torch.nn.functional as F
import seaborn as sns

from torchvision import datasets, transforms
import os


adaptive_noise = True

defending = False

# 0, 2, 8, 32
segment = 32
which_gpu = 1

#l_inf_bound = 0.12

#l_inf_bound = 0.06
#l_inf_bound = 0.06


vae_beta_value = 5.0

model_type = "TCVAE"
#model_type = "VAE"
device = ("cuda:"+str(which_gpu)+"" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training

print("model_type", model_type)

#model = VAE_big_b(device, image_channels=3).to(device)

model = VAE_big(device, image_channels=3).to(device)


train_data_size = 162079
epochs = 199
#model.load_state_dict(torch.load('/home/luser/autoencoder_attacks/train_aautoencoders/saved_model/checkpoints/celebA_seeded_CNN_VAE'+str(vae_beta_value)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epochs)+'.torch'))

model.load_state_dict(torch.load('betaVAEandTcVAE/saved_celebA/checkpoints/celebA_CNN_'+model_type+''+str(vae_beta_value)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epochs)+'.torch'))

#model.load_state_dict(torch.load('/home/luser/autoencoder_attacks/saved_celebA/checkpoints/celebA_CNN_'+model_type+''+str(4.0)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epochs)+'.torch'))

#"/home/luser/autoencoder_attacks/saved_celebA/checkpoints/celebA_CNN_VAE5.0_big_trainSize162079_epochs199.torch"

#attack_types = ["latent_l2", "latent_wasserstein", "latent_SKL", "latent_cosine", "output_attack_l2", "output_attack_wasserstein", "output_attack_SKL", "weighted_combi_k_eq_latent_l2", "weighted_combi_k_eq_latent_wasserstein", "weighted_combi_k_eq_latent_SKL", "weighted_combi_l2_enco", "weighted_combinations_wasserstein", "weighted_combinations_SKL", "weighted_combi_cosine_enco", "weighted_combi_l2_full", "test1", "aclm_l2", "aclmd_l2", "aclmr_l2"]

#attack_types = ["latent_l2", "latent_wasserstein", "latent_SKL", "latent_cosine", "output_attack_l2", "output_attack_wasserstein", "output_attack_SKL", "output_attack_cosine", "weighted_combi_k_eq_latent_l2", "weighted_combi_k_eq_latent_wasserstein", "weighted_combi_k_eq_latent_SKL", "weighted_combi_k_eq_latent_cosine", "aclmd_l2f_cond", "aclmd_wasserstein_cond", "aclmd_SKL_cond", "aclmd_cosine_cond"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#root = 'data_faces/img_align_celeba'
#img_list = os.listdir(root)
#print(len(img_list))
     


#df = pd.read_csv("list_attr_celeba.csv")
#df = df[['image_id', 'Smiling']]



img_list = os.listdir('betaVAEandTcVAE/data_cel1/smile/')
img_list.extend(os.listdir('betaVAEandTcVAE/data_cel1/no_smile/'))


transform = transforms.Compose([
          transforms.Resize((64, 64)),
          transforms.ToTensor()
          ])

batch_size = 1
celeba_data = datasets.ImageFolder('betaVAEandTcVAE/data_cel1', transform=transform)

train_set, test_set = torch.utils.data.random_split(celeba_data, [int(len(img_list) * 0.8), len(img_list) - int(len(img_list) * 0.8)])
train_data_size = len(train_set)
test_data_size = len(test_set)
print('train_data_size', train_data_size)
print('test_data_size', test_data_size)
trainLoader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)
testLoader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

for idx, (image, label) in enumerate(testLoader):
    source_im, label = image.to(device), label.to(device)
    if(idx==2):   # 4 for betaVAE   , 2 for TCVAE ?
        break

mi, ma = source_im.min(), source_im.max()

#attack_types = ["latent_l2", "latent_wasserstein", "latent_SKL", "latent_cosine", "output_attack_l2", "output_attack_wasserstein", "output_attack_SKL", "output_attack_cosine", "weighted_combi_k_eq_latent_l2", "weighted_combi_k_eq_latent_wasserstein", "weighted_combi_k_eq_latent_SKL", "weighted_combi_k_eq_latent_cosine", "aclmd_l2a_cond", "aclmd_wasserstein_cond", "aclmd_SKL_cond", "aclmd_cosine_cond"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]
#attack_types = ["latent_l2", "latent_wasserstein", "latent_SKL", "latent_cosine", "output_attack_l2", "output_attack_wasserstein", "output_attack_SKL", "output_attack_cosine", "aclmd_l2f_cond", "aclmd_wasserstein_cond", "aclmd_SKL_cond", "aclmd_cosine_cond"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]

#attack_types = ["aclmd_l2f_cond", "aclmd_wasserstein_cond"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]


#all_perturbation_norms = [0.04, 0.05, 0.06, 0.07, 0.09]
#all_perturbation_norms = [0.04, 0.05, 0.06, 0.07]
#all_perturbation_norms = [0.07]

'''if adaptive_noise:
    if (model_type=="VAE"):
        attack_types = ["latent_l2_mcmc",  "aclmd_l2a_mcmc"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]
    if (model_type=="TCVAE"):
        attack_types = ["output_wass_mcmc",  "aclmd_l2a_mcmc"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]

else:
    if (model_type=="VAE"):
        attack_types = ["latent_l2",  "aclmd_l2a_cond"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]
    if (model_type=="TCVAE"):
        attack_types = ["output_attack_wasserstein",  "aclmd_l2a_cond"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]'''


if adaptive_noise:
    if (model_type=="VAE"):
        attack_types = ["output_l2_kf_mcmc",  "grill_cos_kf_mcmc"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]
    if (model_type=="TCVAE"):
        attack_types = ["output_l2_kf_mcmc",  "grill_cos_kf_mcmc"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]

else:
    if (model_type=="VAE"):
        attack_types = ["output_l2_kf",  "grill_cos_kf"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]
    if (model_type=="TCVAE"):
        attack_types = ["output_l2_kf",  "grill_cos_kf"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]

#attack_types = ["output_l2_kf_mcmc", "grill_cos_kf_mcmc"]

#attack_types = ["latent_l2", "latent_wasserstein", "latent_SKL", "latent_cosine", "output_attack_l2", "output_attack_wasserstein", "output_attack_SKL", "output_attack_cosine", "weighted_combi_k_eq_latent_l2", "weighted_combi_k_eq_latent_wasserstein", "weighted_combi_k_eq_latent_SKL", "weighted_combi_k_eq_latent_cosine", "aclmd_l2a_cond", "aclmd_wasserstein_cond", "aclmd_SKL_cond", "aclmd_cosine_cond"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]

all_perturbation_norms = [0.04]
#all_perturbation_norms = [0.04, 0.05]

def get_hmc_lat1(z1, normalized_attacked):
    z = z1#.clone().detach().requires_grad_(True)  # Start point for MCMC
    x = normalized_attacked#.detach()              # Adversarial input
    
    #normal 
    #step_size = 0.008
    #n_steps = 50
    #leapfrog_steps = 30

    # when required
    step_size = 0.0008
    n_steps = 50
    leapfrog_steps = 30

    #samples = []
    for i in range(n_steps):
        p = torch.randn_like(z)  # Sample momentum
        z_new = z.clone()
        p_new = p.clone()
        x_mean = model.decoder(model.fc3(z_new))
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
            x_mean = model.decoder(model.fc3(z_new))
            x_flat, x_mean_flat = x.view(x.size(0), -1), x_mean.view(x.size(0), -1)
            log_p_x = -((x_flat - x_mean_flat) ** 2).sum(dim=1) / 2
            log_p_z = -0.5 * (z_new ** 2).sum(dim=1)
            log_post = (log_p_x + log_p_z).sum()
            grad = torch.autograd.grad(log_post, z_new)[0]
            p_new = p_new + step_size * grad
        p_new = p_new + 0.5 * step_size * grad
        p_new = -p_new  # Make symmetric

        #with torch.no_grad():
        logp_current = -0.5 * (z ** 2).sum(dim=1) - ((x.view(x.size(0), -1) - model.decoder(model.fc3(z)).view(x.size(0), -1)) ** 2).sum(dim=1) / 2
        logp_new = -0.5 * (z_new ** 2).sum(dim=1) - ((x.view(x.size(0), -1) - model.decoder(model.fc3(z_new)).view(x.size(0), -1)) ** 2).sum(dim=1) / 2
        
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

def get_mcmc_defended_recon(normalized_attacked, source_im):
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        #encoder_lip_sum += criterion(attack_out, source_out) #*cond_normal[l_ct]
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    if defending:
        z1 = get_hmc_lat1(z1, normalized_attacked)


    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    #loss_to_maximize =  criterion(z1, z2) 

    if defending:
        z2 = get_hmc_lat1(z2, source_im)


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    #decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        #decoder_lip_sum += criterion(attack_out, source_out) #*cond_normal[l_ct]
        attack_flow = attack_out
        source_flow = source_out

    #loss_to_maximize = 1.0

    return attack_flow, source_flow


#with torch.no_grad():


all_mse_lists = []
all_l2_dist_lists = []


all_attack_all_l2_dists_per_perts_mean = []
all_attack_all_l2_dists_per_perts_std = []
for attack_type in attack_types:

    all_l2_dists_per_perts_mean = []
    all_l2_dists_per_perts_std = []

    for l_inf_bound in all_perturbation_norms:
        #for i in range(len(attack_types)):
        optimized_noise = torch.load("betaVAEandTcVAE/robustness_eval_saves_univ/relevancy_test/layerwise_maximum_damage_attack/"+model_type+"_beta_"+str(vae_beta_value)+"_attack_type"+str(attack_type)+"_norm_bound_"+str(l_inf_bound)+".pt").to(device) 
        #attacked = (source_im + optimized_noise)
        #normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        normalized_attacked = torch.clamp(source_im + optimized_noise, mi, ma)


        #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
        adv_gen, _ = get_mcmc_defended_recon(normalized_attacked, source_im)
        print("adv_gen.shape", adv_gen.shape)
        #print("normalized_attacked.shape)", normalized_attacked.shape) '/home/luser/autoencoder_attacks/all_adaptive_classic_qualitative

        source_recon, _ = get_mcmc_defended_recon(source_im, source_im)


        plt.imshow(source_im[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.show()
        plt.savefig("betaVAEandTcVAE/all_adaptive_classic_qualitative/source_im"+model_type+"_beta_"+str(vae_beta_value)+"_attack_type"+str(attack_type)+"_norm_bound_"+str(l_inf_bound)+".png")
        plt.close()

        plt.imshow(normalized_attacked[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.show()
        plt.savefig("betaVAEandTcVAE/all_adaptive_classic_qualitative/normalized_attacked"+model_type+"_beta_"+str(vae_beta_value)+"_attack_type"+str(attack_type)+"_norm_bound_"+str(l_inf_bound)+".png")
        plt.close()

        plt.imshow(adv_gen[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.show()
        plt.savefig("betaVAEandTcVAE/all_adaptive_classic_qualitative/adv_gen"+model_type+"_beta_"+str(vae_beta_value)+"_attack_type"+str(attack_type)+"_norm_bound_"+str(l_inf_bound)+".png")
        plt.close()

        plt.imshow(source_recon[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.show()
        plt.savefig("betaVAEandTcVAE/all_adaptive_classic_qualitative/source_recon"+model_type+"_beta_"+str(vae_beta_value)+"_attack_type"+str(attack_type)+"_norm_bound_"+str(l_inf_bound)+".png")
        plt.close()

