
'''

conda deactivate
conda deactivate
cd illcond
conda activate /home/luser/anaconda3/envs/inn
python betaVAEandTcVAE/BetaVAETCVAEadaptiveAttacksOutputDistortionDistributionPlots.py


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

# 0, 2, 8, 32
segment = 32
which_gpu = 1

#l_inf_bound = 0.12

#l_inf_bound = 0.06
l_inf_bound = 0.05


vae_beta_value = 5.0

#model_type = "TCVAE"
model_type = "VAE"
device = ("cuda:"+str(which_gpu)+"" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training


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

batch_size = 2000
celeba_data = datasets.ImageFolder('betaVAEandTcVAE/data_cel1', transform=transform)

train_set, test_set = torch.utils.data.random_split(celeba_data, [int(len(img_list) * 0.8), len(img_list) - int(len(img_list) * 0.8)])
train_data_size = len(train_set)
test_data_size = len(test_set)
print('train_data_size', train_data_size)
print('test_data_size', test_data_size)
trainLoader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True, drop_last=True)
testLoader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

for idx, (image, label) in enumerate(testLoader):
    source_im, label = image.to(device), label.to(device)
    break

mi, ma = source_im.min(), source_im.max()


def get_hmc_lat1(z1, normalized_attacked):
    z = z1#.clone().detach().requires_grad_(True)  # Start point for MCMC
    x = normalized_attacked#.detach()              # Adversarial input
    #step_size = 0.008
    '''n_steps = 20
    leapfrog_steps = 10'''
    step_size = 0.1
    n_steps = 20
    leapfrog_steps = 10
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


#attack_types = ["output_l2_kf_mcmc", "grill_l2_kf_mcmc"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]

attack_types = ["output_l2_kf_mcmc", "grill_cos_kf_mcmc"]

#output_l2_kf

#attack_types = ["aclmd_l2f_cond", "aclmd_wasserstein_cond"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]


#all_perturbation_norms = [0.04, 0.05, 0.06, 0.07, 0.09]
#all_perturbation_norms = [0.04, 0.05, 0.06, 0.07]
#all_perturbation_norms = [0.07]
all_perturbation_norms = [0.04, 0.05, 0.06, 0.07]
#all_perturbation_norms = [0.03, 0.07]
#all_perturbation_norms = [0.06, 0.07]
#all_perturbation_norms = [ 0.07]
#all_perturbation_norms = [0.04, 0.05, 0.06]
#all_perturbation_norms = [0.04, 0.05]


#with torch.no_grad():
if(True):

    all_mse_lists = []
    all_l2_dist_lists = []


    all_attack_all_l2_dists_per_perts_mean = []
    all_attack_all_l2_dists_per_perts_std = []
    for attack_type in attack_types:

        all_l2_dists_per_perts_mean = []
        all_l2_dists_per_perts_std = []

        for l_inf_bound in all_perturbation_norms:
            #for i in range(len(attack_types)):
            '''if l_inf_bound==0.07 and attack_type=="grill_l2_kf":
                attack_type = "grill_l2_kfNw"
            if l_inf_bound==0.06 and attack_type=="grill_l2_kf":
                attack_type = "grill_l2_kfNw"
            if l_inf_bound==0.05 and attack_type=="grill_l2_kf":
                attack_type = "grill_l2_kfNw"'''
            optimized_noise = torch.load("betaVAEandTcVAE/robustness_eval_saves_univ/relevancy_test/layerwise_maximum_damage_attack/"+model_type+"_beta_"+str(vae_beta_value)+"_attack_type"+str(attack_type)+"_norm_bound_"+str(l_inf_bound)+".pt").to(device) 
            #attacked = (source_im + optimized_noise)
            #normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())
            print("optimized_noise.shape", optimized_noise.shape)
            print("source_im.shape", source_im.shape)
            normalized_attacked = torch.clamp(source_im + optimized_noise, mi, ma)

            #normalized_attacked = torch.sigmoid(normalized_attacked) 


            #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
            embedding = model.encoder(normalized_attacked)

            mu1, logvar1 = model.fc1(embedding), model.fc2(embedding)
            std1 = logvar1.mul(0.5).exp_()
            esp1 = torch.randn(*mu1.size()).to(device)
            z1 = mu1 + std1 * esp1
            z1 = get_hmc_lat1(z1, normalized_attacked)
            attack_flow = model.fc3(z1)
            adv_gen = model.decoder(attack_flow)
            #print("adv_gen.shape", adv_gen.shape)
            #print("normalized_attacked.shape)", normalized_attacked.shape)

            reconstruction_loss = F.mse_loss(normalized_attacked, adv_gen, reduction='none')  # Shape: [50, 3, 64, 64]

            reconstruction_loss_per_image = reconstruction_loss.mean(dim=[1, 2, 3])  # Shape: [50]

            l2_distance_per_image = torch.norm(normalized_attacked - adv_gen, p=2, dim=[1, 2, 3])  # Shape: [50]

            print("attack_type", attack_type)
            print("l_inf_bound", l_inf_bound)
            #print("l2_distance_per_image.shape", l2_distance_per_image.shape)
            l2_dist_mean = l2_distance_per_image.mean()
            print("le_dist_mean", l2_dist_mean)
            l2_dist_standard_deviation = l2_distance_per_image.std()
            print("l2_dist_standard_deviation", l2_dist_standard_deviation)
            print()
            with torch.no_grad():
                numpy_array = reconstruction_loss_per_image.cpu().numpy()

            all_l2_dists_per_perts_mean.append(l2_dist_mean.item())
            all_l2_dists_per_perts_std.append(l2_dist_standard_deviation.item())
            

        all_l2_dists_per_perts_mean = np.array(all_l2_dists_per_perts_mean)
        all_l2_dists_per_perts_std = np.array(all_l2_dists_per_perts_std)


        all_attack_all_l2_dists_per_perts_mean.append(all_l2_dists_per_perts_mean)
        all_attack_all_l2_dists_per_perts_std.append(all_l2_dists_per_perts_std)
        


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Simulated data: epsilon values
#epsilon = np.linspace(0.1, 1.0, 4)

epsilon = all_perturbation_norms



#objective_names = ["LA,l-2", "LA, wasserst.", "LA, SKL", "LA, cosine", "OA, l-2", "OA, wasserst.", "OA, SKL", "OA, cosine", "LMA, l-2", "LMA, wasserst.", "LMA, SKL", "LMA, cosine", "ALMA, l-2", "ALMA, wasserst.", "ALMA, SKL", "ALMA, cosine"]

#objective_names = ["LA,l-2", "LA, wasserst.", "LA, cosine", "OA, l-2", "OA, wasserst.", "OA, cosine", "LMA, l-2", "LMA, wasserst.", "LMA, cosine", "ALMA, l-2", "ALMA, wasserst.", "ALMA, cosine"]
objective_names = [ "OA, l-2", "ALMA, l-2"]

#objective_names = ["LA,l-2", "LA, wasserst.", "LA, SKL", "LA, cosine", "OA, l-2", "OA, wasserst.", "OA, SKL", "OA, cosine", "ALMA, l-2", "ALMA, wasserst.", "ALMA, SKL", "ALMA, cosine"]


# Simulated distributions (mean and standard deviation)
#mean_values = np.sin(2 * np.pi * epsilon)  # Some function to represent the mean
#std_dev = 0.2 + 0.1 * np.cos(2 * np.pi * epsilon)  # Changing spread

#plt.figure(figsize=(12, 8))  # Adjust the width and height as needed
plt.figure(figsize=(6, 7))  # Adjust the width and height as needed

'''color_list = [
    'blue', 'orange', 'green', 'red',
     'purple', 'cyan', 'magenta', 'yellow',
    'brown', 'pink', 'gray', 'olive',
     'lime', 'teal', 'indigo', 'gold'
]'''


color_list = [
     'purple','gold' 
]


for i in range(len(all_attack_all_l2_dists_per_perts_mean)):
#for i in [12, 13, 14, 15]:#, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15]:
    mean_values = all_attack_all_l2_dists_per_perts_mean[i]
    std_dev = all_attack_all_l2_dists_per_perts_std[i]

    print()
    print("epsilon", epsilon)
    print("objective_names[i]", objective_names[i])
    print("mean_values", mean_values)
    print("std_dev", std_dev)

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
#plt.title("Distribution Change with Epsilon")
plt.grid(True)
#plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.yticks(fontsize=28)
plt.xticks(fontsize=28, rotation=45)
#plt.legend()

# Get legend handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

# Increase line thickness in the legend
for handle in handles:
    handle.set_linewidth(4)

#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# Adjust layout to fit the legend
plt.tight_layout()


plt.show()
plt.savefig("betaVAEandTcVAE/perturbation_analysis/kf/mcmc_"+model_type+"_beta_"+str(vae_beta_value)+"_bp.png")
