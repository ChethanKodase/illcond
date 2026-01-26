
'''

conda deactivate
conda deactivate
cd illcond
conda activate /home/luser/anaconda3/envs/inn
python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttackOutputDistortionPlots.py


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

model_type = "TCVAE"
#model_type = "VAE"
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



#if model_type == "VAE":
#    attack_types = ["latent_l2_kf", "latent_wass_kf", "latent_cosine", "output_l2_kf", "output_wass_kf", "output_attack_cosine", "weighted_combi_k_eq_latent_l2", "weighted_combi_k_eq_latent_wasserstein", "weighted_combi_k_eq_latent_cosine", "grill_l2_kf", "grill_wass_kf", "grill_cos_kf"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]
#    #attack_types = ["latent_l2_kf", "latent_wass_kf", "latent_cosine", "output_l2_kf", "output_wass_kf", "output_cos_kf", "lgr_l2_kf", "lgr_wass_kf", "lgr_cos_kf", "grill_l2_kfNw", "grill_wass_kf", "grill_cos_kf"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]

#if model_type == "TCVAE":
#attack_types = ["latent_l2_kf", "latent_wass_kf", "latent_cosine", "output_l2_kf", "output_wass_kf", "output_cos_kf", "lgr_l2_kf", "lgr_wass_kf", "lgr_cos_kf", "grill_l2_kfNw", "grill_wass_kf", "grill_cos_kf"]
attack_types = ["latent_l2_kf", "latent_wass_kf", "latent_cosine", "output_l2_kf", "output_wass_kf", "output_attack_cosine", "weighted_combi_k_eq_latent_l2", "weighted_combi_k_eq_latent_wasserstein", "weighted_combi_k_eq_latent_cosine", "grill_l2_kf", "grill_wass_kf", "grill_cos_kf"] #, "acmld_sens_l2", "acmld_sens_l2_corr", "acmld_sens_l2_deco"]

all_perturbation_norms = [0.04, 0.05, 0.06, 0.07]



with torch.no_grad():


    all_mse_lists = []
    all_l2_dist_lists = []


    all_attack_all_l2_dists_per_perts_mean = []
    all_attack_all_l2_dists_per_perts_std = []
    for attack_type in attack_types:

        all_l2_dists_per_perts_mean = []
        all_l2_dists_per_perts_std = []

        for l_inf_bound in all_perturbation_norms:
            #for i in range(len(attack_types)):
            '''if l_inf_bound==0.07 and attack_type=="grill_l2_kf" and model_type=="TCVAE":
                attack_type = "grill_l2_kfNw"'''
            optimized_noise = torch.load("betaVAEandTcVAE/robustness_eval_saves_univ/relevancy_test/layerwise_maximum_damage_attack/"+model_type+"_beta_"+str(vae_beta_value)+"_attack_type"+str(attack_type)+"_norm_bound_"+str(l_inf_bound)+".pt").to(device) 
            #attacked = (source_im + optimized_noise)
            #normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())
            print("optimized_noise.shape", optimized_noise.shape)
            print("source_im.shape", source_im.shape)
            normalized_attacked = torch.clamp(source_im + optimized_noise, mi, ma)

            #normalized_attacked = torch.sigmoid(normalized_attacked) 


            adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
            #print("adv_gen.shape", adv_gen.shape)
            #print("normalized_attacked.shape)", normalized_attacked.shape)

            reconstruction_loss = F.mse_loss(normalized_attacked, adv_gen, reduction='none')  # Shape: [50, 3, 64, 64]

            reconstruction_loss_per_image = reconstruction_loss.mean(dim=[1, 2, 3])  # Shape: [50]

            l2_distance_per_image = torch.norm(normalized_attacked - adv_gen, p=2, dim=[1, 2, 3])  # Shape: [50]

            #print("attack_type", attack_type)
            #print("l_inf_bound", l_inf_bound)
            #print("l2_distance_per_image.shape", l2_distance_per_image.shape)
            l2_dist_mean = l2_distance_per_image.mean()
            #print("le_dist_mean", l2_dist_mean)
            l2_dist_standard_deviation = l2_distance_per_image.std()
            #print("l2_dist_standard_deviation", l2_dist_standard_deviation)
            #print()
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

objective_names = ["LA,l-2", "LA, wasserst.", "LA, cosine", "OA, l-2", "OA, wasserst.", "OA, cosine", "LGR, l-2", "LGR, wasserst.", "LGR, cosine", "GRILL, l-2", "GRILL, wasserst.", "GRILL, cosine"]

#objective_names = ["LA,l-2", "LA, wasserst.", "LA, SKL", "LA, cosine", "OA, l-2", "OA, wasserst.", "OA, SKL", "OA, cosine", "ALMA, l-2", "ALMA, wasserst.", "ALMA, SKL", "ALMA, cosine"]


# Simulated distributions (mean and standard deviation)
#mean_values = np.sin(2 * np.pi * epsilon)  # Some function to represent the mean
#std_dev = 0.2 + 0.1 * np.cos(2 * np.pi * epsilon)  # Changing spread

#plt.figure(figsize=(12, 8))  # Adjust the width and height as needed
plt.figure(figsize=(6, 5))  # Adjust the width and height as needed

'''color_list = [
    'blue', 'orange', 'green', 'red',
     'purple', 'cyan', 'magenta', 'yellow',
    'brown', 'pink', 'gray', 'olive',
     'lime', 'teal', 'indigo', 'gold'
]'''


color_list = [
    'blue', 'orange', 'red',
     'purple', 'cyan', 'yellow', 
    'brown', 'pink', 'olive', 
     'lime', 'teal', 'gold', 
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
plt.xlabel(r'$c$', fontsize=22)
plt.ylabel('L-2 distance', fontsize=22)
#plt.title("Distribution Change with Epsilon")
plt.grid(True)
#plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.yticks(fontsize=22)
plt.xticks(fontsize=22, rotation=45)
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
plt.savefig("betaVAEandTcVAE/perturbation_analysis/kf/new"+model_type+"_beta_"+str(vae_beta_value)+"_bp.png")
