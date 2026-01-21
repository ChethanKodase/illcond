import torch
import torch.nn as nn
from model_with_defense import AutoEncoder
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
from torchvision import datasets, transforms
import os

#device = "cuda:1" if torch.cuda.is_available() else "cpu"


'''

cd illcond
python nvae/NvaeDeviationsDistributionsAcrossPerturbationsForAdaptiveAttacks.py

'''




# Replace the placeholder values with your actual checkpoint path and parameters
checkpoint_path = '../NVAE/pretrained_checkpoint/checkpoint.pt'


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

xts = []
for i in range(100):
    xts.append(i*10000)
desired_norm_l_inf = 0.03  # Worked very well

attck_types = ["la_l2_kf_mcmc", "grill_wass_kf_mcmc"]


img_list = os.listdir('../data_cel1/smile/')
img_list.extend(os.listdir('../data_cel1/no_smile/'))

transform = transforms.Compose([
          transforms.Resize((64, 64)),
          transforms.ToTensor()
          ])
batch_size = 50
celeba_data = datasets.ImageFolder('../data_cel1', transform=transform)


split_train_frac = 0.95

train_set, test_set = torch.utils.data.random_split(celeba_data, [int(len(img_list) * split_train_frac), len(img_list) - int(len(img_list) * split_train_frac)])
train_data_size = len(train_set)
test_data_size = len(test_set)

print('train_data_size', train_data_size)
print('test_data_size', test_data_size)

trainLoader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)
testLoader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
del trainLoader



desired_norm_l_infs = [0.025, 0.035, 0.037, 0.04, 0.05]

############# plotting for paper ######################
with torch.no_grad():

    row_one_ims = []
    row_two_ims = []

    for idx, (source_im, _) in enumerate(testLoader):
        source_im, _ = source_im.cuda(), _
        break
    mi, ma = source_im.min(), source_im.max()

    del testLoader

    rec_logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps = model(source_im)
    reconstructed_output = model.decoder_output(rec_logits)
    rec_gen = reconstructed_output.sample()

    row_one_ims.append(source_im)
    row_two_ims.append(rec_gen)
    all_mse_lists = []
    all_l2_dist_lists = []

    all_method_means = []
    all_method_stds = []
    for i in range(len(attck_types)):
        
        mean_per_per_accum = []
        std_per_per_accum = []

        for desired_norm_l_inf in desired_norm_l_infs:

            optimized_noise = torch.load("nvae/univ_attack_storage/NVAE_attack_type"+str(attck_types[i])+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")
            
            print("source_im.shape", source_im.shape)
            print("optimized_noise.shape", optimized_noise.shape)
            normalized_attacked = torch.clamp(source_im + optimized_noise, mi, ma)
            print("1  normalized_attacked.min(), normalized_attacked.max()", normalized_attacked.min(), normalized_attacked.max())
            normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())
            #normalized_attacked = torch.sigmoid(normalized_attacked) 
            


            print("2  normalized_attacked.min(), normalized_attacked.max()", normalized_attacked.min(), normalized_attacked.max())
            print()
            #print("normalized_attacked.shape", normalized_attacked.shape)
            adv_logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps = model(normalized_attacked)
            reconstructed_output = model.decoder_output(adv_logits)
            adv_gen = reconstructed_output.sample()



            reconstruction_loss = F.mse_loss(normalized_attacked, adv_gen, reduction='none')  # Shape: [50, 3, 64, 64]

            reconstruction_loss_per_image = reconstruction_loss.mean(dim=[1, 2, 3])  # Shape: [50]

            l2_distance_per_image = torch.norm(normalized_attacked - adv_gen, p=2, dim=[1, 2, 3])  # Shape: [50]

            # Compute total average L2 loss across all images
            l2_distance_per_image_mean = l2_distance_per_image.mean().item()
            l2_distance_per_image_std = l2_distance_per_image.std().item()
            print("i", i)
            print("attck_types", attck_types)
            print("attck_types[i]", attck_types[i])
            print("desired_norm_l_inf", desired_norm_l_inf)
            print("l2_distance_per_image_mean", l2_distance_per_image_mean)
            print("l2_distance_per_image_std", l2_distance_per_image_std)


            mean_per_per_accum.append(l2_distance_per_image_mean)
            std_per_per_accum.append(l2_distance_per_image_std)

        mean_per_per_accum = np.array(mean_per_per_accum)
        std_per_per_accum = np.array(std_per_per_accum)

        all_method_means.append(mean_per_per_accum)
        all_method_stds.append(std_per_per_accum)



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Simulated data: epsilon values
#epsilon = np.linspace(0.1, 1.0, 4)

epsilon = desired_norm_l_infs


objective_names = ["LA,l-2", "LA, wasserst.", "LA, cosine", "ALMA, l-2", "ALMA, wasserst.", "ALMA, cosine"]


color_list = ['blue','teal']


plt.figure(figsize=(6, 7))  # Adjust the width and height as needed


for i in range(len(all_method_means)):
    mean_values = all_method_means[i]
    std_dev = all_method_stds[i]


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
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

#plt.xticks(rotation=45, fontsize=22)
plt.xticks(rotation=45, fontsize=28)

plt.yticks(fontsize=28)
#plt.title("Distribution Change with Epsilon")
plt.grid(True)
#plt.legend()
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# Adjust layout to fit the legend

handles, labels = plt.gca().get_legend_handles_labels()

# Increase line thickness in the legend
for handle in handles:
    handle.set_linewidth(4)

plt.tight_layout()


plt.show()
plt.savefig("nvae/grill_perturbation_analysis/AdaptiveAttackDamageDistributions.png")
