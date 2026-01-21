import torch
import torch.nn as nn
from model import AutoEncoder
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

conda deactivate
conda deactivate
conda deactivate
export CUDA_VISIBLE_DEVICES=2
cd NVAE/
source nvaeenv1/bin/activate
cd ..
cd alma/
python nvae/grill_nvae_all_convergence_epsilon_variation_mcmc.py --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint --uni_noise_path ../NVAE/attack_run_time_univ/attack_noise


'''


import argparse

parser = argparse.ArgumentParser(description='Adversarial attack on VAE')

parser.add_argument('--data_directory', type=str, default=0, help='data directory')
parser.add_argument('--nvae_checkpoint_path', type=str, default=0, help='nvae checkpoint directory')
parser.add_argument('--uni_noise_path', type=str, default=0, help='nvae checkpoint directory')


args = parser.parse_args()

data_directory = args.data_directory
nvae_checkpoint_path = args.nvae_checkpoint_path
uni_noise_path = args.uni_noise_path


#which_gpu = args.which_gpu

#all_features = ["bald", "beard", "oldfemaleGlass", "hat", "blackWomen", "generalWhiteWomen", "blackMen", "generalWhiteMen", "men", "women", "young", "old", "youngmen", "oldmen", "youngwomen", "oldwomen", "oldblackmen", "oldblackwomen", "oldwhitemen", "oldwhitewomen", "youndblackmen", "youndblackwomen", "youngwhitemen", "youngwhitewomen" ]

all_features = ["youngmen", "oldmen", "youngwomen", "oldwomen" ]

feature_no = 2
source_segment = 0
select_feature = all_features[feature_no]



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

xts = []
for i in range(100):
    xts.append(i*10000)
desired_norm_l_inf = 0.03  # Worked very well

#attck_types = ["combi_l2", "combi_wasserstein", "combi_SKL", "combi_cos", "hlatent_l2", "hlatent_wasserstein", "hlatent_SKL", "hlatent_cos"]

attck_types = ["hlatent_l2", "hlatent_wasserstein", "hlatent_SKL", "hlatent_cos", "combi_l2_cond", "combi_wasserstein_cond", "combi_SKL_cond", "combi_cos_cond", ]  # names I used 


attck_types = ["la_l2_kf", "la_wass_kf_cr", "hlatent_SKL", "la_cos_kf_cr", "grill_l2_kf", "grill_wass_kf", "combi_SKL_cond", "grill_cos_kf", ]  # names I used 


#attck_types = ["la_l2", "la_wass", "la_skl", "la_cos", "grill_l2", "grill_wass", "grill_skl", "grill_cos"]   ########### attack names 


#root = '/home/luser/autoencoder_attacks/train_aautoencoders/data_faces/img_align_celeba'
#img_list = os.listdir(root)
#print(len(img_list))
     
#df = pd.read_csv("/home/luser/autoencoder_attacks/train_aautoencoders/list_attr_celeba.csv")
#df = df[['image_id', 'Smiling']]

batch_size = 400
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

del trainLoader



desired_norm_l_infs = [0.03, 0.035, 0.04, 0.05]
############# plotting for paper ######################
with torch.no_grad():

    row_one_ims = []
    row_two_ims = []

    for idx, (source_im, _) in enumerate(testLoader):
        source_im, _ = source_im.cuda(), _
        break
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

            optimized_noise = torch.load(""+uni_noise_path+"/NVAE_attack_type"+str(attck_types[i])+"_norm_bound_"+str(desired_norm_l_inf)+"feature_"+str(select_feature)+"_source_segment_"+str(source_segment)+"_.pt")

            #print("before optimized_noise.max(), optimized_noise.min()", optimized_noise.max(), optimized_noise.min())
            #optimized_noise =optimized_noise.clamp(-0.032, 0.032)
            #optimized_noise =optimized_noise * 0.09

            #print("after optimized_noise.max(), optimized_noise.min()", optimized_noise.max(), optimized_noise.min())
            attacked = (source_im + optimized_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())
            #print("normalized_attacked.shape", normalized_attacked.shape)
            adv_logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps = model(normalized_attacked)
            reconstructed_output = model.decoder_output(adv_logits)
            adv_gen = reconstructed_output.sample()

            #print("normalized_attacked.shape", normalized_attacked.shape)
            #print("adv_gen.shape", adv_gen.shape)

            reconstruction_loss = F.mse_loss(normalized_attacked, adv_gen, reduction='none')  # Shape: [50, 3, 64, 64]

            reconstruction_loss_per_image = reconstruction_loss.mean(dim=[1, 2, 3])  # Shape: [50]

            l2_distance_per_image = torch.norm(normalized_attacked - adv_gen, p=2, dim=[1, 2, 3])  # Shape: [50]

            # Compute total average L2 loss across all images
            l2_distance_per_image_mean = l2_distance_per_image.mean().item()
            l2_distance_per_image_std = l2_distance_per_image.std().item()
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



objective_names = ["LA,l-2", "LA, wasserst.", "LA, SKL", "LA, cosine", "GRILL, l-2", "GRILL, wasserst.", "GRILL, SKL", "GRILL, cosine"]



# Simulated distributions (mean and standard deviation)
#mean_values = np.sin(2 * np.pi * epsilon)  # Some function to represent the mean
#std_dev = 0.2 + 0.1 * np.cos(2 * np.pi * epsilon)  # Changing spread

'''color_list = [
    'blue', 'orange', 'green', 'red', 'purple', 'cyan', 'magenta', 'yellow',
    'brown', 'pink', 'gray', 'olive', 'lime', 'teal', 'indigo', 'gold'
]'''


color_list = ['blue', 'orange', 'green', 'red', 'lime', 'teal', 'indigo', 'gold']

plt.figure(figsize=(12, 8))  # Adjust the width and height as needed


for i in range(len(all_method_means)):
#for i in [12, 13, 14, 15]:#, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15]:
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
plt.xlabel(r'$c$', fontsize=24)
plt.ylabel('L-2 distance', fontsize=24)
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

#plt.xticks(rotation=45, fontsize=24)
plt.xticks(rotation=45, fontsize=24)

plt.yticks(fontsize=24)
#plt.title("Distribution Change with Epsilon")
plt.grid(True)
#plt.legend()
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# Adjust layout to fit the legend
plt.tight_layout()


plt.show()
plt.savefig("damage_distributions_variation/NVAE_mcmc.png")
