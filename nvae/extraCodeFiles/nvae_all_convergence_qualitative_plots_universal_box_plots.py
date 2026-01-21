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
export CUDA_VISIBLE_DEVICES=3
cd NVAE/
source nvaeenv1/bin/activate
cd ..
cd alma/
python nvae/nvae_all_convergence_qualitative_plots_universal_box_plots.py --data_directory data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint --uni_noise_path ../NVAE/attack_run_time_univ/attack_noise --desired_norm_l_inf 0.05


'''



import argparse

parser = argparse.ArgumentParser(description='Adversarial attack on VAE')

parser.add_argument('--data_directory', type=str, default=0, help='data directory')
parser.add_argument('--nvae_checkpoint_path', type=str, default=0, help='nvae checkpoint directory')
parser.add_argument('--uni_noise_path', type=str, default=0, help='nvae checkpoint directory')
parser.add_argument('--desired_norm_l_inf', type=float, default=0, help='L-inf norm bound')


args = parser.parse_args()

data_directory = args.data_directory
nvae_checkpoint_path = args.nvae_checkpoint_path
uni_noise_path = args.uni_noise_path
desired_norm_l_inf = args.desired_norm_l_inf


all_features = ["youngmen", "oldmen", "youngwomen", "oldwomen" ]

feature_no = 2
source_segment = 0
select_feature = all_features[feature_no]


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

#attck_types = ["combi_l2", "combi_wasserstein", "combi_SKL", "combi_cos", "hlatent_l2", "hlatent_wasserstein", "hlatent_SKL", "hlatent_cos"]

#attck_types = ["hlatent_l2", "hlatent_wasserstein", "hlatent_SKL", "hlatent_cos", "combi_l2_cond", "combi_wasserstein_cond", "combi_SKL_cond", "combi_cos_cond", ]   ########### attack names I used

attck_types = ["la_l2", "la_wass", "la_skl", "la_cos", "grill_l2", "grill_wass", "grill_skl", "grill_cos"]   ########### attack names 



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
testLoader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
del trainLoader




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

    for i in range(len(attck_types)):

        optimized_noise = torch.load(""+uni_noise_path+"/NVAE_attack_type"+str(attck_types[i])+"_norm_bound_"+str(desired_norm_l_inf)+"feature_"+str(select_feature)+"_source_segment_"+str(source_segment)+"_.pt")

        print("before optimized_noise.max(), optimized_noise.min()", optimized_noise.max(), optimized_noise.min())
        #optimized_noise =optimized_noise.clamp(-0.032, 0.032)
        #optimized_noise =optimized_noise * 0.09

        print("after optimized_noise.max(), optimized_noise.min()", optimized_noise.max(), optimized_noise.min())
        attacked = (source_im + optimized_noise)
        normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())
        print("normalized_attacked.shape", normalized_attacked.shape)
        adv_logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps = model(normalized_attacked)
        reconstructed_output = model.decoder_output(adv_logits)
        adv_gen = reconstructed_output.sample()

        print("normalized_attacked.shape", normalized_attacked.shape)
        print("adv_gen.shape", adv_gen.shape)

        reconstruction_loss = F.mse_loss(normalized_attacked, adv_gen, reduction='none')  # Shape: [50, 3, 64, 64]

        reconstruction_loss_per_image = reconstruction_loss.mean(dim=[1, 2, 3])  # Shape: [50]

        l2_distance_per_image = torch.norm(normalized_attacked - adv_gen, p=2, dim=[1, 2, 3])  # Shape: [50]

        # Compute total average L2 loss across all images
        #total_l2_loss = l2_distance_per_image.mean()


        print("reconstruction_loss_per_image", reconstruction_loss_per_image)

        numpy_array = reconstruction_loss_per_image.cpu().numpy()
        all_mse_lists.append(numpy_array)
        all_l2_dist_lists.append(l2_distance_per_image.cpu().numpy())



    data = pd.DataFrame({
        "LA,\nl-2": all_mse_lists[0],
        "LA, \n wasserstein": all_mse_lists[1],
        "LA, \nSKL": all_mse_lists[2],
        "LA, \ncosine": all_mse_lists[3],
        "GRILL, \nl-2": all_mse_lists[4],
        "GRILL, \nwasserstein": all_mse_lists[5],
        "GRILL, \nSKL": all_mse_lists[6],
        "GRILL, \ncosine": all_mse_lists[7],
    })

    '''data = pd.DataFrame({
        "latent,l2": ar0,
        "latent, wasserstein": ar1,
        "latent, SKL": ar2,
        "latent, cosine": ar3
    })'''

    # Define colors for the boxplots
    #colors = ['blue', 'orange', 'green', 'red', 'purple', 'cyan', 'magenta', 'yellow']


    colors = ['blue', 'orange', 'green', 'red', 'lime', 'teal', 'indigo', 'gold']

    #colors = ['blue', 'orange', 'green', 'red']


    plt.figure(figsize=(12, 8))  # Adjust the width and height as needed

    sns.boxplot(data=data, palette=colors)
    #plt.xlabel("Feature", fontsize=14)  # Adjust the fontsize as needed
    plt.ylabel(r"MSE", fontsize=16)  # Using LaTeX formatting for the ylabel
    #plt.ylabel(r"$||x_a - x_a'||_2$", fontsize=12)  # Using LaTeX formatting for the ylabel
    #plt.ylabel(r"$||x_a - x_a'||_2$", fontsize=12)  # Using LaTeX formatting for the ylabel
    #plt.ylabel(r"$\frac{1}{n} \sum (x_a - x_a')^2$", fontsize=14)  # Using LaTeX formatting
    #plt.ylabel(r"$\mathrm{MSE}(x_a, x_a')$", fontsize=14)  # Using LaTeX formatting

    # Increase font size of xticks and yticks
    plt.xticks(fontsize=12, rotation=45)  # Adjust the fontsize as needed
    #plt.yticks(np.arange(200, 400, 5), fontsize=8)  # Adjust the fontsize as needed

    print(data.min().min(), data.max().max())
    y_min, y_max = data.min().min(), data.max().max()
    plt.yticks(np.arange(y_min, y_max + 0.01, (y_max - y_min) / 5), fontsize=16)

    plt.tight_layout()  # Adjust layout to prevent cutoff of labels

    plt.savefig("box_plots/NVAE_norm_bound_"+str(desired_norm_l_inf)+"_.png")
    plt.show()



    data = pd.DataFrame({
        "LA,\nl-2": all_l2_dist_lists[0],
        "LA, \n wasserstein": all_l2_dist_lists[1],
        "LA, \nSKL": all_l2_dist_lists[2],
        "LA, \ncosine": all_l2_dist_lists[3],
        "GRILL, \nl-2": all_l2_dist_lists[4],
        "GRILL, \nwasserstein": all_l2_dist_lists[5],
        "GRILL, \nSKL": all_l2_dist_lists[6],
        "GRILL, \ncosine": all_l2_dist_lists[7],
    })

    '''data = pd.DataFrame({
        "latent,l2": ar0,
        "latent, wasserstein": ar1,
        "latent, SKL": ar2,
        "latent, cosine": ar3
    })'''

    # Define colors for the boxplots
    colors = ['blue', 'orange', 'green', 'red', 'lime', 'teal', 'indigo', 'gold']


    #colors = ['blue', 'orange', 'green', 'red']


    plt.figure(figsize=(12, 8))  # Adjust the width and height as needed

    sns.boxplot(data=data, palette=colors)
    #plt.xlabel("Feature", fontsize=14)  # Adjust the fontsize as needed
    plt.ylabel(r"L-2 distance", fontsize=24)  # Using LaTeX formatting for the ylabel
    #plt.ylabel(r"$||x_a - x_a'||_2$", fontsize=12)  # Using LaTeX formatting for the ylabel
    #plt.ylabel(r"$||x_a - x_a'||_2$", fontsize=12)  # Using LaTeX formatting for the ylabel
    #plt.ylabel(r"$\frac{1}{n} \sum (x_a - x_a')^2$", fontsize=14)  # Using LaTeX formatting
    #plt.ylabel(r"$\mathrm{MSE}(x_a, x_a')$", fontsize=14)  # Using LaTeX formatting

    # Increase font size of xticks and yticks
    #plt.xticks(fontsize=24, rotation=45)  # Adjust the fontsize as needed
    #plt.yticks(np.arange(200, 400, 5), fontsize=8)  # Adjust the fontsize as needed

    print(data.min().min(), data.max().max())
    y_min, y_max = data.min().min(), data.max().max()
    plt.yticks(np.arange(y_min, y_max + 0.01, (y_max - y_min) / 5), fontsize=24)

    plt.tight_layout()  # Adjust layout to prevent cutoff of labels

    plt.savefig("box_plots/NVAE_l2_norm_bound_"+str(desired_norm_l_inf)+"_.png")
    plt.show()
