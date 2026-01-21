import torch
import torch.nn as nn
from model import AutoEncoder
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os
from torchvision import datasets, transforms


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
python nvae/nvae_all_qualitative_plots_comparision.py --data_directory data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint --uni_noise_path ../NVAE/attack_run_time_univ/attack_noise --desired_norm_l_inf 0.05


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
#desired_norm_l_inf = 0.04  # Worked very well

#attck_types = ["combi_l2", "combi_wasserstein", "combi_SKL", "combi_cos", "hlatent_l2", "hlatent_wasserstein", "hlatent_SKL", "hlatent_cos"]

attck_types = ["hlatent_l2", "hlatent_wasserstein", "hlatent_SKL", "hlatent_cos", "combi_l2_cond", "combi_wasserstein_cond", "combi_SKL_cond", "combi_cos_cond", ]


############# plotting for paper ######################

batch_size = 15
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


with torch.no_grad():

    row_one_ims = []
    row_two_ims = []
    #source_im = torch.load("/home/luser/NVAE/a_mixed_data/"+str(select_feature)+"_d/images.pt")[:50].unsqueeze(0).cuda()  
    select_feature_dd = all_features[0]

    for idx, (source_im, _) in enumerate(testLoader):
        source_im, _ = source_im.cuda(), _
        break
    del testLoader

    rec_logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps = model(source_im)
    reconstructed_output = model.decoder_output(rec_logits)
    rec_gen = reconstructed_output.sample()

    row_one_ims.append(source_im)
    row_two_ims.append(rec_gen)

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

        row_one_ims.append(normalized_attacked)
        row_two_ims.append(adv_gen)


        del optimized_noise, attacked, normalized_attacked, adv_logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps, reconstructed_output, adv_gen

    all_row_ims = row_one_ims + row_two_ims
    num_ims = len(all_row_ims)


column_labels = ["Original \nImage", "LA \nL-2", "LA \nWasserstein", "LA \nSKL", "LA \nCosine", "ALMA \nL-2", "ALMA \nWasserstein", "ALMA \nSKL", "ALMA \nCosine"]

for ch in range(15):

    with torch.no_grad():

        fig, axes = plt.subplots(2, num_ims//2, figsize=(45, 10), gridspec_kw={'wspace': 0.02, 'hspace': 0.02})  # 2 rows, 10 columns

        # Loop through axes and images
        #for ax, img in zip(axes.flat, all_row_ims):
        for idx, (ax, img) in enumerate(zip(axes.flat, all_row_ims)):
            ax.imshow(img[ch].permute(1, 2, 0).cpu().numpy())
            ax.axis('off')  # Hide axes
            # Add text only for the second row images
            if idx < num_ims // 2:  # If the index corresponds to the second row
                col_index = idx % (num_ims // 2)  # Get the column index
                ax.set_title(column_labels[col_index], fontsize=40, pad=10)  # Add text below the image

        #plt.subplots_adjust(wspace=0.0, hspace=0.0)  # Adjust horizontal and vertical space
        #plt.tight_layout()
        #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()
        plt.show()
        plt.savefig("all_universal_qualitative/paperInd_"+str(ch)+"_universal_NVAE_attacks_norm_bound_"+str(desired_norm_l_inf)+"feature_"+str(select_feature_dd)+"_source_segment_.png", bbox_inches='tight')
        plt.close()
        break


