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
cd NVAE/
export CUDA_VISIBLE_DEVICES=3
source nvaeenv1/bin/activate
python nvae/NvaeLayerFractionsAblation.py

'''



checkpoint_path = '../NVAE/pretrained_checkpoint/checkpoint.pt'
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


attck_types = ["grill_wass_kf_30pRev", "grill_wass_kf_50pRev", "grill_wass_kf_70pRev", "grill_wass_kf_90pRev", "grill_wass_kf_allSum", "grill_wass_kf" ]


attck_typesNames = ["GRILL-30%", "GRILL-50%", "GRILL-70%", "GRILL-90%", "Layer losses \nsumation", "GRILL-100%"]

img_list = os.listdir('../data_cel1/smile/')
img_list.extend(os.listdir('../data_cel1/no_smile/'))

transform = transforms.Compose([
          transforms.Resize((64, 64)),
          transforms.ToTensor()
          ])
batch_size = 1000
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



desired_norm_l_infs = [0.05]

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

    l2_all_methods = {}

    for i in range(len(attck_types)):
        
        mean_per_per_accum = []
        std_per_per_accum = []

        for desired_norm_l_inf in desired_norm_l_infs:

            optimized_noise = torch.load("nvae/univ_attack_storage/NVAE_attack_type"+str(attck_types[i])+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")
            

            normalized_attacked = torch.clamp(source_im + optimized_noise, mi, ma)

            normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            adv_logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps = model(normalized_attacked)
            reconstructed_output = model.decoder_output(adv_logits)
            adv_gen = reconstructed_output.sample()

            reconstruction_loss = F.mse_loss(normalized_attacked, adv_gen, reduction='none')  # Shape: [50, 3, 64, 64]

            reconstruction_loss_per_image = reconstruction_loss.mean(dim=[1, 2, 3])  # Shape: [50]

            l2_distance_per_image = torch.norm(normalized_attacked - adv_gen, p=2, dim=[1, 2, 3])  # Shape: [50]

            print("l2_distance_per_image.shape", l2_distance_per_image.shape)

            print()

            attack_name = attck_typesNames[i]

            l2_all_methods.setdefault(attack_name, [])
            l2_all_methods[attack_name].extend(
                l2_distance_per_image.detach().cpu().tolist()
            )


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 16
})

rows = []
for method, values in l2_all_methods.items():
    for v in values:
        rows.append({
            "Attack": method,
            "L2 distance": v
        })

df = pd.DataFrame(rows)

plt.figure(figsize=(6, 8))
sns.boxplot(data=df, x="Attack", y="L2 distance")
plt.xticks(rotation=90, ha="right")
plt.tight_layout()
plt.savefig("nvae/structured_ablation_plots/l2_boxplot_all_attacks.png", dpi=300)
plt.show()
