
'''

cd alma
conda activate dt2
python diffae/DiffAEOutPutDistortionDistributionPlotsForUniversalAdaptiveAttacks.py --epsilon_list 0.21 0.24 0.25 0.30 0.33 



'''


import numpy as np
from matplotlib import pyplot as plt
import torch
from templates import *


from matplotlib.ticker import FuncFormatter



import argparse

parser = argparse.ArgumentParser(description='DiffAE celebA training')


parser.add_argument("--epsilon_list", type=float, nargs='+', required=True, help="List of epsilon values")


args = parser.parse_args()
epsilon_list = args.epsilon_list


which_gpu = 7
source_segment = 0

#device = 'cuda:'+str(which_gpu)+''

def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()

xts = []
for i in range(100):
    xts.append(i*10000)

#attack_types = ["latent_l2", "latent_wasserstein", "latent_SKL", "latent_cosine", "combi_l2", "combi_wasserstein", "combi_SKL", "combi_cos"]
#attack_types = ["la_cos_kfAdamNoScheduler1", "latent_wasserstein", "latent_SKL", "latent_cosine", "combi_l2", "combi_wasserstein", "combi_SKL", "combi_cos_cond_dir_cap"]
attack_types = ["la_l2_kfAdamNoScheduler1", "la_wass_kfAdamNoScheduler1", "latent_SKL", "la_cos_kfAdamNoScheduler1_mcmc", "grill_l2_kfAdamNoScheduler1", "grill_wass_kfAdamNoScheduler1", "combi_SKL", "grill_cos_kfAdamNoScheduler1_mcmc"]
#attack_types = ["la_l2_kfAdamNoScheduler1", "la_wass_kfAdamNoScheduler1", "la_cos_kfAdamNoScheduler1", "grill_l2_kfAdamNoScheduler1", "grill_wass_kfAdamNoScheduler1", "grill_cos_kfAdamNoScheduler1"]

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

all_metric_types = ["adv_recons", "adv_divs", "adv_divs_wass", "adv_divs_abs", "adv_divs_wass", "ssim", "psnr"]

objective_names = ["LA,l-2", "LA, wasserst.", "LA, SKL", "LA, cosine", "GRILL, l-2", "GRILL, wasserst.", "GRILL, SKL", "GRILL, cosine"]

#objective_names = ["LA, cosine",  "GRILL, cosine"]


metric_type = all_metric_types[1]

#desired_norm_l_inf = 0.33

#all_l_inf_norms = [0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33]

all_l_inf_norms = epsilon_list

#all_l_inf_norms = [0.27, 0.31]sss


#considered_attack_inds = [0, 1, 2, 3, 4, 5, 6, 7]

#considered_attack_inds = [0, 1, 3, 4, 5, 7]
considered_attack_inds = [3, 7]
mean_per_methd = []
std_per_methd = []

for i in considered_attack_inds:
    per_ep_means = []
    per_ep_std_div = []
    print("attack_types[i]", attack_types[i])
    for desired_norm_l_inf in all_l_inf_norms:
        ar0 = np.load("diffae/attack_qualitative_untargeted_univ_quantitative/deviations_p/"+metric_type+"_DiffAE_attack_type"+str(attack_types[i])+"_norm_bound_"+str(desired_norm_l_inf)+"_segment_"+str(source_segment)+".npy", allow_pickle=True)
        colors = ['blue', 'orange', 'red', 'lime', 'teal', 'gold']

        #print("ar0",ar0)
        ar0_mean  = np.mean(ar0)
        ar0_std = np.std(ar0)

        print("desired_norm_l_inf", desired_norm_l_inf)
        print("ar0_mean", ar0_mean)
        print("ar0_std", ar0_std)
        print()
        per_ep_means.append(ar0_mean)
        per_ep_std_div.append(ar0_std)

    per_ep_means = np.array(per_ep_means)
    per_ep_std_div = np.array(per_ep_std_div)
    
    mean_per_methd.append(per_ep_means)
    std_per_methd.append(per_ep_std_div)

plt.figure(figsize=(6, 6))  # Adjust the width and height as needed

# Compute upper and lower bounds for the shaded region
for i in range(len(considered_attack_inds)):
    colors = ['red', 'gold']

    upper_bound = mean_per_methd[i] + std_per_methd[i]
    lower_bound = mean_per_methd[i] - std_per_methd[i]

    plt.plot(all_l_inf_norms, mean_per_methd[i], label=objective_names[considered_attack_inds[i]], color=colors[i])

    plt.fill_between(all_l_inf_norms, lower_bound, upper_bound, color=colors[i], alpha=0.2)

# Labels and legend
plt.xlabel(r'$c$', fontsize=28)
plt.ylabel('L-2 distance', fontsize=28)


formatter = FuncFormatter(lambda x, _: f'{x:.2f}')
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)


plt.xticks(rotation=45, fontsize=28)
plt.yticks(fontsize=28)
#plt.title("Distribution Change with Epsilon")
plt.grid(True)
#plt.legend()
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=24)
# Adjust layout to fit the legend
handles, labels = plt.gca().get_legend_handles_labels()

# Increase line thickness in the legend
for handle in handles:
    handle.set_linewidth(4)
plt.tight_layout()


plt.show()
plt.savefig("diffae/grill_damage_distributions_variation/diffAE_bp_mcmc.png")
