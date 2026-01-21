


'''

conda deactivate
conda deactivate
conda deactivate
export CUDA_VISIBLE_DEVICES=3
cd NVAE/
source nvaeenv1/bin/activate
cd ..
cd alma/
python nvae/nvae_actual_ablation_plots.py

'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


attck_types = ["alma_l2_mcmc", "alma_l2_mcmc1", "alma_l2_mcmc2"]



desired_norm_l_inf = 0.035
#desired_norm_l_inf = 0.04

color_list = ['blue', 'orange', 'green']


objective_names = ["$\kappa$ weights" , "Equal weights", "Random weights"]

plt.figure(figsize=(6, 5))  # Adjust the width and height as needed

for i in range(len(attck_types)):
    adv_div_list = np.load("nvae/deviation_store/NVAE_attack_type"+str(attck_types[i])+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy")

    #np.save("nvae/deviation_store/NVAE_attack_type"+str(attck_types[i])+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", adv_div_list)

    print("len(adv_div_list)", len(adv_div_list))
    plt.plot(adv_div_list, label=objective_names[i], color=color_list[i])


# Labels and legend
plt.xlabel('Epoch', fontsize=28)
plt.ylabel('Mean L-2 dist.', fontsize=28)
#plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.yticks(fontsize=28)
plt.xticks(fontsize=28, rotation=45)

# Get legend handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

# Increase line thickness in the legend
for handle in handles:
    handle.set_linewidth(4)

#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# Adjust layout to fit the legend
plt.tight_layout()


#plt.legend()
plt.savefig("nvae/weighting_strategy_study/nvae_ablation_l_inf_norm_"+str(desired_norm_l_inf)+"_plots.png")
plt.close()

