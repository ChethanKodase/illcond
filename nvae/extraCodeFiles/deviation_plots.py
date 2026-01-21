
'''

conda deactivate
conda deactivate
conda deactivate
export CUDA_VISIBLE_DEVICES=3
cd NVAE/
source nvaeenv1/bin/activate
cd ..
cd alma/
python nvae/deviation_plots.py

'''



import numpy as np

#attck_types = [ "la_l2_kf", "grill_l2_kf",  "la_wass_kf_cr", "grill_wass_kf", "grill_wass_kf_cr"]

attck_types = [ "la_l2_kf", "grill_l2_kf",  "la_wass_kf_cr", "grill_wass_kf"]


all_desired_norm_l_inf = [0.01, 0.025, 0.035, 0.037, 0.04, 0.05]


selectedNoise = all_desired_norm_l_inf[3]

for j in range(len(attck_types)):

    selectedAttack = attck_types[j]

    print("selectedNoise", selectedNoise)
    print('selectedAttack', selectedAttack)

    adv_div_list = np.load("nvae/deviation_store/NVAE_attack_type"+str(selectedAttack)+"_norm_bound_"+str(selectedNoise)+"_.npy")

    print("adv_div_list", adv_div_list)