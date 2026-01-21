
'''

conda deactivate
conda deactivate
conda deactivate
export CUDA_VISIBLE_DEVICES=7
cd NVAE/
source nvaeenv1/bin/activate
cd ..
cd alma/
python nvae/layerLossesPlots.py 


'''


import numpy as np
import matplotlib.pyplot as plt



attck_type = "grill_wass_kf_layerLosses"
#attck_type = "grill_l2_kf_layerLosses"

#attck_type = "la_l2_kf_layerLosses"
desired_norm_l_inf = 0.05


allStepLayerLossesArray = np.load("nvae/stepLayerLossstore//NVAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy")


'''for i in range(len(allStepLayerLossesArray)):
    plt.plot(allStepLayerLossesArray[i])
    I want all the fintsizes to be 28 in size
    I want x axis to be named layer Index
    I want y axis to be named Layer Losses l2
    plt.savefig('allLayerLossPlots/'+attck_type+'/test_'+str(i)+'_.png')
    plt.close()'''


for i in range(len(allStepLayerLossesArray)):
    plt.figure(figsize=(10, 6))  # optional: control image size
    
    plt.plot(allStepLayerLossesArray[i])

    # Set font sizes
    plt.xlabel("Layer Index", fontsize=28)
    plt.ylabel("Layer Losses (L2)", fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    # Optional: title
    #plt.title(f"Layer Losses (Step {i})", fontsize=28)

    # Save figure
    plt.savefig(f'allLayerLossPlots/{attck_type}/test_{i}_.png', bbox_inches='tight')
    plt.close()
