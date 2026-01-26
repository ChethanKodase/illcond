import torch

import torch.nn as nn

#from nvae.utils import add_sn
#from nvae.vae_celeba import NVAE
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.jit import script
import pandas as pd
#from nvae.utils import reparameterize


'''

######################################################################################################################################################


######################################################################################################################################################



# Kuzina framework
conda deactivate
conda deactivate
cd illcond
conda activate /home/luser/anaconda3/envs/inn
python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py --which_gpu 0 --beta_value 5.0 --attck_type latent_l2_kf_SS --which_model TCVAE --desired_norm_l_inf 0.5
python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py --which_gpu 0 --beta_value 5.0 --attck_type latent_l2_kf_SS --which_model VAE --desired_norm_l_inf 0.03


python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 1 --beta_value 5.0 --attck_type latent_wass_kf_SS --which_model TCVAE --desired_norm_l_inf 0.03
python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 1 --beta_value 5.0 --attck_type latent_wass_kf_SS --which_model VAE --desired_norm_l_inf 0.03



python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 1 --beta_value 5.0 --attck_type latent_cos_kf_SS --which_model TCVAE --desired_norm_l_inf 0.03
python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 1 --beta_value 5.0 --attck_type latent_cos_kf_SS --which_model VAE --desired_norm_l_inf 0.03

python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 1 --beta_value 5.0 --attck_type grill_l2_kf_SS --which_model TCVAE --desired_norm_l_inf 0.03
python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 1 --beta_value 5.0 --attck_type grill_l2_kf_SS --which_model VAE --desired_norm_l_inf 0.03



python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 0 --beta_value 5.0 --attck_type grill_wass_kf_SS --which_model TCVAE --desired_norm_l_inf 0.03
python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 0 --beta_value 5.0 --attck_type grill_wass_kf_SS --which_model VAE --desired_norm_l_inf 0.03



python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 0 --beta_value 5.0 --attck_type grill_cos_kf_SS --which_model TCVAE --desired_norm_l_inf 0.03
python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 0 --beta_value 5.0 --attck_type grill_cos_kf_SS --which_model VAE --desired_norm_l_inf 0.03


python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 0 --beta_value 5.0 --attck_type output_l2_kf_SS --which_model TCVAE --desired_norm_l_inf 0.03
python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 0 --beta_value 5.0 --attck_type output_l2_kf_SS --which_model VAE --desired_norm_l_inf 0.03


python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 1 --beta_value 5.0 --attck_type output_wass_kf_SS --which_model TCVAE --desired_norm_l_inf 0.03
python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 1 --beta_value 5.0 --attck_type output_wass_kf_SS --which_model VAE --desired_norm_l_inf 0.03


python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 1 --beta_value 5.0 --attck_type output_attack_cosine --which_model TCVAE --desired_norm_l_inf 0.03
python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 1 --beta_value 5.0 --attck_type output_attack_cosine --which_model VAE --desired_norm_l_inf 0.03

python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 0 --beta_value 5.0 --attck_type output_cos_kf_SS --which_model TCVAE --desired_norm_l_inf 0.03
python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 0 --beta_value 5.0 --attck_type output_cos_kf_SS --which_model VAE --desired_norm_l_inf 0.03


python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 0 --beta_value 5.0 --attck_type lgr_l2_kf_SS --which_model TCVAE --desired_norm_l_inf 0.03
python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 0 --beta_value 5.0 --attck_type lgr_l2_kf_SS --which_model VAE --desired_norm_l_inf 0.03

python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 1 --beta_value 5.0 --attck_type lgr_wass_kf_SS --which_model TCVAE --desired_norm_l_inf 0.03
python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 1 --beta_value 5.0 --attck_type lgr_wass_kf_SS --which_model VAE --desired_norm_l_inf 0.03

python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 1 --beta_value 5.0 --attck_type lgr_cos_kf_SS --which_model TCVAE --desired_norm_l_inf 0.03
python betaVAEandTcVAE/BetaVAEandTCVAEsampleSpecificAttacks.py  --which_gpu 1 --beta_value 5.0 --attck_type lgr_cos_kf_SS --which_model VAE --desired_norm_l_inf 0.03


'''


#device = "cuda:1" if torch.cuda.is_available() else "cpu"

import argparse

parser = argparse.ArgumentParser(description='Adversarial attack on VAE')
#parser.add_argument('--segment', type=int, default=3, help='Segment index')
parser.add_argument('--which_gpu', type=int, default=3, help='Index of the GPU to use (0-N)')
parser.add_argument('--attck_type', type=str, default=5, help='Index of the feature to attack (0-5)')
parser.add_argument('--beta_value', type=str, default=5, help='Index of the feature to attack (0-5)')
parser.add_argument('--which_model', type=str, default=5, help='model to attack')
parser.add_argument('--desired_norm_l_inf', type=float, default=5, help='perturbation norm bounding')


args = parser.parse_args()

#segment = args.segment
which_gpu = args.which_gpu
attck_type = args.attck_type
beta_value = args.beta_value
which_model = args.which_model
desired_norm_l_inf = args.desired_norm_l_inf
device = ("cuda:"+str(which_gpu)+"" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training




import random
from torchvision import datasets, transforms
import os

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

batch_size = 10
celeba_data = datasets.ImageFolder('betaVAEandTcVAE/data_cel1', transform=transform)

train_set, test_set = torch.utils.data.random_split(celeba_data, [int(len(img_list) * 0.08), len(img_list) - int(len(img_list) * 0.08)])
train_data_size = len(train_set)
test_data_size = len(test_set)
print('train_data_size', train_data_size)
print('test_data_size', test_data_size)
trainLoader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)
testLoader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

for idx, (image, label) in enumerate(testLoader):
    images, label = image.to(device), label.to(device)
    print(images.shape)
    break
    

mi, ma = images.min().item(), images.max().item()




from vae import VAE_big, VAE_big_b

#model = VAE_big(device, image_channels=3).to(device)
model = VAE_big(device, image_channels=3).to(device)
train_data_size = 162079
epochs = 199

model_type = which_model

#model_type = "TCVAE"
#model_type = "VAE"

#beta_value = 10.0

#below is for beta = 1.0
#model.load_state_dict(torch.load('/home/luser/autoencoder_attacks/saved_celebA/checkpoints/celebA_CNN_VAE_big_trainSize'+str(train_data_size)+'_epochs'+str(epochs)+'.torch'))

#below is for beta = 5 or 10
#model.load_state_dict(torch.load('/home/luser/autoencoder_attacks/saved_celebA/checkpoints/celebA_seeded_CNN_VAE'+str(beta_value)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epochs)+'.torch'))
#model.load_state_dict(torch.load('/home/luser/autoencoder_attacks/train_aautoencoders/saved_model/checkpoints/celebA_seeded_CNN_VAE'+str(beta_value)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epochs)+'.torch'))
model.load_state_dict(torch.load('betaVAEandTcVAE/saved_celebA/checkpoints/celebA_CNN_'+model_type+''+str(beta_value)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epochs)+'.torch'))
model.eval()


if(attck_type == "aclmd_l2f_cond" or attck_type == "aclmd_l2a_cond" or attck_type == "aclmd_l2a_cond_grad_norm" or attck_type == "aclmd_l2a_cond1" or attck_type == "aclmd_l2a_cond1_mcmc" or attck_type == "aclmd_l2a_mcmc" or attck_type == "aclmd_wasserstein_cond" or attck_type == "aclmd_SKL_cond" or attck_type == "aclmd_cosine_cond" or attck_type=="aclmd_equal_weights" or attck_type=="aclmd_equal_weights_mcmc" or attck_type=="aclmd_random_weights" or attck_type=="aclmd_random_weights_mcmc"):
    def get_condition_weights(model):
        cond_nums = []
        for layer in model.encoder:
            if isinstance(layer, nn.Conv2d):  
                wt_tensor = layer.weight.detach()
                W_matrix = wt_tensor.view(wt_tensor.shape[0], -1)  # Flatten kernels into a 2D matrix
                U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
                condition_number = S.max() / S.min()
                cond_nums.append(condition_number.item())
            else:
                condition_number = 1.0 # technically it is not zero. But since I want to convert these numbers to weights  I am putting it as zero so that more weight doesnt go to activations llayer losses
                cond_nums.append(condition_number)
        wt_tensor = model.fc1.weight.detach()
        W_matrix = wt_tensor.view(wt_tensor.shape[0], -1)  # Flatten kernels into a 2D matrix
        U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
        condition_number = S.max() / S.min()
        cond_nums.append(condition_number.item())

        wt_tensor = model.fc2.weight.detach()
        W_matrix = wt_tensor.view(wt_tensor.shape[0], -1)  # Flatten kernels into a 2D matrix
        U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
        condition_number = S.max() / S.min()
        cond_nums.append(condition_number.item())

        wt_tensor = model.fc3.weight.detach()
        W_matrix = wt_tensor.view(wt_tensor.shape[0], -1)  # Flatten kernels into a 2D matrix
        U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
        condition_number = S.max() / S.min()
        cond_nums.append(condition_number.item())

        for layer in model.decoder:
            if isinstance(layer, nn.ConvTranspose2d):  
                wt_tensor = layer.weight.detach()
                W_matrix = wt_tensor.view(wt_tensor.shape[0], -1)  # Flatten kernels into a 2D matrix
                U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
                condition_number = S.max() / S.min()
                cond_nums.append(condition_number.item())
            else:
                condition_number = 1.0
                cond_nums.append(condition_number)
        cond_nums_array = np.array(cond_nums)

        if(attck_type == "aclmd_l2f_cond"):
            #cond_nums_normalized = (np.sum(cond_nums_array) - cond_nums_array) / np.sum(cond_nums_array)
            cond_cmpli = np.sum(cond_nums_array) - cond_nums_array
            cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
            #cond_nums_array_inv = 1.0/cond_nums_array
            #inv_summ = np.sum(cond_nums_array_inv)
            #cond_nums_normalized = cond_nums_array_inv/inv_summ
            #cond_nums_compli = np.sum(cond_nums_array) - cond_nums_array
            #cond_nums_normalized = cond_nums_compli/ np.sum(cond_nums_compli)
        if(attck_type == "aclmd_wasserstein_cond" or attck_type == "aclmd_SKL_cond" or attck_type == "aclmd_cosine_cond"):
            #cond_nums_normalized = (np.sum(cond_nums_array) - cond_nums_array) / np.sum(cond_nums_array)
            max_ind = np.where(cond_nums_array==cond_nums_array.max())[0][0]
            print("max_ind", max_ind)
            part1 = cond_nums_array[:max_ind+1]
            print('part1', part1)
            part2 = cond_nums_array[max_ind-1:]
            print('part2', part2)
            part_1_quants = part1
            print('part_1_quants', part_1_quants)
            part_2_quants = (np.sum(cond_nums_array)-part2)
            print('part_2_quants', part_2_quants)
            quants_cat = np.concatenate((part_1_quants, part_2_quants))
            print('quants_cat', quants_cat)
            cond_nums_normalized = quants_cat/np.sum(quants_cat)
            print('cond_nums_normalized', cond_nums_normalized)
            cond_nums_normalized = np.concatenate((part1/np.sum(part1), part2/np.sum(part2)))
            cond_cmpli = np.sum(cond_nums_array) - cond_nums_array
            cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)

        if(attck_type == "aclmd_l2a_cond" or attck_type == "aclmd_l2a_mcmc" or attck_type == "aclmd_l2a_cond_grad_norm" or attck_type == "grill_l2_kf_SSw"):
            #cond_nums_normalized = (np.sum(cond_nums_array) - cond_nums_array) / np.sum(cond_nums_array)
            max_ind = np.where(cond_nums_array==cond_nums_array.max())[0][0]
            print("max_ind", max_ind)
            part1 = cond_nums_array[:max_ind+1]
            print('part1', part1)
            part2 = cond_nums_array[max_ind-1:]
            print('part2', part2)
            part_1_quants = part1
            print('part_1_quants', part_1_quants)
            part_2_quants = (np.sum(cond_nums_array)-part2)
            print('part_2_quants', part_2_quants)
            quants_cat = np.concatenate((part_1_quants, part_2_quants))
            print('quants_cat', quants_cat)
            cond_nums_normalized = quants_cat/np.sum(quants_cat)
            print('cond_nums_normalized', cond_nums_normalized)
            cond_nums_normalized = np.concatenate((part1/np.sum(part1), part2/np.sum(part2)))
            cond_cmpli = np.sum(cond_nums_array) - cond_nums_array
            cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)


        if(attck_type == "aclmd_l2a_cond1"):
            #cond_nums_normalized = (np.sum(cond_nums_array) - cond_nums_array) / np.sum(cond_nums_array)
            max_ind = np.where(cond_nums_array==cond_nums_array.max())[0][0]
            print("max_ind", max_ind)
            part1 = cond_nums_array[:max_ind+1]
            print('part1', part1)
            part2 = cond_nums_array[max_ind-1:]
            print('part2', part2)
            part_1_quants = part1
            print('part_1_quants', part_1_quants)
            part_2_quants = (np.sum(cond_nums_array)-part2)
            print('part_2_quants', part_2_quants)
            quants_cat = np.concatenate((part_1_quants, part_2_quants))
            print('quants_cat', quants_cat)
            cond_nums_normalized = quants_cat/np.sum(quants_cat)
            print('cond_nums_normalized', cond_nums_normalized)
            cond_nums_normalized = np.concatenate((part1/np.sum(part1), part2/np.sum(part2)))
            cond_cmpli = np.sum(cond_nums_array) - cond_nums_array
            cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)


        if(attck_type == "aclmd_l2a_cond1_mcmc"):
            #cond_nums_normalized = (np.sum(cond_nums_array) - cond_nums_array) / np.sum(cond_nums_array)
            max_ind = np.where(cond_nums_array==cond_nums_array.max())[0][0]
            print("max_ind", max_ind)
            part1 = cond_nums_array[:max_ind+1]
            print('part1', part1)
            part2 = cond_nums_array[max_ind-1:]
            print('part2', part2)
            part_1_quants = part1
            print('part_1_quants', part_1_quants)
            part_2_quants = (np.sum(cond_nums_array)-part2)
            print('part_2_quants', part_2_quants)
            quants_cat = np.concatenate((part_1_quants, part_2_quants))
            print('quants_cat', quants_cat)
            cond_nums_normalized = quants_cat/np.sum(quants_cat)
            print('cond_nums_normalized', cond_nums_normalized)
            cond_nums_normalized = np.concatenate((part1/np.sum(part1), part2/np.sum(part2)))
            cond_cmpli = np.sum(cond_nums_array) - cond_nums_array
            cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)


        if(attck_type == "aclmd_equal_weights"):
            print("cond_nums_array.shape", cond_nums_array.shape)
            cond_nums_normalized = np.full((cond_nums_array.shape), 1 / cond_nums_array.shape[0])

            print("all_condition_nums.shape", cond_nums_normalized.shape)
            print("all_condition_nums", cond_nums_normalized)

        if(attck_type == "aclmd_equal_weights_mcmc"):
            print("cond_nums_array.shape", cond_nums_array.shape)
            cond_nums_normalized = np.full((cond_nums_array.shape), 1 / cond_nums_array.shape[0])

            print("all_condition_nums.shape", cond_nums_normalized.shape)
            print("all_condition_nums", cond_nums_normalized)

        if(attck_type == "aclmd_random_weights"):
            print("cond_nums_array.shape", cond_nums_array.shape)
            cond_nums_normalized = np.random.rand((cond_nums_array.shape[0]))

            print("all_condition_nums.shape", cond_nums_normalized.shape)
            print("all_condition_nums", cond_nums_normalized)

        if(attck_type == "aclmd_random_weights_mcmc"):
            print("cond_nums_array.shape", cond_nums_array.shape)
            cond_nums_normalized = np.random.rand((cond_nums_array.shape[0]))

            print("all_condition_nums.shape", cond_nums_normalized.shape)
            print("all_condition_nums", cond_nums_normalized)


        if(attck_type == "aclmd_l2_cond_nonlin" or attck_type == "aclmd_wasserstein_nonlin" or attck_type == "aclmd_SKL_cond_nonlin" or attck_type == "aclmd_cosine_cond_nonlin"):
            #cond_nums_normalized = (np.sum(cond_nums_array) - cond_nums_array) / np.sum(cond_nums_array)
            #cond_cmpli = np.sum(cond_nums_array) - cond_nums_array
            #cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
            cond_nums_array_inv = 1.0/cond_nums_array
            inv_summ = np.sum(cond_nums_array_inv)
            cond_nums_normalized = cond_nums_array_inv/inv_summ
            #cond_nums_compli = np.sum(cond_nums_array) - cond_nums_array
            #cond_nums_normalized = cond_nums_compli/ np.sum(cond_nums_compli)

        if(attck_type == "aclmd_l2_cond_dir"):
            cond_nums_normalized = (cond_nums_array) / np.sum(cond_nums_array)
        return cond_nums, cond_nums_normalized
    with torch.no_grad():
        cond_nums, cond_normal = get_condition_weights(model)
        print("cond_nums", cond_nums)
        print("cond_normal", cond_normal)



roll_no = 1

#segment = 20


all_features = ["bald", "beard", "oldfemaleGlass", "hat", "blackWomen", "generalWhiteWomen", "blackMen", "generalWhiteMen", "men", "women", "young", "old", "youngmen", "oldmen", "youngwomen", "oldwomen", "oldblackmen", "oldblackwomen", "oldwhitemen", "oldwhitewomen", "youndblackmen", "youndblackwomen", "youngwhitemen", "youngwhitewomen" ]

populations_all_features = ["bald", "beard", "oldfemaleGlass", "hat", "blackWomen", "generalWhiteWomen", "blackMen", "generalWhiteMen", "men : 84434", "women : 118165", "young : 156734", "old : 45865", "youngmen : 53448 ", "oldmen : 7003", "youngwomen : 103287", "oldwomen : 1116" ]



#source_im = torch.load("/home/luser/autoencoder_attacks/train_aautoencoders/fairness_trials/attack_saves/"+str(select_feature)+"_d/images.pt")[segment].unsqueeze(0).to(device) 

#source_im = torch.load("/home/luser/autoencoder_attacks/test_sets/celebA_test_set.pt")[segment].unsqueeze(0).to(device) 

#source_ims = torch.load("/home/luser/autoencoder_attacks/test_sets/celebA_test_set.pt").to(device) 


model.eval()


if attck_type == "weighted_combi_l2":
    noise_addition = 2.0 * torch.rand(1, 3, 64, 64).to(device) - 1.0
    layer_weights_un = nn.Parameter(torch.rand(1, 29), requires_grad=True).to(device)
if attck_type == "weighted_combi_l2_tw":
    noise_addition = 2.0 * torch.rand(1, 3, 64, 64).to(device) - 1.0
    layer_weights_un = nn.Parameter(torch.tensor([ 0.0, 0.0, 0.0, 1/23, 1/23, 1/23, 1/23, 1/23, 1/23, 1/23, 1/23, 1/23, 1/23, 1/23,1/23, 1/23, 1/23, 1/23, 1/23, 1/23, 1/23,1/23, 1/23, 1/23, 1/23, 1/23, 0.0, 0.0, 0.0]), requires_grad=True).to(device)
else:
    noise_addition = 2.0 * torch.rand(1, 3, 64, 64).to(device) - 1.0
    
#noise_addition = 0.08 * (2 * noise_addition - 1)


#desired_norm_l_inf = 0.094  # Worked very well
#desired_norm_l_inf = 0.094  # Worked very well
#desired_norm_l_inf = 0.07  # Worked very well 0.15 is goog



import torch.optim as optim


noise_addition = (torch.randn_like(images) * 0.05).to(device)
#noise_addition = 0.2 * torch.rand(1, 3, 64, 64).to(device)
print("noise_addition.max(), noise_addition.min()", noise_addition.max(), noise_addition.min())
noise_addition = noise_addition.clone().detach().requires_grad_(True)
optimizer = optim.Adam([noise_addition], lr=0.0001)
print("what is going out", noise_addition.shape)


adv_alpha = 0.5

criterion = nn.MSELoss()

num_steps = 30000

prev_loss = 0.0

def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()

def wasserstein_distance(tensor_a, tensor_b):

    tensor_a_flat = torch.flatten(tensor_a)
    tensor_b_flat = torch.flatten(tensor_b)
    tensor_a_sorted, _ = torch.sort(tensor_a_flat)
    tensor_b_sorted, _ = torch.sort(tensor_b_flat)    
    wasserstein_dist = torch.mean(torch.abs(tensor_a_sorted - tensor_b_sorted))
    
    return wasserstein_dist

def compute_mean_and_variance(tensor):
    flattened_tensor = torch.flatten(tensor)  # Flatten the tensor
    mean = torch.mean(flattened_tensor)  # Compute mean
    variance = torch.var(flattened_tensor, unbiased=False)  # Compute variance (unbiased=False for population variance)
    return mean, variance

# Function to compute symmetric KL divergence between two tensors
def get_symmetric_KLDivergence(input1, input2):
    mu1, var1 = compute_mean_and_variance(input1)
    mu2, var2 = compute_mean_and_variance(input2)
    
    kl_1_to_2 = torch.log(var2 / var1) + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5
    kl_2_to_1 = torch.log(var1 / var2) + (var2 + (mu2 - mu1) ** 2) / (2 * var1) - 0.5
    
    symmetric_kl = (kl_1_to_2 + kl_2_to_1) / 2
    return symmetric_kl





def get_weighted_combinations_l2(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) 
        encoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) 
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out


    lipschitzt_loss_encoder = criterion(z1, z2) 
    lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  

    loss_to_maximize =  (lipschitzt_loss_encoder + encoder_lip_sum  +  lipschitzt_loss_decoder + decoder_lip_sum ) * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_l2_enco(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) 
        encoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    #attack_flow = model.fc3(z1)
    #source_flow = model.fc3(z2)
    #decoder_lip_sum = 0

    '''for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) 
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out'''


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  

    loss_to_maximize =  encoder_lip_sum * criterion(z1, z2)

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_cosine_enco(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = (cos(attack_out, source_out)-1.0)**2 
        encoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    return encoder_lip_sum * (cos(z1, z2)-1)**2



def get_true_weighted_combinations_l2(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer, wt in zip(model.encoder, layer_weights_un[:14]):
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) * wt
        encoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    for layer, wt in zip(model.decoder, layer_weights_un[14:]):
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) * wt
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out


    lipschitzt_loss_encoder = criterion(z1, z2) * layer_weights_un[15]
    lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  

    loss_to_maximize =  (lipschitzt_loss_encoder + encoder_lip_sum  + decoder_lip_sum ) * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_k_eq_latent_l2(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize =  criterion(z1, z2)   * criterion(source_recon, adv_gen)

    return loss_to_maximize, adv_gen, source_recon


def output_attack_l2(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize = criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon

def output_attack_wasserstein(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize = wasserstein_distance(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon


def output_attack_SKL(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize = get_symmetric_KLDivergence(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon


def output_attack_cosine(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
     
    loss_to_maximize = (cos(normalized_attacked, adv_gen)-1.0)**2

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_k_eq_latent_wasserstein(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize =  wasserstein_distance(z1, z2)   * wasserstein_distance(source_recon, adv_gen)

    return loss_to_maximize, adv_gen, source_recon

def get_weighted_combinations_k_eq_latent_SKL(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize =  get_symmetric_KLDivergence(z1, z2)   * get_symmetric_KLDivergence(source_recon, adv_gen)

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_k_eq_latent_cos(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2
 
    loss_to_maximize =  ((cos(z1, z2)-1.0)**2)   * (cos(source_recon, adv_gen)-1.0)**2

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_wasserstein(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = wasserstein_distance(attack_out, source_out) 
        encoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    '''attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0'''

    '''for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = wasserstein_distance(attack_out, source_out) 
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out'''


    #lipschitzt_loss_encoder = wasserstein_distance(z1, z2) 
    #lipschitzt_loss_decoder = wasserstein_distance(normalized_attacked, adv_gen)  

    loss_to_maximize =  encoder_lip_sum * wasserstein_distance(z1, z2)

    return loss_to_maximize, adv_gen, source_recon

def get_weighted_combinations_SKL(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = get_symmetric_KLDivergence(attack_out, source_out) 
        encoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    '''attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = get_symmetric_KLDivergence(attack_out, source_out) 
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out'''


    #lipschitzt_loss_encoder = get_symmetric_KLDivergence(z1, z2) 
    #lipschitzt_loss_decoder = get_symmetric_KLDivergence(normalized_attacked, adv_gen)  

    loss_to_maximize =  encoder_lip_sum * get_symmetric_KLDivergence(z1, z2)

    return loss_to_maximize, adv_gen, source_recon


def get_layer_prod_loss(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) 
        encoder_lip_sum *= layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) 
        decoder_lip_sum *= layer_lip_const
        attack_flow = attack_out
        source_flow = source_out


    lipschitzt_loss_encoder = criterion(z1, z2) 
    lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  

    loss_to_maximize =  lipschitzt_loss_encoder * encoder_lip_sum * decoder_lip_sum 

    return loss_to_maximize, adv_gen, source_recon




def get_l2_combinations_k_equals_latent_loss(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    lipschitzt_loss_encoder = criterion(z1, z2) 
    lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  

    loss_to_maximize =  lipschitzt_loss_encoder  *  lipschitzt_loss_decoder 

    return loss_to_maximize, adv_gen, source_recon


def get_latent_space_l2_loss(normalized_attacked, source_im):
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    loss_to_maximize =  criterion(z1, z2) 

    return loss_to_maximize



def get_latent_space_wass_loss(normalized_attacked, source_im):
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    #loss_to_maximize =  wasserstein_distance(z1, z2) 

    loss_to_maximize = 0
    for zs1, zs2 in zip(z1, z2):
        loss_to_maximize = loss_to_maximize + wasserstein_distance(zs1, zs2) 

    return loss_to_maximize


def get_latent_space_cos_loss(normalized_attacked, source_im):
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    #loss_to_maximize =  wasserstein_distance(z1, z2) 

    loss_to_maximize = 0
    for zs1, zs2 in zip(z1, z2):
        loss_to_maximize = loss_to_maximize + (cos(zs1, zs2)-1.0)**2

        

    return loss_to_maximize


def get_output_space_l2_loss_kf(normalized_attacked, source_im):
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

    #loss_to_maximize =  criterion(adv_gen, source_recon) 

    loss_to_maximize = 0
    for adv_gen_k, source_recon_k in zip(adv_gen,source_recon):
        loss_to_maximize = loss_to_maximize + criterion(adv_gen_k, source_recon_k) 


    return loss_to_maximize


def get_output_space_wass_loss_kf(normalized_attacked, source_im):
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

    #loss_to_maximize =  criterion(adv_gen, source_recon) 
    

    loss_to_maximize = 0
    for adv_gen_k, source_recon_k in zip(adv_gen,source_recon):
        loss_to_maximize = loss_to_maximize + wasserstein_distance(adv_gen_k, source_recon_k) 

    return loss_to_maximize

def get_hmc_lat(z1, normalized_attacked):
    z = z1.clone().detach().requires_grad_(True)  # Start point for MCMC
    x = normalized_attacked.detach()              # Adversarial input
    step_size = 0.002
    n_steps = 5
    leapfrog_steps = 5

    #samples = []
    for i in range(n_steps):
        p = torch.randn_like(z)  # Sample momentum
        z_new = z.clone()
        p_new = p.clone()
        x_mean = model.decoder(model.fc3(z_new))
        x_flat, x_mean_flat = x.view(x.size(0), -1), x_mean.view(x.size(0), -1)
        log_p_x = -((x_flat - x_mean_flat) ** 2).sum(dim=1) / 2  # assuming Gaussian decoder
        log_p_z = -0.5 * (z_new ** 2).sum(dim=1)                # standard normal prior
        log_post = (log_p_x + log_p_z).sum()
        grad = torch.autograd.grad(log_post, z_new)[0]

        # Leapfrog integration
        p_new = p_new + 0.5 * step_size * grad
        for _ in range(leapfrog_steps):
            z_new = z_new + step_size * p_new
            z_new = z_new.detach().requires_grad_(True)
            x_mean = model.decoder(model.fc3(z_new))
            x_flat, x_mean_flat = x.view(x.size(0), -1), x_mean.view(x.size(0), -1)
            log_p_x = -((x_flat - x_mean_flat) ** 2).sum(dim=1) / 2
            log_p_z = -0.5 * (z_new ** 2).sum(dim=1)
            log_post = (log_p_x + log_p_z).sum()
            grad = torch.autograd.grad(log_post, z_new)[0]
            p_new = p_new + step_size * grad
        p_new = p_new + 0.5 * step_size * grad
        p_new = -p_new  # Make symmetric

        #with torch.no_grad():
        logp_current = -0.5 * (z ** 2).sum(dim=1) - ((x.view(x.size(0), -1) - model.decoder(model.fc3(z)).view(x.size(0), -1)) ** 2).sum(dim=1) / 2
        logp_new = -0.5 * (z_new ** 2).sum(dim=1) - ((x.view(x.size(0), -1) - model.decoder(model.fc3(z_new)).view(x.size(0), -1)) ** 2).sum(dim=1) / 2
        
        accept_ratio = torch.exp(logp_new - logp_current).clamp(max=1.0)
        #print("accept_ratio", accept_ratio)
        #print("torch.rand_like(accept_ratio) ", torch.rand_like(accept_ratio))
        mask = torch.rand_like(accept_ratio) < accept_ratio
        #print("mask", mask)
        z = torch.where(mask.unsqueeze(1), z_new, z)
        z = z.detach().requires_grad_(True)  # Prepare for next iteration
        #samples.append(z)

    z_mcmc = z.detach()  # Final robust latent sample
    #print("z_mcmc.shape", z_mcmc.shape)
    return z_mcmc


def get_hmc_lat1(z1, normalized_attacked):
    z = z1#.clone().detach().requires_grad_(True)  # Start point for MCMC
    x = normalized_attacked#.detach()              # Adversarial input
    step_size = 0.008
    n_steps = 20
    leapfrog_steps = 10

    #samples = []
    for i in range(n_steps):
        p = torch.randn_like(z)  # Sample momentum
        z_new = z.clone()
        p_new = p.clone()
        x_mean = model.decoder(model.fc3(z_new))
        x_flat, x_mean_flat = x.view(x.size(0), -1), x_mean.view(x.size(0), -1)
        log_p_x = -((x_flat - x_mean_flat) ** 2).sum(dim=1) / 2  # assuming Gaussian decoder
        log_p_z = -0.5 * (z_new ** 2).sum(dim=1)                # standard normal prior
        log_post = (log_p_x + log_p_z).sum()
        grad = torch.autograd.grad(log_post, z_new)[0]

        # Leapfrog integration
        p_new = p_new + 0.5 * step_size * grad
        for _ in range(leapfrog_steps):
            z_new = z_new + step_size * p_new
            z_new = z_new#.detach().requires_grad_(True)
            x_mean = model.decoder(model.fc3(z_new))
            x_flat, x_mean_flat = x.view(x.size(0), -1), x_mean.view(x.size(0), -1)
            log_p_x = -((x_flat - x_mean_flat) ** 2).sum(dim=1) / 2
            log_p_z = -0.5 * (z_new ** 2).sum(dim=1)
            log_post = (log_p_x + log_p_z).sum()
            grad = torch.autograd.grad(log_post, z_new)[0]
            p_new = p_new + step_size * grad
        p_new = p_new + 0.5 * step_size * grad
        p_new = -p_new  # Make symmetric

        #with torch.no_grad():
        logp_current = -0.5 * (z ** 2).sum(dim=1) - ((x.view(x.size(0), -1) - model.decoder(model.fc3(z)).view(x.size(0), -1)) ** 2).sum(dim=1) / 2
        logp_new = -0.5 * (z_new ** 2).sum(dim=1) - ((x.view(x.size(0), -1) - model.decoder(model.fc3(z_new)).view(x.size(0), -1)) ** 2).sum(dim=1) / 2
        
        accept_ratio = torch.exp(logp_new - logp_current).clamp(max=1.0)
        #print("accept_ratio", accept_ratio)
        #print("torch.rand_like(accept_ratio) ", torch.rand_like(accept_ratio))
        mask = torch.rand_like(accept_ratio) < accept_ratio
        #print("mask", mask)
        z = torch.where(mask.unsqueeze(1), z_new, z)
        z = z#.detach().requires_grad_(True)  # Prepare for next iteration
        #samples.append(z)

    z_mcmc = z#.detach()  # Final robust latent sample
    #print("z_mcmc.shape", z_mcmc.shape)
    return z_mcmc




def get_latent_space_l2_loss_mcmc(normalized_attacked, source_im):
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    z1 = get_hmc_lat1(z1, normalized_attacked)

    loss_to_maximize =  criterion(z1, z2) 

    #print("see convergence", samples)
    return loss_to_maximize


def get_latent_space_cosine_loss(normalized_attacked, source_im):
    
    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    return cos(z1, z2) 


def get_latent_space_wasserstein_loss(normalized_attacked, source_im):
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    loss_to_maximize =  wasserstein_distance(z1, z2) 

    return loss_to_maximize, adv_gen, source_recon


def get_latent_space_SKL_loss(normalized_attacked, source_im):
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    loss_to_maximize =  get_symmetric_KLDivergence(z1, z2) 

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_l2_aclmd_l2(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) 
        encoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = criterion(attack_out, source_out) 
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    loss_to_maximize =  (criterion(z1, z2) + decoder_lip_sum) * criterion(normalized_attacked, adv_gen)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon



def get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += criterion(attack_out, source_out)#*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = criterion(mu1, mu2)#*cond_normal[l_ct]
    l_ct += 1

    rep_loss = criterion(std1 * esp1, std2 * esp2)#*cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = criterion(attack_flow, source_flow)#*cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += criterion(attack_out, source_out) #*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    #loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(normalized_attacked, attack_flow)
    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(source_flow, attack_flow)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, attack_flow, source_flow



def idivLossesSumL2(adv_gen, source_recon):
    loss_to_maximize = 0
    for adv_gen_k, source_recon_k in zip(adv_gen,source_recon):
        loss_to_maximize = loss_to_maximize + criterion(adv_gen_k, source_recon_k) 
    return loss_to_maximize

def get_grill_l2_kf(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += idivLossesSumL2(attack_out, source_out)#*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = idivLossesSumL2(mu1, mu2)#*cond_normal[l_ct]
    l_ct += 1

    rep_loss = idivLossesSumL2(std1 * esp1, std2 * esp2)#*cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = idivLossesSumL2(attack_flow, source_flow)#*cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += idivLossesSumL2(attack_out, source_out) #*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    #loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(normalized_attacked, attack_flow)
    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * idivLossesSumL2(source_flow, attack_flow)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, attack_flow, source_flow



def get_grill_l2_kfNw(normalized_attacked, source_im, cond_normal):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    lossItems = []
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        theLoss = idivLossesSumL2(attack_out, source_out)
        lossItems.append(theLoss.item())
        encoder_lip_sum += theLoss *cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    #ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    ae_perturbed_embeds = attack_out
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    #train_embeds = model.encoder(source_im.to(device))
    train_embeds = source_out
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    #esp2 = torch.randn(*mu2.size()).to(device)
    #esp2 = esp1
    z2 = mu2 + std2 * esp1

    theLoss = idivLossesSumL2(mu1, mu2)
    lossItems.append(theLoss.item())
    mu_loss = theLoss*cond_normal[l_ct]
    l_ct += 1

    theLoss = idivLossesSumL2(std1 * esp1, std2 * esp1)
    lossItems.append(theLoss.item())
    rep_loss = theLoss*cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    theLoss = idivLossesSumL2(attack_flow, source_flow)
    lossItems.append(theLoss.item())
    fc3_loss = theLoss *cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        theLoss = idivLossesSumL2(attack_out, source_out)
        lossItems.append(theLoss.item())
        decoder_lip_sum += theLoss *cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    #loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(normalized_attacked, attack_flow)
    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * idivLossesSumL2(source_flow, attack_flow)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, attack_flow, source_flow, np.array(lossItems)


def get_grill_l2_kf_mcmc(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += idivLossesSumL2(attack_out, source_out)#*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1
    z1 = get_hmc_lat1(z1, normalized_attacked)


    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = idivLossesSumL2(mu1, mu2)#*cond_normal[l_ct]
    l_ct += 1

    rep_loss = idivLossesSumL2(std1 * esp1, std2 * esp2)#*cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = idivLossesSumL2(attack_flow, source_flow)#*cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += idivLossesSumL2(attack_out, source_out) #*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    #loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(normalized_attacked, attack_flow)
    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * idivLossesSumL2(source_flow, attack_flow)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, attack_flow, source_flow



def get_output_l2_kf_mcmc(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    #l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        #encoder_lip_sum += idivLossesSumL2(attack_out, source_out)#*cond_normal[l_ct]
        #l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    z1 = get_hmc_lat1(z1, normalized_attacked)


    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    #mu_loss = idivLossesSumL2(mu1, mu2)#*cond_normal[l_ct]
    #l_ct += 1

    #rep_loss = idivLossesSumL2(std1 * esp1, std2 * esp2)#*cond_normal[l_ct]
    #l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    #decoder_lip_sum = 0

    #fc3_loss = idivLossesSumL2(attack_flow, source_flow)#*cond_normal[l_ct]
    #l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        #decoder_lip_sum += idivLossesSumL2(attack_out, source_out) #*cond_normal[l_ct]
        #l_ct += 1
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    #loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(normalized_attacked, attack_flow)
    #loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * idivLossesSumL2(source_flow, attack_flow)
    loss_to_maximize =  idivLossesSumL2(source_flow, attack_flow)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, attack_flow, source_flow



def get_grill_l2_kfd(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += idivLossesSumL2(attack_out, source_out)#*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = idivLossesSumL2(mu1, mu2)#*cond_normal[l_ct]
    l_ct += 1

    rep_loss = idivLossesSumL2(std1 * esp1, std2 * esp2)#*cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = idivLossesSumL2(attack_flow, source_flow)#*cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += idivLossesSumL2(attack_out, source_out) #*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    #loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(normalized_attacked, attack_flow)
    #loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(source_flow, attack_flow)
    loss_to_maximize =  (decoder_lip_sum) * idivLossesSumL2(source_flow, attack_flow)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, attack_flow, source_flow



def get_grill_l2_kfe(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += criterion(attack_out, source_out)#*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = criterion(mu1, mu2)#*cond_normal[l_ct]
    l_ct += 1

    rep_loss = criterion(std1 * esp1, std2 * esp2)#*cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = criterion(attack_flow, source_flow)#*cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += criterion(attack_out, source_out) #*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    #loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(normalized_attacked, attack_flow)
    #loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(source_flow, attack_flow)
    loss_to_maximize =  (encoder_lip_sum) * criterion(source_flow, attack_flow)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, attack_flow, source_flow




def get_grill_cos_kf(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += (cos(attack_out, source_out)-1.0)**2  #*cond_normal[l_ct] 
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = (cos(mu1, mu2)-1.0)**2    #*cond_normal[l_ct]
    l_ct += 1

    rep_loss = (cos(std1 * esp1, std2 * esp2)-1.0)**2   #*cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = (cos(attack_flow, source_flow)-1.0)**2#*cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += (cos(attack_out, source_out)-1.0)**2 #*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    #loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(normalized_attacked, attack_flow)
    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * (cos(attack_flow, source_flow)-1.0)**2

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, attack_flow, source_flow




def get_grill_cos_kf_mcmc(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += (cos(attack_out, source_out)-1.0)**2  #*cond_normal[l_ct] 
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    z1 = get_hmc_lat1(z1, normalized_attacked)

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = (cos(mu1, mu2)-1.0)**2    #*cond_normal[l_ct]
    l_ct += 1

    rep_loss = (cos(std1 * esp1, std2 * esp2)-1.0)**2   #*cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = (cos(attack_flow, source_flow)-1.0)**2#*cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += (cos(attack_out, source_out)-1.0)**2 #*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    #loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(normalized_attacked, attack_flow)
    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * (cos(attack_flow, source_flow)-1.0)**2

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, attack_flow, source_flow




def get_grill_wass_kf(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += wasserstein_distance(attack_out, source_out)#*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = wasserstein_distance(mu1, mu2)#*cond_normal[l_ct]
    l_ct += 1

    rep_loss = wasserstein_distance(std1 * esp1, std2 * esp2)#*cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = wasserstein_distance(attack_flow, source_flow)#*cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += wasserstein_distance(attack_out, source_out) #*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    #loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(normalized_attacked, attack_flow)
    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * wasserstein_distance(source_flow, attack_flow)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, attack_flow, source_flow


def get_grill_l2_kfw(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += criterion(attack_out, source_out)*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = criterion(mu1, mu2)*cond_normal[l_ct]
    l_ct += 1

    rep_loss = criterion(std1 * esp1, std2 * esp2)*cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = criterion(attack_flow, source_flow)*cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += criterion(attack_out, source_out) *cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    #loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(normalized_attacked, attack_flow)
    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(source_flow, attack_flow)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, attack_flow, source_flow





def get_weighted_combinations_l2_aclmd_l2_cond_mcmc(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += criterion(attack_out, source_out)*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    z1 = get_hmc_lat1(z1, normalized_attacked)

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = criterion(mu1, mu2)*cond_normal[l_ct]
    l_ct += 1

    rep_loss = criterion(std1 * esp1, std2 * esp2)*cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = criterion(attack_flow, source_flow)*cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += criterion(attack_out, source_out) *cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    #loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(normalized_attacked, attack_flow)
    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(source_flow, attack_flow)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, attack_flow, source_flow




def get_weighted_combinations_l2_aclmd_l2_mcmc(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += criterion(attack_out, source_out)*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    z1 = get_hmc_lat1(z1, normalized_attacked)


    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = criterion(mu1, mu2)*cond_normal[l_ct]
    l_ct += 1

    rep_loss = criterion(std1 * esp1, std2 * esp2)*cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = criterion(attack_flow, source_flow)*cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += criterion(attack_out, source_out) *cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    #loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(normalized_attacked, attack_flow)
    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(source_flow, attack_flow)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, attack_flow, source_flow








def get_output_wass_mcmc(normalized_attacked, source_im):
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        #encoder_lip_sum += criterion(attack_out, source_out) #*cond_normal[l_ct]
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    z1 = get_hmc_lat1(z1, normalized_attacked)


    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    #loss_to_maximize =  criterion(z1, z2) 

    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    #decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        #decoder_lip_sum += criterion(attack_out, source_out) #*cond_normal[l_ct]
        attack_flow = attack_out
        source_flow = source_out

    loss_to_maximize =  wasserstein_distance(attack_flow, source_flow) 


    return loss_to_maximize, attack_flow, source_flow






def get_weighted_combinations_aclmd_wasserstein_cond(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += wasserstein_distance(attack_out, source_out)*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = wasserstein_distance(mu1, mu2)*cond_normal[l_ct]
    l_ct += 1

    rep_loss = wasserstein_distance(std1 * esp1, std2 * esp2)*cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = wasserstein_distance(attack_flow, source_flow)*cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += wasserstein_distance(attack_out, source_out) *cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * wasserstein_distance(source_flow, attack_flow)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, attack_flow, source_flow





def get_weighted_combinations_aclmd_SKL_cond(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += get_symmetric_KLDivergence(attack_out, source_out)*cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = get_symmetric_KLDivergence(mu1, mu2)*cond_normal[l_ct]
    l_ct += 1

    rep_loss = get_symmetric_KLDivergence(std1 * esp1, std2 * esp2)*cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = get_symmetric_KLDivergence(attack_flow, source_flow)*cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += get_symmetric_KLDivergence(attack_out, source_out) *cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * get_symmetric_KLDivergence(source_flow, attack_flow)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, attack_flow, source_flow




def get_weighted_combinations_aclmd_cos_cond(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        encoder_lip_sum += (cos(attack_out, source_out)-1.0)**2  *cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    mu_loss = (cos(mu1, mu2)-1.0)**2  *cond_normal[l_ct]
    l_ct += 1

    rep_loss = (cos(std1 * esp1, std2 * esp2)-1.0)**2 *cond_normal[l_ct]
    l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    fc3_loss = (cos(attack_flow, source_flow)-1.0)**2  *cond_normal[l_ct]
    l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        decoder_lip_sum += (cos(attack_out, source_out)-1.0)**2 *cond_normal[l_ct]
        l_ct += 1
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * (cos(source_flow, attack_flow)-1.0)**2

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, attack_flow, source_flow





def get_weighted_combinations_l2_aclmd_wasserstein(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = wasserstein_distance(attack_out, source_out) 
        encoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = wasserstein_distance(attack_out, source_out) 
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    loss_to_maximize =  (wasserstein_distance(z1, z2) + decoder_lip_sum) * wasserstein_distance(normalized_attacked, adv_gen)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_l2_aclmd_SKL(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = get_symmetric_KLDivergence(attack_out, source_out) 
        encoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = get_symmetric_KLDivergence(attack_out, source_out) 
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    loss_to_maximize =  (get_symmetric_KLDivergence(z1, z2) + decoder_lip_sum) * get_symmetric_KLDivergence(normalized_attacked, adv_gen)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon


def get_weighted_combinations_l2_aclmd_cos(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = (cos(attack_out, source_out)-1.0)**2 
        encoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        layer_lip_const = (cos(attack_out, source_out)-1.0)**2 
        decoder_lip_sum += layer_lip_const
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #loss_to_maximize =  (get_symmetric_KLDivergence(z1, z2) + decoder_lip_sum) * get_symmetric_KLDivergence(normalized_attacked, adv_gen)

    loss_to_maximize =  ((cos(z1, z2)-1.0)**2 + decoder_lip_sum ) * (cos(normalized_attacked, adv_gen)-1.0)**2


    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, adv_gen, source_recon



def get_oa_cos_kf(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    #encoder_lip_sum = 0
    #l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        #encoder_lip_sum += (cos(attack_out, source_out)-1.0)**2  #*cond_normal[l_ct] 
        #l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    #mu_loss = (cos(mu1, mu2)-1.0)**2    #*cond_normal[l_ct]
    #l_ct += 1

    #rep_loss = (cos(std1 * esp1, std2 * esp2)-1.0)**2   #*cond_normal[l_ct]
    #l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    #decoder_lip_sum = 0

    #fc3_loss = (cos(attack_flow, source_flow)-1.0)**2#*cond_normal[l_ct]
    #l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        #decoder_lip_sum += (cos(attack_out, source_out)-1.0)**2 #*cond_normal[l_ct]
        #l_ct += 1
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    #loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(normalized_attacked, attack_flow)
    loss_to_maximize =  (cos(attack_flow, source_flow)-1.0)**2

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, attack_flow, source_flow



def get_latent_l2_mcmc(normalized_attacked, source_im):
    
    attack_flow = normalized_attacked
    source_flow = source_im
    encoder_lip_sum = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        #encoder_lip_sum += criterion(attack_out, source_out) #*cond_normal[l_ct]
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    z1 = get_hmc_lat1(z1, normalized_attacked)


    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2

    loss_to_maximize =  criterion(z1, z2) 

    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    #decoder_lip_sum = 0

    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        #decoder_lip_sum += criterion(attack_out, source_out) #*cond_normal[l_ct]
        attack_flow = attack_out
        source_flow = source_out

    return loss_to_maximize, attack_flow, source_flow




def get_lgr_l2_kf(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    #encoder_lip_sum = 0
    #l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        #encoder_lip_sum += criterion(attack_out, source_out)#*cond_normal[l_ct]
        #l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    #mu_loss = criterion(mu1, mu2)#*cond_normal[l_ct]
    #l_ct += 1

    #rep_loss = criterion(std1 * esp1, std2 * esp2)#*cond_normal[l_ct]
    #l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    #decoder_lip_sum = 0

    #fc3_loss = criterion(attack_flow, source_flow)#*cond_normal[l_ct]
    #l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        #decoder_lip_sum += criterion(attack_out, source_out) #*cond_normal[l_ct]
        #l_ct += 1
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    #loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(normalized_attacked, attack_flow)
    loss_to_maximize = criterion(z1, z2)* criterion(source_flow, attack_flow)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, attack_flow, source_flow





def get_lgr_wass_kf(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    #encoder_lip_sum = 0
    #l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        #encoder_lip_sum += criterion(attack_out, source_out)#*cond_normal[l_ct]
        #l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    #mu_loss = criterion(mu1, mu2)#*cond_normal[l_ct]
    #l_ct += 1

    #rep_loss = criterion(std1 * esp1, std2 * esp2)#*cond_normal[l_ct]
    #l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    #decoder_lip_sum = 0

    #fc3_loss = criterion(attack_flow, source_flow)#*cond_normal[l_ct]
    #l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        #decoder_lip_sum += criterion(attack_out, source_out) #*cond_normal[l_ct]
        #l_ct += 1
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    #loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(normalized_attacked, attack_flow)
    loss_to_maximize = wasserstein_distance(z1, z2)* wasserstein_distance(source_flow, attack_flow)

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, attack_flow, source_flow




def get_lgr_cos_kf(normalized_attacked, source_im):
    #model = torch.jit.trace(model, normalized_attacked) 
    #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
    #model = torch.jit.trace(model, normalized_attacked) 
    #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)
    
    attack_flow = normalized_attacked
    source_flow = source_im
    #encoder_lip_sum = 0
    #l_ct = 0
    for layer in model.encoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        #encoder_lip_sum += criterion(attack_out, source_out)#*cond_normal[l_ct]
        #l_ct += 1
        attack_flow = attack_out
        source_flow = source_out

    ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?
    mu1, logvar1 = model.fc1(ae_perturbed_embeds), model.fc2(ae_perturbed_embeds)
    std1 = logvar1.mul(0.5).exp_()
    esp1 = torch.randn(*mu1.size()).to(device)
    z1 = mu1 + std1 * esp1

    train_embeds = model.encoder(source_im.to(device))
    mu2, logvar2 = model.fc1(train_embeds), model.fc2(train_embeds)
    std2 = logvar2.mul(0.5).exp_()
    esp2 = torch.randn(*mu2.size()).to(device)
    z2 = mu2 + std2 * esp2


    #mu_loss = criterion(mu1, mu2)#*cond_normal[l_ct]
    #l_ct += 1

    #rep_loss = criterion(std1 * esp1, std2 * esp2)#*cond_normal[l_ct]
    #l_ct += 1


    attack_flow = model.fc3(z1)
    source_flow = model.fc3(z2)
    #decoder_lip_sum = 0

    #fc3_loss = criterion(attack_flow, source_flow)#*cond_normal[l_ct]
    #l_ct += 1


    for layer in model.decoder:
        attack_out = layer(attack_flow)
        source_out = layer(source_flow)
        #decoder_lip_sum += criterion(attack_out, source_out) #*cond_normal[l_ct]
        #l_ct += 1
        attack_flow = attack_out
        source_flow = source_out


    #lipschitzt_loss_encoder = criterion(z1, z2) 
    #lipschitzt_loss_decoder = criterion(normalized_attacked, adv_gen)  
    #print("attack_flow.shape", attack_flow.shape)
    #print("source_flow.shape", source_flow.shape)
    #loss_to_maximize =  (encoder_lip_sum + mu_loss + rep_loss + fc3_loss +  decoder_lip_sum) * criterion(normalized_attacked, attack_flow)
    loss_to_maximize = (1-cos(z1, z2))**2  * (1-cos(source_flow, attack_flow))**2

    #loss_to_maximize =  decoder_lip_sum * criterion(normalized_attacked, adv_gen)

    return loss_to_maximize, attack_flow, source_flow






def run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen):

    print(f"Step {step}, Loss: {total_loss.item()}, distortion L-2: {l2_distortion}, distortion L-inf: {l_inf_distortion}, outputChange: {instability}, deviation: {deviation}")
    print()
    print("attack type", attck_type)    
    adv_div_list.append(deviation.item())
    with torch.no_grad():
        fig, ax = plt.subplots(1, 3, figsize=(10, 10))
        ax[0].imshow(normalized_attacked[0].permute(1, 2, 0).cpu().numpy())
        ax[0].set_title('Attacked Image')
        ax[0].axis('off')

        ax[1].imshow(scaled_noise[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        ax[1].set_title('Noise')
        ax[1].axis('off')

        ax[2].imshow(adv_gen[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        ax[2].set_title('Attack reconstruction')
        ax[2].axis('off')
        plt.show()
        plt.savefig("betaVAEandTcVAE/robustness_eval_saves_univ/optimization_time_plots/"+model_type+"_beta_"+str(beta_value)+"_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+".png")

    optimized_noise = scaled_noise
    torch.save(optimized_noise, "betaVAEandTcVAE/robustness_eval_saves_univ/relevancy_test/layerwise_maximum_damage_attack/"+model_type+"_beta_"+str(beta_value)+"_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+".pt")
    np.save("betaVAEandTcVAE/robustness_eval_saves_univ/adversarial_div_convergence/"+model_type+"_beta_"+str(beta_value)+"_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+".npy", adv_div_list)
    plt.close()

    plt.plot(adv_div_list)
    plt.savefig('betaVAEandTcVAE/robustness_eval_saves_univ/run_time_div/deviation_attack_type_'+attck_type+'_desired_norm_l_inf_'+str(desired_norm_l_inf)+'_.png')
    plt.close()


'''def run_time_plots_and_saves_weighted(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen, layer_weights_un):

    print(f"Step {step}, Loss: {total_loss.item()}, distortion L-2: {l2_distortion}, distortion L-inf: {l_inf_distortion}, outputChange: {instability}, deviation: {deviation}")
    print()
    #print("layer_weights_un", layer_weights_un)
    #print()
    adv_div_list.append(deviation.item())
    with torch.no_grad():
        fig, ax = plt.subplots(1, 3, figsize=(10, 10))
        ax[0].imshow(normalized_attacked[0].permute(1, 2, 0).cpu().numpy())
        ax[0].set_title('Attacked Image')
        ax[0].axis('off')

        ax[1].imshow(scaled_noise[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        ax[1].set_title('Noise')
        ax[1].axis('off')

        ax[2].imshow(adv_gen[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        ax[2].set_title('Attack reconstruction')
        ax[2].axis('off')
        plt.show()
        plt.savefig("/home/luser/autoencoder_attacks/robustness_eval_saves_univ/optimization_time_plots/VAE_beta_"+str(beta_value)+"_attack_type"+str(attck_type)+".png")

    optimized_noise = scaled_noise
    torch.save(optimized_noise, "/home/luser/autoencoder_attacks/robustness_eval_saves_univ/relevancy_test/layerwise_maximum_damage_attack/VAE_beta_"+str(beta_value)+"_attack_type"+str(attck_type)+".pt")
    np.save("/home/luser/autoencoder_attacks/robustness_eval_saves_univ/adversarial_div_convergence/VAE_beta_"+str(beta_value)+"_attack_type"+str(attck_type)+".npy", adv_div_list)'''

batch_size = 50
#print("1source_ims.shape", source_ims.shape)
#source_ims = source_ims.reshape(-1,50, 3, 64, 64)
#print("2 source_ims.shape", source_ims.shape)





if(attck_type == "latent_l2"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            #print("source_im.shape", source_im.shape)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            #print("scaled_noise.shape", scaled_noise.shape)
            attacked = (source_im + scaled_noise)
            #print("attacked.shape", attacked.shape)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize = get_latent_space_l2_loss(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #if step % 10000 == 0:
        with torch.no_grad():

            adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
            source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "latent_l2_kf_SS"):
    adv_div_list = []
    for step in range(num_steps):
        #for idx, (image, label) in enumerate(testLoader):

        optimizer.zero_grad()
        source_im, label = images, label
        #print("source_im.shape", source_im.shape)
        #print("noise_addition.shape", noise_addition.shape)
        #print()
        normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)


        loss_to_maximize = get_latent_space_l2_loss(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        #optimizer.zero_grad()

        if step % 300 == 0:
            with torch.no_grad():

                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                #l2_distortion = torch.norm(scaled_noise, p=2)

                l2_distortion = torch.norm(noise_addition, p=2)
                l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

                deviation = torch.norm(adv_gen - source_recon, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, noise_addition, adv_gen)




if(attck_type == "latent_wass_kf_SS"):
    adv_div_list = []
    for step in range(num_steps):
        #for idx, (image, label) in enumerate(testLoader):

        optimizer.zero_grad()
        source_im, label = images, label
        #print("source_im.shape", source_im.shape)
        #print("noise_addition.shape", noise_addition.shape)
        #print()
        normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)


        loss_to_maximize = get_latent_space_wass_loss(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        #optimizer.zero_grad()

        if step % 300 == 0:
            with torch.no_grad():

                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                #l2_distortion = torch.norm(scaled_noise, p=2)

                l2_distortion = torch.norm(noise_addition, p=2)
                l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

                deviation = torch.norm(adv_gen - source_recon, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, noise_addition, adv_gen)



if(attck_type == "latent_cos_kf_SS"):
    adv_div_list = []
    for step in range(num_steps):
        #for idx, (image, label) in enumerate(testLoader):

        optimizer.zero_grad()
        source_im, label = images, label
        #print("source_im.shape", source_im.shape)
        #print("noise_addition.shape", noise_addition.shape)
        #print()
        normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)


        loss_to_maximize = get_latent_space_cos_loss(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        #optimizer.zero_grad()

        if step % 300 == 0:
            with torch.no_grad():
                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 

                l2_distortion = torch.norm(noise_addition, p=2)
                l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

                deviation = torch.norm(adv_gen - source_recon, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, noise_addition, adv_gen)



if(attck_type == "output_l2_kf_SS"):
    adv_div_list = []
    for step in range(num_steps):
        #for idx, (image, label) in enumerate(testLoader):

        optimizer.zero_grad()
        source_im, label = images, label
        #print("source_im.shape", source_im.shape)
        #print("noise_addition.shape", noise_addition.shape)
        #print()
        normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)
        #normalized_attacked = torch.sigmoid(normalized_attacked) 


        loss_to_maximize = get_output_space_l2_loss_kf(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        #optimizer.zero_grad()

        if step % 300 == 0:
            with torch.no_grad():

                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                #l2_distortion = torch.norm(scaled_noise, p=2)

                l2_distortion = torch.norm(noise_addition, p=2)
                l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

                deviation = torch.norm(adv_gen - source_recon, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, noise_addition, adv_gen)





if(attck_type == "output_cos_kf_SS"):
    adv_div_list = []
    for step in range(num_steps):
        #for idx, (image, label) in enumerate(testLoader):

        optimizer.zero_grad()
        source_im, label = images, label

        normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

        #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        #scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        #attacked = (source_im + scaled_noise)
        #normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        #loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)
        loss_to_maximize, adv_gen, source_recon = get_oa_cos_kf(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        #optimizer.zero_grad()
        if step%300==0:
            with torch.no_grad():

                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                #l2_distortion = torch.norm(scaled_noise, p=2)

                l2_distortion = torch.norm(noise_addition, p=2)
                l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

                deviation = torch.norm(adv_gen - source_recon, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, noise_addition, adv_gen)





if(attck_type == "output_wass_kf_SS"):
    adv_div_list = []
    for step in range(num_steps):
        #for idx, (image, label) in enumerate(testLoader):

        optimizer.zero_grad()
        source_im, label = images, label
        #print("source_im.shape", source_im.shape)
        #print("noise_addition.shape", noise_addition.shape)
        #print()
        normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)


        loss_to_maximize = get_output_space_wass_loss_kf(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        #optimizer.zero_grad()

        if step % 300 == 0:
            with torch.no_grad():

                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                #l2_distortion = torch.norm(scaled_noise, p=2)

                l2_distortion = torch.norm(noise_addition, p=2)
                l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

                deviation = torch.norm(adv_gen - source_recon, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, noise_addition, adv_gen)





if(attck_type == "lgr_l2_kf_SS"):
    adv_div_list = []
    all_adv_div_list = []

    for step in range(num_steps):
        #for idx, (image, label) in enumerate(testLoader):

        optimizer.zero_grad()
        source_im, label = images, label

        normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

        normalized_attacked = torch.sigmoid(normalized_attacked) 

        #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        #scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        #attacked = (source_im + scaled_noise)
        #normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        #loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)
        loss_to_maximize, adv_gen, source_recon = get_lgr_l2_kf(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        #optimizer.zero_grad()
        if step%300==0:
            with torch.no_grad():

                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                #l2_distortion = torch.norm(scaled_noise, p=2)

                l2_distortion = torch.norm(noise_addition, p=2)
                l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

                deviation = torch.norm(adv_gen - source_recon, p=2)
                print("cur deviation", deviation)

                all_adv_div_list.append(deviation.item())
                updateQuest = deviation >= max(all_adv_div_list)
                print("updateQuest", updateQuest)

                if updateQuest:
                    get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, noise_addition, adv_gen)





if(attck_type == "lgr_wass_kf_SS"):
    adv_div_list = []
    all_adv_div_list = []

    for step in range(num_steps):
        #for idx, (image, label) in enumerate(testLoader):

        optimizer.zero_grad()
        source_im, label = images, label

        normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

        normalized_attacked = torch.sigmoid(normalized_attacked) 

        #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        #scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        #attacked = (source_im + scaled_noise)
        #normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        #loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)
        loss_to_maximize, adv_gen, source_recon = get_lgr_wass_kf(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        #optimizer.zero_grad()

        if step%300==0:
            with torch.no_grad():

                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                #l2_distortion = torch.norm(scaled_noise, p=2)

                l2_distortion = torch.norm(noise_addition, p=2)
                l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

                deviation = torch.norm(adv_gen - source_recon, p=2)
                print("cur deviation", deviation)

                all_adv_div_list.append(deviation.item())
                updateQuest = deviation >= max(all_adv_div_list)
                print("updateQuest", updateQuest)

                if updateQuest:
                    get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, noise_addition, adv_gen)



if(attck_type == "lgr_cos_kf_SS"):
    adv_div_list = []
    all_adv_div_list = []

    for step in range(num_steps):
        #for idx, (image, label) in enumerate(testLoader):

        optimizer.zero_grad()
        source_im, label = images, label

        normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

        normalized_attacked = torch.sigmoid(normalized_attacked) 

        #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        #scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        #attacked = (source_im + scaled_noise)
        #normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        #loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)
        loss_to_maximize, adv_gen, source_recon = get_lgr_cos_kf(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        #optimizer.zero_grad()      
        if step%300==0:
            with torch.no_grad():

                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                #l2_distortion = torch.norm(scaled_noise, p=2)

                l2_distortion = torch.norm(noise_addition, p=2)
                l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

                deviation = torch.norm(adv_gen - source_recon, p=2)
                print("cur deviation", deviation)

                all_adv_div_list.append(deviation.item())
                updateQuest = deviation >= max(all_adv_div_list)
                print("updateQuest", updateQuest)

                if updateQuest:
                    get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, noise_addition, adv_gen)








if(attck_type == "latent_l2_mcmc"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            #print("source_im.shape", source_im.shape)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            #print("scaled_noise.shape", scaled_noise.shape)
            attacked = (source_im + scaled_noise)
            #print("attacked.shape", attacked.shape)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_latent_l2_mcmc(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #if step % 10000 == 0:
        with torch.no_grad():

            #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
            #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "output_wass_mcmc"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            #print("source_im.shape", source_im.shape)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            #print("scaled_noise.shape", scaled_noise.shape)
            attacked = (source_im + scaled_noise)
            #print("attacked.shape", attacked.shape)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_output_wass_mcmc(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #if step % 10000 == 0:
        with torch.no_grad():

            #adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
            #source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "latent_cosine"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            #loss_to_maximize = get_latent_space_cosine_loss(normalized_attacked, source_im)
            loss_to_maximize = (get_latent_space_cosine_loss(normalized_attacked, source_im)-1.0)**2 

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #if step % 10000 == 0:
        with torch.no_grad():

            adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
            source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "latent_wasserstein"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)

            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_latent_space_wasserstein_loss(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "latent_SKL"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)

            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_latent_space_SKL_loss(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "weighted_combi_l2"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "weighted_combi_l2_enco"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_enco(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():
            adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
            source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "weighted_combi_cosine_enco"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

            loss_to_maximize = get_weighted_combinations_cosine_enco(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
            source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "weighted_combi_l2_tw"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)

if(attck_type == "weighted_combinations_wasserstein"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_wasserstein(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)

if(attck_type == "weighted_combinations_SKL"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            ae_perturbed_embeds = model.encoder(normalized_attacked) # some confusion here. Why are you doin whatever you doing . Should not the input be  data + noise_outputs ?

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_SKL(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked)
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)



            



if(attck_type == "weighted_combi_k_eq_latent_l2"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_k_eq_latent_l2(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)

if(attck_type == "weighted_combi_k_eq_latent_wasserstein"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_k_eq_latent_wasserstein(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "weighted_combi_k_eq_latent_SKL"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_k_eq_latent_SKL(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "weighted_combi_k_eq_latent_cosine"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_k_eq_latent_cos(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "output_attack_l2"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = output_attack_l2(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "output_attack_wasserstein"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = output_attack_wasserstein(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "output_attack_SKL"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = output_attack_SKL(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "output_attack_cosine"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = output_attack_cosine(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #if step % 10000 == 0:
        with torch.no_grad():

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "aclmd_l2"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "aclmd_l2f_cond" or attck_type=="aclmd_l2_cond_dir"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "aclmd_l2a_cond"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "grill_l2_kf_SS"):
    adv_div_list = []
    all_adv_div_list = []

    for step in range(num_steps):
        #for idx, (image, label) in enumerate(testLoader):

        optimizer.zero_grad()
        source_im, label = images, label

        normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

        normalized_attacked = torch.sigmoid(normalized_attacked) 

        #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        #scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        #attacked = (source_im + scaled_noise)
        #normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        #loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)
        loss_to_maximize, adv_gen, source_recon = get_grill_l2_kf(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        #optimizer.zero_grad()
        if step%300==0:
            with torch.no_grad():

                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                #l2_distortion = torch.norm(scaled_noise, p=2)

                l2_distortion = torch.norm(noise_addition, p=2)
                l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

                deviation = torch.norm(adv_gen - source_recon, p=2)
                print("cur deviation", deviation)

                all_adv_div_list.append(deviation.item())
                updateQuest = deviation >= max(all_adv_div_list)
                print("updateQuest", updateQuest)

                if updateQuest:
                    get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, noise_addition, adv_gen)


if(attck_type == "grill_l2_kfNw"):
    adv_div_list = []
    all_adv_div_list = []

    all_condition_nums = np.random.rand(33) 

    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):

            optimizer.zero_grad()
            source_im, label = image.to(device), label.to(device)

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            #normalized_attacked = torch.sigmoid(normalized_attacked) 

            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            #attacked = (source_im + scaled_noise)
            #normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            #loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)
            loss_to_maximize, adv_gen, source_recon, lossItems = get_grill_l2_kfNw(normalized_attacked, source_im, all_condition_nums)
            #print("lossItems", len(lossItems)) 
            all_condition_nums = lossItems/ lossItems.sum()
            #print("all_condition_nums", len(all_condition_nums))
            #print("all_condition_nums.sum()", all_condition_nums.sum())
            #print("all_condition_nums", all_condition_nums)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #optimizer.zero_grad()

        with torch.no_grad():

            adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
            source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            #l2_distortion = torch.norm(scaled_noise, p=2)

            l2_distortion = torch.norm(noise_addition, p=2)
            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            deviation = torch.norm(adv_gen - source_recon, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)

            if updateQuest:
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, noise_addition, adv_gen)


if(attck_type == "grill_l2_kf_mcmc"):
    adv_div_list = []
    all_adv_div_list = []

    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):

            optimizer.zero_grad()
            source_im, label = image.to(device), label.to(device)

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            normalized_attacked = torch.sigmoid(normalized_attacked) 

            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            #attacked = (source_im + scaled_noise)
            #normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            #loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)
            loss_to_maximize, adv_gen, source_recon = get_grill_l2_kf_mcmc(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #optimizer.zero_grad()

        with torch.no_grad():

            adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
            source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            #l2_distortion = torch.norm(scaled_noise, p=2)

            l2_distortion = torch.norm(noise_addition, p=2)
            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            deviation = torch.norm(adv_gen - source_recon, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)

            if updateQuest:
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, noise_addition, adv_gen)


if(attck_type == "output_l2_kf_mcmc"):
    adv_div_list = []
    all_adv_div_list = []

    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):

            optimizer.zero_grad()
            source_im, label = image.to(device), label.to(device)

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            normalized_attacked = torch.sigmoid(normalized_attacked) 

            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            #attacked = (source_im + scaled_noise)
            #normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            #loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)
            loss_to_maximize, adv_gen, source_recon = get_output_l2_kf_mcmc(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #optimizer.zero_grad()

        with torch.no_grad():

            adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
            source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            #l2_distortion = torch.norm(scaled_noise, p=2)

            l2_distortion = torch.norm(noise_addition, p=2)
            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            deviation = torch.norm(adv_gen - source_recon, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)

            if updateQuest:
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, noise_addition, adv_gen)


if(attck_type == "grill_l2_kfd"):
    adv_div_list = []
    all_adv_div_list = []

    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):

            optimizer.zero_grad()
            source_im, label = image.to(device), label.to(device)

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            #attacked = (source_im + scaled_noise)
            #normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            #loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)
            loss_to_maximize, adv_gen, source_recon = get_grill_l2_kfd(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #optimizer.zero_grad()

        with torch.no_grad():

            adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
            source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            #l2_distortion = torch.norm(scaled_noise, p=2)

            l2_distortion = torch.norm(noise_addition, p=2)
            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            deviation = torch.norm(adv_gen - source_recon, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)

            if updateQuest:
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, noise_addition, adv_gen)




if(attck_type == "grill_l2_kfe"):
    adv_div_list = []
    all_adv_div_list = []

    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):

            optimizer.zero_grad()
            source_im, label = image.to(device), label.to(device)

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            #attacked = (source_im + scaled_noise)
            #normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            #loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)
            loss_to_maximize, adv_gen, source_recon = get_grill_l2_kfe(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #optimizer.zero_grad()

        with torch.no_grad():

            adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
            source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            #l2_distortion = torch.norm(scaled_noise, p=2)

            l2_distortion = torch.norm(noise_addition, p=2)
            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            deviation = torch.norm(adv_gen - source_recon, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)

            if updateQuest:
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, noise_addition, adv_gen)


if(attck_type == "grill_cos_kf_SS"):
    adv_div_list = []
    for step in range(num_steps):
        #for idx, (image, label) in enumerate(testLoader):

        optimizer.zero_grad()
        source_im, label = images, label

        normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

        #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        #scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        #attacked = (source_im + scaled_noise)
        #normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        #loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)
        loss_to_maximize, adv_gen, source_recon = get_grill_cos_kf(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        #optimizer.zero_grad()

        if step%300==0:
            with torch.no_grad():

                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                #l2_distortion = torch.norm(scaled_noise, p=2)

                l2_distortion = torch.norm(noise_addition, p=2)
                l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

                deviation = torch.norm(adv_gen - source_recon, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, noise_addition, adv_gen)




if(attck_type == "grill_cos_kf_mcmc"):
    adv_div_list = []
    all_adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):

            optimizer.zero_grad()
            source_im, label = image.to(device), label.to(device)

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            #attacked = (source_im + scaled_noise)
            #normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            #loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)
            loss_to_maximize, adv_gen, source_recon = get_grill_cos_kf_mcmc(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #optimizer.zero_grad()

        with torch.no_grad():

            adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
            source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            #l2_distortion = torch.norm(scaled_noise, p=2)

            l2_distortion = torch.norm(noise_addition, p=2)
            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            deviation = torch.norm(adv_gen - source_recon, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)

            if updateQuest:
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, noise_addition, adv_gen)





if(attck_type == "grill_wass_kf_SS"):
    adv_div_list = []
    for step in range(num_steps):
        #for idx, (image, label) in enumerate(testLoader):

        optimizer.zero_grad()
        source_im, label = images, label

        normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

        #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        #scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
        #attacked = (source_im + scaled_noise)
        #normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

        #loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)
        loss_to_maximize, adv_gen, source_recon = get_grill_wass_kf(normalized_attacked, source_im)

        total_loss = -1 * loss_to_maximize

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

        #optimizer.zero_grad()
        if step%300==0:
            with torch.no_grad():

                adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
                source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

                instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
                #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                #l2_distortion = torch.norm(scaled_noise, p=2)

                l2_distortion = torch.norm(noise_addition, p=2)
                l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

                deviation = torch.norm(adv_gen - source_recon, p=2)
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, noise_addition, adv_gen)



if(attck_type == "grill_l2_kfw"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):

            optimizer.zero_grad()
            source_im, label = image.to(device), label.to(device)

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            #attacked = (source_im + scaled_noise)
            #normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            #loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)
            loss_to_maximize, adv_gen, source_recon = get_grill_l2_kf(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #optimizer.zero_grad()

        with torch.no_grad():

            adv_gen, adv_recon_loss, adv_kl_losses = model(normalized_attacked)
            source_recon, source_recon_loss, source_recon_kl_losses = model(source_im)

            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            #l2_distortion = torch.norm(scaled_noise, p=2)

            l2_distortion = torch.norm(noise_addition, p=2)
            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, noise_addition, adv_gen)



if(attck_type == "aclmd_l2a_cond_grad_norm"):
    adv_div_list = []
    all_grad_norms = []
    for step in range(num_steps):
        count_batch = 0
        epoch_avg_grad_norm = 0
        for idx, (image, label) in enumerate(testLoader):
            count_batch+=1
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            grad_l2_norm = torch.norm(noise_addition.grad, p=2)
            epoch_avg_grad_norm = (epoch_avg_grad_norm + grad_l2_norm) /count_batch

            optimizer.step()
            optimizer.zero_grad()


        print("grad_l2_norm", epoch_avg_grad_norm)
        all_grad_norms.append(epoch_avg_grad_norm.item())
        np.save("/home/luser/autoencoder_attacks/train_aautoencoders/all_grad_norms/"+model_type+"grad_norms_list_"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_step_"+str(step)+".npy", all_grad_norms)

        with torch.no_grad():
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)




if(attck_type == "aclmd_l2a_cond1"):
    adv_div_list = []
    for step in range(num_steps):
        total_epoch_div = 0
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #l2_distance_per_image = torch.norm(normalized_attacked - adv_gen, p=2, dim=[1, 2, 3])  # Shape: [50]

            deviation = torch.norm(adv_gen - source_recon, p=2, dim=[1, 2, 3])
            deviation = torch.mean(deviation)
            total_epoch_div = total_epoch_div + deviation

        with torch.no_grad():
            mean_div = total_epoch_div/(idx+1)
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            #deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, mean_div, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "aclmd_l2a_cond1_mcmc"):
    adv_div_list = []
    for step in range(num_steps):
        total_epoch_div = 0
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond_mcmc(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            deviation = torch.norm(adv_gen - source_recon, p=2, dim=[1, 2, 3])
            deviation = torch.mean(deviation)
            total_epoch_div = total_epoch_div + deviation

        with torch.no_grad():
            mean_div = total_epoch_div/(idx+1)
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            #deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, mean_div, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "aclmd_equal_weights"):
    adv_div_list = []
    for step in range(num_steps):
        total_epoch_div = 0
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            deviation = torch.norm(adv_gen - source_recon, p=2, dim=[1, 2, 3])
            deviation = torch.mean(deviation)
            total_epoch_div = total_epoch_div + deviation

        with torch.no_grad():
            mean_div = total_epoch_div/(idx+1)
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            #deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, mean_div, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "aclmd_equal_weights_mcmc"):
    adv_div_list = []
    for step in range(num_steps):
        total_epoch_div = 0
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond_mcmc(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            deviation = torch.norm(adv_gen - source_recon, p=2, dim=[1, 2, 3])
            deviation = torch.mean(deviation)
            total_epoch_div = total_epoch_div + deviation

        with torch.no_grad():
            mean_div = total_epoch_div/(idx+1)
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            #deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, mean_div, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "aclmd_random_weights"):
    adv_div_list = []
    for step in range(num_steps):
        total_epoch_div = 0
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            deviation = torch.norm(adv_gen - source_recon, p=2, dim=[1, 2, 3])
            deviation = torch.mean(deviation)
            total_epoch_div = total_epoch_div + deviation

        with torch.no_grad():
            mean_div = total_epoch_div/(idx+1)
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            #deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, mean_div, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "aclmd_random_weights_mcmc"):
    adv_div_list = []
    for step in range(num_steps):
        total_epoch_div = 0
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond_mcmc(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            deviation = torch.norm(adv_gen - source_recon, p=2, dim=[1, 2, 3])
            deviation = torch.mean(deviation)
            total_epoch_div = total_epoch_div + deviation

        with torch.no_grad():
            mean_div = total_epoch_div/(idx+1)
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            #deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, mean_div, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "aclmd_l2a_mcmc"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_mcmc(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "aclmd_l2_cond_nonlin"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_l2_cond(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)



if(attck_type == "aclmd_wasserstein_cond"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_aclmd_wasserstein_cond(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "aclmd_SKL_cond"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_aclmd_SKL_cond(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "aclmd_cosine_cond"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_aclmd_cos_cond(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)




if(attck_type == "aclmd_wasserstein"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_wasserstein(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)




if(attck_type == "aclmd_SKL"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_SKL(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)


if(attck_type == "aclmd_cosine"):
    adv_div_list = []
    for step in range(num_steps):
        for idx, (image, label) in enumerate(testLoader):
            source_im, label = image.to(device), label.to(device)
            current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm) 
            attacked = (source_im + scaled_noise)
            normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            loss_to_maximize, adv_gen, source_recon = get_weighted_combinations_l2_aclmd_cos(normalized_attacked, source_im)

            total_loss = -1 * loss_to_maximize

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            instability = criterion(adv_gen, source_recon) / criterion(source_im, normalized_attacked) 
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)
            deviation = torch.norm(adv_gen - source_recon, p=2)
            get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, instability, deviation, normalized_attacked, scaled_noise, adv_gen)
