import torch
import torch.nn as nn
from model import AutoEncoder
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import random
import matplotlib.ticker as ticker

from torchvision import datasets, transforms
import os
import pandas as pd
import torch.optim as optim

'''






conda deactivate
conda deactivate
conda deactivate
export CUDA_VISIBLE_DEVICES=4
cd NVAE/
source nvaeenv1/bin/activate
cd ..
cd illcond/
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "la_l2_kf" --desired_norm_l_inf 0.037 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "la_wass_kf_cr" --desired_norm_l_inf 0.037 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "la_cos_kf_cr" --desired_norm_l_inf 0.037 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "grill_l2_kf" --desired_norm_l_inf 0.037 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "grill_wass_kf" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "grill_cos_kf" --desired_norm_l_inf 0.09 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint





############################################################ Structured ablations ######################################################################################################
conda deactivate
conda deactivate
conda deactivate
export CUDA_VISIBLE_DEVICES=7
cd NVAE/
source nvaeenv1/bin/activate
cd ..
cd illcond/
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "grill_wass_kf_noLast" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "grill_wass_kf_allSum" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "grill_wass_kf_5p" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "grill_wass_kf_10p" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "grill_wass_kf_30p" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "grill_wass_kf_70p" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "grill_wass_kf_30pRev" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "grill_wass_kf_50pRev" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "grill_wass_kf_70pRev" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "grill_wass_kf_90pRev" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint







#######################################################################################################################  Layer Losses tracking    #########################################################################################################################################################################################################################################################
conda deactivate
conda deactivate
conda deactivate
export CUDA_VISIBLE_DEVICES=4
cd NVAE/
source nvaeenv1/bin/activate
cd ..
cd illcond/
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "la_l2_kf_layerLosses" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "grill_wass_kf_layerLosses" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "grill_l2_kf_layerLosses" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint


################################################################################################################################################################################################################################################################################################################################################################################



post_reviews


########################################################################   Experiments with layer weights   ##########################################################################################################################################################################################################################################################################################

conda deactivate
conda deactivate
conda deactivate
export CUDA_VISIBLE_DEVICES=7
cd NVAE/
source nvaeenv1/bin/activate
cd ..
cd illcond/
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "grill_l2_mcmc" --desired_norm_l_inf 0.035 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoin
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "grill_l2_mcmc_eqwts" --desired_norm_l_inf 0.035 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "grill_l2_mcmc_rndwts" --desired_norm_l_inf 0.035 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint



# extras# extras # extras # extras # extras # extras 
conda deactivate
conda deactivate
conda deactivate
export CUDA_VISIBLE_DEVICES=2
cd NVAE/
source nvaeenv1/bin/activate
cd ..
cd illcond/
python nvae/nvae_all_kids_of_attacks_universal.py --attck_type "grill_wass_kf_cr" --desired_norm_l_inf 0.025 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint


'''

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import argparse

parser = argparse.ArgumentParser(description='Adversarial attack on VAE')
parser.add_argument('--attck_type', type=str, default="lip", help='Segment index')
parser.add_argument('--desired_norm_l_inf', type=float, default="lip", help='Segment index')
parser.add_argument('--data_directory', type=str, default=0, help='data directory')
parser.add_argument('--nvae_checkpoint_path', type=str, default=0, help='nvae checkpoint directory')


args = parser.parse_args()


attck_type = args.attck_type
desired_norm_l_inf = args.desired_norm_l_inf
data_directory = args.data_directory
nvae_checkpoint_path = args.nvae_checkpoint_path


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

arch_instance = utils.get_arch_cells(args.arch_instance)  
model = AutoEncoder(args, None, arch_instance)


model.load_state_dict(checkpoint['state_dict'], strict=False)
model = model.cuda()
model.eval()


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

########delete some stuff to clear memory
del celeba_data
del train_set
del test_set
del trainLoader


# Initialize a list to store batch tensors
batch_list = []

for idx, (source_im, _) in enumerate(testLoader):
    source_im, _ = source_im.cuda(), _
    batch_list.append(source_im)  # Store batch in a list
    #if(len(batch_list)==5):
        #break


big_tensor = torch.stack(batch_list)  

mi, ma = big_tensor.min(), big_tensor.max()





def hook_fn(module, input, output):
    print(f"Layer: {module.__class__.__name__}")
    print(f"Output shape: {output.shape}")

print("what is going in", source_im.shape)
source_im = source_im[0].unsqueeze(0)

noise_addition = (torch.randn_like(source_im) * 0.2).cuda()
noise_addition = noise_addition.clone().detach().requires_grad_(True)
optimizer = optim.Adam([noise_addition], lr=0.0001)



criterion = nn.MSELoss()


layerwise_outputs = {}

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

def wasserstein_distance_cr(tensor_a, tensor_b):

    wass_sum = 0
    for i in range(len(tensor_a)):
        tensor_a_flat = torch.flatten(tensor_a[i])
        tensor_b_flat = torch.flatten(tensor_b[i])
        tensor_a_sorted, _ = torch.sort(tensor_a_flat)
        tensor_b_sorted, _ = torch.sort(tensor_b_flat)    
        wasserstein_dist = torch.mean(torch.abs(tensor_a_sorted - tensor_b_sorted))
        wass_sum += wasserstein_dist

    return wass_sum

def cos_cr(a, b):
    cos_sum = 0
    for i in range(len(a)):
        a_fl = a[i].view(-1)
        b_fl = b[i].view(-1)
        a_fl = F.normalize(a_fl, dim=0)
        b_fl = F.normalize(b_fl, dim=0)
        cos_sum += (a_fl * b_fl).sum()
    return cos_sum


def compute_mean_and_variance(tensor):
    flattened_tensor = torch.flatten(tensor)  # Flatten the tensor
    mean = torch.mean(flattened_tensor)  # Compute mean
    variance = torch.var(flattened_tensor, unbiased=False)  # Compute variance (unbiased=False for population variance)
    return mean, variance

def get_symmetric_KLDivergence(input1, input2):
    mu1, var1 = compute_mean_and_variance(input1)
    mu2, var2 = compute_mean_and_variance(input2)
    
    kl_1_to_2 = torch.log(var2 / var1) + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5
    kl_2_to_1 = torch.log(var1 / var2) + (var2 + (mu2 - mu1) ** 2) / (2 * var1) - 0.5
    
    symmetric_kl = (kl_1_to_2 + kl_2_to_1) / 2
    return symmetric_kl

def get_symmetric_KLDivergence(input1, input2):
    mu1, var1 = compute_mean_and_variance(input1)
    mu2, var2 = compute_mean_and_variance(input2)
    
    kl_1_to_2 = torch.log(var2 / var1) + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5
    kl_2_to_1 = torch.log(var1 / var2) + (var2 + (mu2 - mu1) ** 2) / (2 * var1) - 0.5
    
    symmetric_kl = (kl_1_to_2 + kl_2_to_1) / 2
    return symmetric_kl

def get_symmetric_KLDivergence_agg(input1, input2):
    mu1, var1 = compute_mean_and_variance(input1)
    mu2, var2 = compute_mean_and_variance(input2)
    
    var1 = var1 + 1e-6
    var2 = var2 + 1e-6

    kl_1_to_2 = torch.log(var2 / var1) + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5
    kl_2_to_1 = torch.log(var1 / var2) + (var2 + (mu2 - mu1) ** 2) / (2 * var1) - 0.5
    
    symmetric_kl = (kl_1_to_2 + kl_2_to_1) / 2
    return symmetric_kl


def encoder_hook_fn(module, input, output):
    layerwise_outputs[module] = output

# Register hooks for encoder layers
encoder_hook_handles = []

for name, layer in model.enc_tower.named_modules():
    handle = layer.register_forward_hook(encoder_hook_fn)
    encoder_hook_handles.append(handle)



def run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, scaled_noise, adv_gen):
    with torch.no_grad():
        print(f"Step {step}, Loss: {total_loss.item()}, distortion L-2: {l2_distortion}, distortion L-inf: {l_inf_distortion}, deviation: {deviation}, recon mse: {mase_dev}")
        print()
        print("attack type", attck_type)    
        adv_div_list.append(deviation.item())
        adv_mse_list.append(mase_dev.item())
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
        plt.savefig("nvae/optimization_time_plots/NVAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.png")

    optimized_noise = scaled_noise
    torch.save(optimized_noise, "nvae/univ_attack_storage/NVAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.pt")
    print("adv_div_list", adv_div_list)
    np.save("nvae/deviation_store/NVAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", adv_div_list)
    plt.figure(figsize=(8, 5))
    plt.plot(adv_div_list, marker='o')
    plt.xlabel('Step')
    plt.ylabel('Deviation')
    plt.title(f'Deviation over Steps: {attck_type}, L∞={desired_norm_l_inf}')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("nvae/optimization_time_plots/div_NVAE_attack_type"+str(attck_type)+"_step_"+str(step)+"_norm_bound_"+str(desired_norm_l_inf)+"_.png")
    plt.show()



def l2_batch_agg_loss(adv_output, orig_output):
    loss = 0
    for adv_output_k, orig_output_k in zip(adv_output, orig_output):
        loss += criterion(adv_output_k, orig_output_k)
    return loss


if(attck_type == "grill_l2_kf"):
    all_condition_nums = np.load('nvae/saved_cond_nums/nvae_cond_nums.npy')

    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())


    adv_div_list = []
    all_adv_div_list = []
    adv_mse_list = []
    allAvgEpochLosses = []
    for step in range(100):
        avgEpochLoss = 0
        prevAvgEpochLoss = 0
        betachCounter = 0

        for source_im in big_tensor:

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            layerwise_outputs.clear()
            #print("layerwise_outputs", layerwise_outputs)
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            #print("adv_layerwise_outputs", adv_layerwise_outputs)
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()

            loss = 0
            counter = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                #print("adv_output.shape", adv_output.shape)
                #print("orig_output.shape", orig_output.shape)

                #loss += l2_batch_agg_loss(adv_output, orig_output) * cond_nums_normalized[counter]
                loss += criterion(adv_output, orig_output) * cond_nums_normalized[counter]

                counter+=1
            #print("counter", counter)
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                #lat_loss += l2_batch_agg_loss(adv_latent_reps[i], orig_latent_reps[i])
                lat_loss += criterion(adv_latent_reps[i], orig_latent_reps[i])

            total_loss = -1 * loss * lat_loss

            total_loss.backward()
            optimizer.step()

            

            with torch.no_grad():
                totLossMag =  loss + lat_loss
                avgEpochLoss = avgEpochLoss + totLossMag.item()
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #if step % 400 == 0:
            betachCounter = betachCounter + 1

        with torch.no_grad():
            avgEpochLoss = avgEpochLoss/betachCounter
            allAvgEpochLosses.append(avgEpochLoss)
            #print("all_adv_div_list", all_adv_div_list)
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
            l2_distortion = torch.norm(noise_addition, p=2)

            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            #updateQuest = avgEpochLoss >= max(allAvgEpochLosses)
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)
            if(updateQuest):
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, noise_addition, adv_gen)
            prevDeviation = deviation


if(attck_type == "grill_l2_mcmc_eqwts"):
    all_condition_nums = np.full((1084,), 1 / 1084)

    print(all_condition_nums)
    print("Sum:", np.sum(all_condition_nums))

    print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    #all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())


    adv_div_list = []
    adv_mse_list = []
    for step in range(100):
        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label
            #print("source_im.shape", source_im.shape)
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            layerwise_outputs.clear()
            #print("layerwise_outputs", layerwise_outputs)
            adv_logits, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            #print("adv_layerwise_outputs", adv_layerwise_outputs)
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()

            loss = 0
            counter = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                loss += criterion(adv_output, orig_output) * cond_nums_normalized[counter]
                counter+=1
            #print("counter", counter)
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                lat_loss += criterion(adv_latent_reps[i], orig_latent_reps[i])
            
            total_loss = -1 * loss * lat_loss

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #if step % 400 == 0:

        adv_gen = model.decoder_output(adv_logits)
        adv_gen = adv_gen.sample()

        with torch.no_grad():
            scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)

            '''layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()'''

            deviation = torch.norm(adv_gen - source_im, p=2)

            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            get_em = run_time_plots_and_saves(step, loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, scaled_noise, adv_gen)




if(attck_type == "grill_l2_mcmc_rndwts"):

    all_condition_nums = np.random.rand(1084) 

    print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    #all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())


    adv_div_list = []
    adv_mse_list = []
    for step in range(100):
        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label
            #print("source_im.shape", source_im.shape)
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            layerwise_outputs.clear()
            #print("layerwise_outputs", layerwise_outputs)
            adv_logits, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            #print("adv_layerwise_outputs", adv_layerwise_outputs)
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()

            loss = 0
            counter = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                loss += criterion(adv_output, orig_output) * cond_nums_normalized[counter]
                counter+=1
            #print("counter", counter)
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                lat_loss += criterion(adv_latent_reps[i], orig_latent_reps[i])
            
            total_loss = -1 * loss * lat_loss

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #if step % 400 == 0:

        adv_gen = model.decoder_output(adv_logits)
        adv_gen = adv_gen.sample()

        with torch.no_grad():
            scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
            l2_distortion = torch.norm(scaled_noise, p=2)

            '''layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()'''

            deviation = torch.norm(adv_gen - source_im, p=2)

            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            get_em = run_time_plots_and_saves(step, loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, scaled_noise, adv_gen)




if(attck_type == "grill_wass_kf"):

    #all_condition_nums = np.random.rand(1084) 
    #all_condition_nums = np.full((1084,), 1 / 1084)

    all_condition_nums = np.load('nvae/saved_cond_nums/nvae_cond_nums.npy')

    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = []
    for step in range(100):
        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label
            
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            #normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            #normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()


            loss = 0
            counter = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                loss += wasserstein_distance(adv_output, orig_output) * cond_nums_normalized[counter]
                counter+=1
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                lat_loss += wasserstein_distance(adv_latent_reps[i], orig_latent_reps[i])
            
            total_loss = -1 * loss * lat_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #if step % 400 == 0:
        with torch.no_grad():
            #avgEpochLoss = avgEpochLoss/betachCounter
            #allAvgEpochLosses.append(avgEpochLoss)
            #print("all_adv_div_list", all_adv_div_list)
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
            l2_distortion = torch.norm(noise_addition, p=2)

            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            #updateQuest = avgEpochLoss >= max(allAvgEpochLosses)
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)
            if(updateQuest):
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, noise_addition, adv_gen)
            prevDeviation = deviation


if(attck_type == "grill_wass_kf_5p"):

    all_condition_nums = np.load('nvae/saved_cond_nums/nvae_cond_nums.npy')

    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    selector = np.zeros_like(cond_nums_normalized)
    print("selector before", selector)
    untilIndex = int(len(selector) * 0.05)

    selector[:untilIndex] = 1.0
    print("selector after", selector)
    print("test sum", np.sum(selector))
    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = []
    for step in range(100):
        for source_im in big_tensor:

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()


            loss = 0
            counter = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                loss += wasserstein_distance(adv_output, orig_output) * cond_nums_normalized[counter] * selector[counter]
                counter+=1
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                lat_loss += wasserstein_distance(adv_latent_reps[i], orig_latent_reps[i])
            
            total_loss = -1 * loss * lat_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #if step % 400 == 0:
        with torch.no_grad():
            #avgEpochLoss = avgEpochLoss/betachCounter
            #allAvgEpochLosses.append(avgEpochLoss)
            #print("all_adv_div_list", all_adv_div_list)
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
            l2_distortion = torch.norm(noise_addition, p=2)

            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            #updateQuest = avgEpochLoss >= max(allAvgEpochLosses)
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)
            if(updateQuest):
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, noise_addition, adv_gen)
            prevDeviation = deviation



if(attck_type == "grill_wass_kf_10p"):

    all_condition_nums = np.load('nvae/saved_cond_nums/nvae_cond_nums.npy')

    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    selector = np.zeros_like(cond_nums_normalized)
    print("selector before", selector)
    untilIndex = int(len(selector) * 0.1)

    selector[:untilIndex] = 1.0
    print("selector after", selector)
    print("test sum", np.sum(selector))
    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = []
    for step in range(100):
        for source_im in big_tensor:

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()


            loss = 0
            counter = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                loss += wasserstein_distance(adv_output, orig_output) * cond_nums_normalized[counter] * selector[counter]
                counter+=1
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                lat_loss += wasserstein_distance(adv_latent_reps[i], orig_latent_reps[i])
            
            total_loss = -1 * loss * lat_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #if step % 400 == 0:
        with torch.no_grad():

            l2_distortion = torch.norm(noise_addition, p=2)

            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            #updateQuest = avgEpochLoss >= max(allAvgEpochLosses)
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)
            if(updateQuest):
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, noise_addition, adv_gen)
            prevDeviation = deviation



if(attck_type == "grill_wass_kf_30p"):

    all_condition_nums = np.load('nvae/saved_cond_nums/nvae_cond_nums.npy')

    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    selector = np.zeros_like(cond_nums_normalized)
    print("selector before", selector)
    untilIndex = int(len(selector) * 0.3)

    selector[:untilIndex] = 1.0
    print("selector after", selector)
    print("test sum", np.sum(selector))
    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = []
    for step in range(100):
        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label
            
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            #normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            #normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()


            loss = 0
            counter = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                loss += wasserstein_distance(adv_output, orig_output) * cond_nums_normalized[counter] * selector[counter]
                counter+=1
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                lat_loss += wasserstein_distance(adv_latent_reps[i], orig_latent_reps[i])
            
            total_loss = -1 * loss * lat_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #if step % 400 == 0:
        with torch.no_grad():
            #avgEpochLoss = avgEpochLoss/betachCounter
            #allAvgEpochLosses.append(avgEpochLoss)
            #print("all_adv_div_list", all_adv_div_list)
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
            l2_distortion = torch.norm(noise_addition, p=2)

            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            #updateQuest = avgEpochLoss >= max(allAvgEpochLosses)
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)
            if(updateQuest):
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, noise_addition, adv_gen)
            prevDeviation = deviation



if(attck_type == "grill_wass_kf_30pRev"):

    all_condition_nums = np.load('nvae/saved_cond_nums/nvae_cond_nums.npy')

    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    selector = np.zeros_like(cond_nums_normalized)
    print("selector before", selector)
    untilIndex = int(len(selector) * 0.3)

    selector[len(selector)-untilIndex:] = 1.0
    print("selector after", selector)
    print("test sum", np.sum(selector))
    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = []
    for step in range(100):
        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label
            
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            #normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            #normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()


            loss = 0
            counter = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                loss += wasserstein_distance(adv_output, orig_output) * cond_nums_normalized[counter] * selector[counter]
                counter+=1
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                lat_loss += wasserstein_distance(adv_latent_reps[i], orig_latent_reps[i])
            
            total_loss = -1 * loss * lat_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #if step % 400 == 0:
        with torch.no_grad():
            #avgEpochLoss = avgEpochLoss/betachCounter
            #allAvgEpochLosses.append(avgEpochLoss)
            #print("all_adv_div_list", all_adv_div_list)
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
            l2_distortion = torch.norm(noise_addition, p=2)

            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            #updateQuest = avgEpochLoss >= max(allAvgEpochLosses)
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)
            if(updateQuest):
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, noise_addition, adv_gen)
            prevDeviation = deviation

if(attck_type == "grill_wass_kf_50pRev"):

    all_condition_nums = np.load('nvae/saved_cond_nums/nvae_cond_nums.npy')

    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    selector = np.zeros_like(cond_nums_normalized)
    print("selector before", selector)
    untilIndex = int(len(selector) * 0.5)

    selector[len(selector)-untilIndex:] = 1.0
    print("selector after", selector)
    print("test sum", np.sum(selector))
    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = []
    for step in range(100):
        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label
            
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            #normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            #normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()


            loss = 0
            counter = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                loss += wasserstein_distance(adv_output, orig_output) * cond_nums_normalized[counter] * selector[counter]
                counter+=1
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                lat_loss += wasserstein_distance(adv_latent_reps[i], orig_latent_reps[i])
            
            total_loss = -1 * loss * lat_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #if step % 400 == 0:
        with torch.no_grad():
            #avgEpochLoss = avgEpochLoss/betachCounter
            #allAvgEpochLosses.append(avgEpochLoss)
            #print("all_adv_div_list", all_adv_div_list)
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
            l2_distortion = torch.norm(noise_addition, p=2)

            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            #updateQuest = avgEpochLoss >= max(allAvgEpochLosses)
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)
            if(updateQuest):
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, noise_addition, adv_gen)
            prevDeviation = deviation


if(attck_type == "grill_wass_kf_70pRev"):

    all_condition_nums = np.load('nvae/saved_cond_nums/nvae_cond_nums.npy')

    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    selector = np.zeros_like(cond_nums_normalized)
    print("selector before", selector)
    untilIndex = int(len(selector) * 0.7)

    selector[len(selector)-untilIndex:] = 1.0
    print("selector after", selector)
    print("test sum", np.sum(selector))
    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = []
    for step in range(100):
        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label
            
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            #normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            #normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()


            loss = 0
            counter = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                loss += wasserstein_distance(adv_output, orig_output) * cond_nums_normalized[counter] * selector[counter]
                counter+=1
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                lat_loss += wasserstein_distance(adv_latent_reps[i], orig_latent_reps[i])
            
            total_loss = -1 * loss * lat_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #if step % 400 == 0:
        with torch.no_grad():
            #avgEpochLoss = avgEpochLoss/betachCounter
            #allAvgEpochLosses.append(avgEpochLoss)
            #print("all_adv_div_list", all_adv_div_list)
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
            l2_distortion = torch.norm(noise_addition, p=2)

            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            #updateQuest = avgEpochLoss >= max(allAvgEpochLosses)
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)
            if(updateQuest):
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, noise_addition, adv_gen)
            prevDeviation = deviation



if(attck_type == "grill_wass_kf_90pRev"):

    all_condition_nums = np.load('nvae/saved_cond_nums/nvae_cond_nums.npy')

    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    selector = np.zeros_like(cond_nums_normalized)
    print("selector before", selector)
    untilIndex = int(len(selector) * 0.9)

    selector[len(selector)-untilIndex:] = 1.0
    print("selector after", selector)
    print("test sum", np.sum(selector))
    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = []
    for step in range(100):
        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label
            
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            #normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            #normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()


            loss = 0
            counter = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                loss += wasserstein_distance(adv_output, orig_output) * cond_nums_normalized[counter] * selector[counter]
                counter+=1
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                lat_loss += wasserstein_distance(adv_latent_reps[i], orig_latent_reps[i])
            
            total_loss = -1 * loss * lat_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #if step % 400 == 0:
        with torch.no_grad():
            #avgEpochLoss = avgEpochLoss/betachCounter
            #allAvgEpochLosses.append(avgEpochLoss)
            #print("all_adv_div_list", all_adv_div_list)
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
            l2_distortion = torch.norm(noise_addition, p=2)

            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            #updateQuest = avgEpochLoss >= max(allAvgEpochLosses)
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)
            if(updateQuest):
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, noise_addition, adv_gen)
            prevDeviation = deviation



if(attck_type == "grill_wass_kf_70p"):

    all_condition_nums = np.load('nvae/saved_cond_nums/nvae_cond_nums.npy')

    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    selector = np.zeros_like(cond_nums_normalized)
    print("selector before", selector)
    untilIndex = int(len(selector) * 0.7)

    selector[:untilIndex] = 1.0
    print("selector after", selector)
    print("test sum", np.sum(selector))
    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = []
    for step in range(100):
        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label
            
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            #normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            #normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()


            loss = 0
            counter = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                loss += wasserstein_distance(adv_output, orig_output) * cond_nums_normalized[counter] * selector[counter]
                counter+=1
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                lat_loss += wasserstein_distance(adv_latent_reps[i], orig_latent_reps[i])
            
            total_loss = -1 * loss * lat_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #if step % 400 == 0:
        with torch.no_grad():
            #avgEpochLoss = avgEpochLoss/betachCounter
            #allAvgEpochLosses.append(avgEpochLoss)
            #print("all_adv_div_list", all_adv_div_list)
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
            l2_distortion = torch.norm(noise_addition, p=2)

            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            #updateQuest = avgEpochLoss >= max(allAvgEpochLosses)
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)
            if(updateQuest):
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, noise_addition, adv_gen)
            prevDeviation = deviation



if(attck_type == "grill_wass_kf_allSum"):

    all_condition_nums = np.load('nvae/saved_cond_nums/nvae_cond_nums.npy')

    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = []
    for step in range(100):
        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label
            
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            #normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            #normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()


            loss = 0
            counter = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                loss += wasserstein_distance(adv_output, orig_output) * cond_nums_normalized[counter]
                counter+=1
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                lat_loss += wasserstein_distance(adv_latent_reps[i], orig_latent_reps[i])
            
            total_loss = -1 * (loss + lat_loss)

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #if step % 400 == 0:
        with torch.no_grad():
            #avgEpochLoss = avgEpochLoss/betachCounter
            #allAvgEpochLosses.append(avgEpochLoss)
            #print("all_adv_div_list", all_adv_div_list)
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
            l2_distortion = torch.norm(noise_addition, p=2)

            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            #updateQuest = avgEpochLoss >= max(allAvgEpochLosses)
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)
            if(updateQuest):
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, noise_addition, adv_gen)
            prevDeviation = deviation



if(attck_type == "grill_wass_kf_noLast"):

    all_condition_nums = np.load('nvae/saved_cond_nums/nvae_cond_nums.npy')

    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = []
    for step in range(100):
        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label
            
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            #normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            #normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()


            loss = 0
            counter = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                loss += wasserstein_distance(adv_output, orig_output) * cond_nums_normalized[counter]
                counter+=1
            layerwise_outputs.clear()      
            '''lat_loss = 0
            for i in range(len(adv_latent_reps)):
                lat_loss += wasserstein_distance_cr(adv_latent_reps[i], orig_latent_reps[i])'''
            
            total_loss = -1 * loss #* lat_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #if step % 400 == 0:
        with torch.no_grad():
            #avgEpochLoss = avgEpochLoss/betachCounter
            #allAvgEpochLosses.append(avgEpochLoss)
            #print("all_adv_div_list", all_adv_div_list)
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
            l2_distortion = torch.norm(noise_addition, p=2)

            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            #updateQuest = avgEpochLoss >= max(allAvgEpochLosses)
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)
            if(updateQuest):
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, noise_addition, adv_gen)
            prevDeviation = deviation



if(attck_type == "grill_wass_kf_layerLosses"):

    all_condition_nums = np.load('nvae/saved_cond_nums/nvae_cond_nums.npy')

    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = []
    allStepLayerLosses = []
    for step in range(100):
        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label
            
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            #normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            #normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()


            loss = 0
            counter = 0
            EpochEndlayerLosses = []
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                theLoss = wasserstein_distance(adv_output, orig_output) 
                loss += theLoss * cond_nums_normalized[counter]
                EpochEndlayerLosses.append(theLoss.item())
                counter+=1
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                theLoss = wasserstein_distance(adv_latent_reps[i], orig_latent_reps[i])
                lat_loss += theLoss
                EpochEndlayerLosses.append(theLoss.item())

            
            total_loss = -1 * loss * lat_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #if step % 400 == 0:
        with torch.no_grad():
            EpochEndlayerLosses = np.array(EpochEndlayerLosses)
            allStepLayerLosses.append(EpochEndlayerLosses)
            print("EpochEndlayerLosses.shape", EpochEndlayerLosses.shape)
            allStepLayerLossesArray = np.array(allStepLayerLosses)
            print("allStepLayerLossesArray.shape", allStepLayerLossesArray.shape)
            print()
            np.save("nvae/stepLayerLossstore/NVAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", allStepLayerLossesArray)

            #avgEpochLoss = avgEpochLoss/betachCounter
            #allAvgEpochLosses.append(avgEpochLoss)
            #print("all_adv_div_list", all_adv_div_list)
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
            l2_distortion = torch.norm(noise_addition, p=2)

            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            #updateQuest = avgEpochLoss >= max(allAvgEpochLosses)
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)
            if(updateQuest):
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, noise_addition, adv_gen)
            prevDeviation = deviation


if(attck_type == "grill_l2_kf_layerLosses"):

    all_condition_nums = np.load('nvae/saved_cond_nums/nvae_cond_nums.npy')

    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = []
    allStepLayerLosses = []
    for step in range(100):
        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label
            
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            #normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            #normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()


            loss = 0
            counter = 0
            EpochEndlayerLosses = []
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                theLoss = criterion(adv_output, orig_output) 
                loss += theLoss * cond_nums_normalized[counter]
                EpochEndlayerLosses.append(theLoss.item())
                counter+=1
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                theLoss = criterion(adv_latent_reps[i], orig_latent_reps[i])
                lat_loss += theLoss
                EpochEndlayerLosses.append(theLoss.item())

            
            total_loss = -1 * loss * lat_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #if step % 400 == 0:
        with torch.no_grad():
            EpochEndlayerLosses = np.array(EpochEndlayerLosses)
            allStepLayerLosses.append(EpochEndlayerLosses)
            print("EpochEndlayerLosses.shape", EpochEndlayerLosses.shape)
            allStepLayerLossesArray = np.array(allStepLayerLosses)
            print("allStepLayerLossesArray.shape", allStepLayerLossesArray.shape)
            print()
            np.save("nvae/stepLayerLossstore/NVAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", allStepLayerLossesArray)

            #avgEpochLoss = avgEpochLoss/betachCounter
            #allAvgEpochLosses.append(avgEpochLoss)
            #print("all_adv_div_list", all_adv_div_list)
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
            l2_distortion = torch.norm(noise_addition, p=2)

            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            #updateQuest = avgEpochLoss >= max(allAvgEpochLosses)
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)
            if(updateQuest):
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, noise_addition, adv_gen)
            prevDeviation = deviation




if(attck_type == "la_l2_kf_layerLosses"):

    all_condition_nums = np.load('nvae/saved_cond_nums/nvae_cond_nums.npy')

    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = []
    allStepLayerLosses = []
    for step in range(100):
        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label
            
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            #normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            #normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()


            loss = 0
            counter = 0
            EpochEndlayerLosses = []
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                theLoss = criterion(adv_output, orig_output) 
                #loss += theLoss * cond_nums_normalized[counter]
                EpochEndlayerLosses.append(theLoss.item())
                counter+=1
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                theLoss = criterion(adv_latent_reps[i], orig_latent_reps[i])
                lat_loss += theLoss
                EpochEndlayerLosses.append(theLoss.item())

            
            total_loss = -1 * lat_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #if step % 400 == 0:
        with torch.no_grad():
            EpochEndlayerLosses = np.array(EpochEndlayerLosses)
            allStepLayerLosses.append(EpochEndlayerLosses)
            print("EpochEndlayerLosses.shape", EpochEndlayerLosses.shape)
            allStepLayerLossesArray = np.array(allStepLayerLosses)
            print("allStepLayerLossesArray.shape", allStepLayerLossesArray.shape)
            print()
            np.save("nvae/stepLayerLossstore/NVAE_attack_type"+str(attck_type)+"_norm_bound_"+str(desired_norm_l_inf)+"_.npy", allStepLayerLossesArray)

            #avgEpochLoss = avgEpochLoss/betachCounter
            #allAvgEpochLosses.append(avgEpochLoss)
            #print("all_adv_div_list", all_adv_div_list)
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
            l2_distortion = torch.norm(noise_addition, p=2)

            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            #updateQuest = avgEpochLoss >= max(allAvgEpochLosses)
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)
            if(updateQuest):
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, noise_addition, adv_gen)
            prevDeviation = deviation




if(attck_type == "grill_cos_kf"):

    all_condition_nums = np.load('nvae/saved_cond_nums/nvae_cond_nums.npy')

    #print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = []

    for step in range(100):
        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label
            
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            #normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            #normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()


            loss = 0
            counter = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                #loss += wasserstein_distance(adv_output, orig_output) * cond_nums_normalized[counter]
                loss +=  (cos(adv_output, orig_output)-1)**2 * cond_nums_normalized[counter]
                #loss +=  cos_cr(adv_output, orig_output) * cond_nums_normalized[counter]

                counter+=1
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                #lat_loss += wasserstein_distance(adv_latent_reps[i], orig_latent_reps[i])
                lat_loss +=  (cos(adv_latent_reps[i], orig_latent_reps[i])-1)**2
                #lat_loss +=  cos_cr(adv_latent_reps[i], orig_latent_reps[i])

            total_loss = -1 * loss * lat_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #if step % 400 == 0:
        with torch.no_grad():
            #avgEpochLoss = avgEpochLoss/betachCounter
            #allAvgEpochLosses.append(avgEpochLoss)
            #print("all_adv_div_list", all_adv_div_list)
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
            l2_distortion = torch.norm(noise_addition, p=2)

            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            #updateQuest = avgEpochLoss >= max(allAvgEpochLosses)
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)
            if(updateQuest):
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, noise_addition, adv_gen)
            prevDeviation = deviation



if(attck_type == "grill_wass_kf_cr"):

    all_condition_nums = np.load('nvae/saved_cond_nums/nvae_cond_nums.npy')

    print("before all_condition_nums.max(), all_condition_nums.max()", all_condition_nums.max(), all_condition_nums.min())
    #all_condition_nums[all_condition_nums>100.0]=100

    cond_cmpli = (all_condition_nums.max() - all_condition_nums) + 1e-6
    cond_nums_normalized = cond_cmpli / np.sum(cond_cmpli)
    print("cond_nums_normalized.shape", cond_nums_normalized.shape)

    print("after normalizing all_condition_nums.max(), all_condition_nums.max()", cond_nums_normalized.max(), cond_nums_normalized.min())

    adv_div_list = []
    adv_mse_list = []
    for step in range(100):
        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label
            
            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

            #normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
            #normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            layerwise_outputs.clear()
            _, _, _, _, _, adv_latent_reps = model(normalized_attacked)
            #adv_gen = model.decoder_output(adv_logits)
            #adv_gen = adv_gen.sample()
            adv_layerwise_outputs = layerwise_outputs.copy()
            layerwise_outputs.clear()

            _, _, _, _, _, orig_latent_reps = model(source_im)
            orig_layerwise_outputs = layerwise_outputs.copy()


            loss = 0
            counter = 0
            for layer, adv_output in adv_layerwise_outputs.items():
                #print("adv_output.shape", adv_output.shape)
                orig_output = orig_layerwise_outputs[layer]
                loss += wasserstein_distance_cr(adv_output, orig_output) * cond_nums_normalized[counter]
                counter+=1
            layerwise_outputs.clear()      
            lat_loss = 0
            for i in range(len(adv_latent_reps)):
                lat_loss += wasserstein_distance_cr(adv_latent_reps[i], orig_latent_reps[i])
            
            total_loss = -1 * loss * lat_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #if step % 400 == 0:
        with torch.no_grad():
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
            l2_distortion = torch.norm(noise_addition, p=2)

            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)

            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            get_em = run_time_plots_and_saves(step, loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, noise_addition, adv_gen)



if(attck_type == "la_l2_kf"):
    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = []
    allAvgEpochLosses = []

    for step in range(100):
        avgEpochLoss = 0
        prevAvgEpochLoss = 0
        betachCounter = 0

        for source_im in big_tensor:
            #print(" check how many")
            #source_im, label = source_im.cuda(), label

            optimizer.zero_grad()
            #print("source_im.shape", source_im.shape)
            #print("noise_addition.shape", noise_addition.shape)
            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm)
            #attacked = (source_im + scaled_noise)
            #normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            adv_logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps = model(normalized_attacked)
            reconstructed_output = model.decoder_output(adv_logits)
            adv_gen = reconstructed_output.sample()

            source_logits, log_q, log_p, kl_all, kl_diag, orig_latent_reps = model(source_im)


            distortion = torch.norm(noise_addition, 2)

            #l2_distortion = torch.norm(scaled_noise, 2)
            #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))


            loss = 0
            for i in range(len(adv_latent_reps)):
                loss = loss + criterion(adv_latent_reps[i], orig_latent_reps[i])

            total_loss = (-1 * loss )

            total_loss.backward()
            optimizer.step()
            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            with torch.no_grad():
                totLossMag =  loss 
                avgEpochLoss = avgEpochLoss + totLossMag.item()
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #if step % 400 == 0:
            betachCounter = betachCounter + 1


        with torch.no_grad():
            avgEpochLoss = avgEpochLoss/betachCounter
            allAvgEpochLosses.append(avgEpochLoss)
            #print("all_adv_div_list", all_adv_div_list)
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
            l2_distortion = torch.norm(noise_addition, p=2)

            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            #updateQuest = avgEpochLoss >= max(allAvgEpochLosses)
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)
            if(updateQuest):
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, noise_addition, adv_gen)
            prevDeviation = deviation




if(attck_type == "la_wass_kf_cr"):
    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = []

    for step in range(100):

        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm)
            #attacked = (source_im + scaled_noise)
            #normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            adv_logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps = model(normalized_attacked)
            reconstructed_output = model.decoder_output(adv_logits)
            adv_gen = reconstructed_output.sample()

            source_logits, log_q, log_p, kl_all, kl_diag, orig_latent_reps = model(source_im)


            distortion = torch.norm(noise_addition, 2)

            #l2_distortion = torch.norm(scaled_noise, 2)
            #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))


            loss = 0
            for i in range(len(adv_latent_reps)):
                loss = loss + wasserstein_distance(adv_latent_reps[i], orig_latent_reps[i])

            total_loss = (-1 * loss )

            total_loss.backward()
            optimizer.step()
            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)


        with torch.no_grad():

            l2_distortion = torch.norm(noise_addition, p=2)

            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            #updateQuest = avgEpochLoss >= max(allAvgEpochLosses)
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)
            if(updateQuest):
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, noise_addition, adv_gen)
            prevDeviation = deviation


if(attck_type == "la_cos_kf_cr"):
    adv_div_list = []
    adv_mse_list = []
    all_adv_div_list = []

    for step in range(100):

        for source_im in big_tensor:
            #source_im, label = source_im.cuda(), label

            optimizer.zero_grad()

            normalized_attacked = torch.clamp(source_im + noise_addition, mi, ma)

            #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
            #scaled_noise = noise_addition * (desired_norm_l_inf / current_L_inf_norm)
            #attacked = (source_im + scaled_noise)
            #normalized_attacked = (attacked-attacked.min())/(attacked.max()-attacked.min())

            adv_logits, log_q, log_p, kl_all, kl_diag, adv_latent_reps = model(normalized_attacked)
            reconstructed_output = model.decoder_output(adv_logits)
            adv_gen = reconstructed_output.sample()

            source_logits, log_q, log_p, kl_all, kl_diag, orig_latent_reps = model(source_im)


            distortion = torch.norm(noise_addition, 2)

            #l2_distortion = torch.norm(scaled_noise, 2)
            #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
                # ######## loss = loss + (cos(adv_latent_reps[i], orig_latent_reps[i])-1)**2


            loss = 0
            for i in range(len(adv_latent_reps)):
                #loss = loss + cos_cr(adv_latent_reps[i], orig_latent_reps[i])
                #loss +=  (cos(adv_latent_reps[i], orig_latent_reps[i])-1)**2
                loss +=  cos_cr(adv_latent_reps[i], orig_latent_reps[i])


            total_loss = (1 * loss )

            total_loss.backward()
            optimizer.step()
            with torch.no_grad():
                noise_addition.clamp_(-desired_norm_l_inf, desired_norm_l_inf)

            #if step % 400 == 0:
        with torch.no_grad():
            #avgEpochLoss = avgEpochLoss/betachCounter
            #allAvgEpochLosses.append(avgEpochLoss)
            #print("all_adv_div_list", all_adv_div_list)
            #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
            #l_inf_distortion = torch.norm(noise_addition, p=float('inf'))
            l2_distortion = torch.norm(noise_addition, p=2)

            l_inf_distortion = torch.norm(normalized_attacked - source_im, p=float('inf'))

            layerwise_outputs.clear()
            adv_logits, _, _, _, _, _ = model(normalized_attacked)
            adv_gen = model.decoder_output(adv_logits)
            adv_gen = adv_gen.sample()
            layerwise_outputs.clear()

            deviation = torch.norm(adv_gen - source_im, p=2)
            print("cur deviation", deviation)

            all_adv_div_list.append(deviation.item())
            mase_dev = torch.mean((adv_gen - normalized_attacked) ** 2)

            #updateQuest = avgEpochLoss >= max(allAvgEpochLosses)
            updateQuest = deviation >= max(all_adv_div_list)
            print("updateQuest", updateQuest)
            if(updateQuest):
                get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, noise_addition, adv_gen)
            prevDeviation = deviation


