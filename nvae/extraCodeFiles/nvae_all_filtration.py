import torch
import torch.nn as nn

from model import *


from model import AutoEncoder
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import random

from torchvision import datasets, transforms
import os
import pandas as pd

#device = "cuda:1" if torch.cuda.is_available() else "cpu"


'''

0, 1, 2, 3

conda deactivate
conda deactivate
conda deactivate
export CUDA_VISIBLE_DEVICES=3
cd NVAE/
source nvaeenv1/bin/activate
cd ..
cd alma/
python nvae/nvae_all_filtration.py --feature_no 2 --source_segment 0 --attck_type "combi_l2_cond" --nvae_checkpoint_path ../NVAE/pretrained_checkpoint --data_directory data_cel1  --uni_noise_path ../NVAE/attack_run_time_univ/attack_noise --desired_norm_l_inf 0.05 --filter_location nvae/filter_storage

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
parser.add_argument('--feature_no', type=int, default=5, help='Index of the feature to attack (0-5)')
parser.add_argument('--source_segment', type=int, default=3, help='Segment index')
parser.add_argument('--attck_type', type=str, default="lip", help='Segment index')
parser.add_argument('--nvae_checkpoint_path', type=str, default=0, help='nvae checkpoint directory')
parser.add_argument('--data_directory', type=str, default=0, help='data directory')
parser.add_argument('--uni_noise_path', type=str, default=0, help='nvae checkpoint directory')
parser.add_argument('--desired_norm_l_inf', type=float, default="lip", help='Segment index')
parser.add_argument('--filter_location', type=str, default=0, help='filter directory')


args = parser.parse_args()

feature_no = args.feature_no
source_segment = args.source_segment
attck_type = args.attck_type
nvae_checkpoint_path = args.nvae_checkpoint_path
data_directory = args.data_directory
uni_noise_path = args.uni_noise_path
desired_norm_l_inf = args.desired_norm_l_inf
filter_location = args.filter_location

all_features = ["youngmen", "oldmen", "youngwomen", "oldwomen" ]

populations_all_features = ["bald", "beard", "oldfemaleGlass", "hat", "blackWomen", "generalWhiteWomen", "blackMen", "generalWhiteMen", "men : 84434", "women : 118165", "young : 156734", "old : 45865", "youngmen : 53448 ", "oldmen : 7003", "youngwomen : 103287", "oldwomen : 1116" ]

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



#desired_norm_l_inf = 0.05  # Worked very well

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


batch_list = []

for idx, (source_im, _) in enumerate(testLoader):
    source_im, _ = source_im.cuda(), _
    batch_list.append(source_im)  # Store batch in a list

big_tensor = torch.stack(batch_list)  # Shape: (num_batches, batch_size, C, H, W)
del batch_list
del testLoader

noise_addition = 2.0 * torch.rand(1, 3, 64, 64).cuda() - 1.0


def hook_fn(module, input, output):
    print(f"Layer: {module.__class__.__name__}")
    print(f"Output shape: {output.shape}")



class SimpleCNN(nn.Module):
    def __init__(self, image_channels):
        super(SimpleCNN, self).__init__()

        # Store multiple layers in a ModuleList
        self.all_encs = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(image_channels, 3, kernel_size=3, stride=1, padding=1),
                #nn.ReLU()
            ) for _ in range(1)] +  # First set of layers
            
            [nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                #nn.ReLU()
            ) for _ in range(2)] +  # Second set of layers

            [nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                #nn.ReLU()
            ) for _ in range(41)] +  # third set of layers

            [nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                #nn.ReLU()
            ) for _ in range(21)] +  # third set of layers

            [nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                #nn.ReLU()
            ) for _ in range(10)]  # third set of layers
        )

    def forward(self, x):
        for encoder in self.all_encs:
            x = encoder(x)
        return x

inter_model = SimpleCNN(image_channels=3).cuda()


import torch.optim as optim
optimizer = optim.Adam(inter_model.parameters(), lr=0.0001)
adv_alpha = 0.5
noise_addition.requires_grad = True
criterion = nn.MSELoss()
num_steps = 40000
prev_loss = 0.0

layerwise_outputs = {}

def encoder_hook_fn(module, input, output):
    layerwise_outputs[module] = output

# Register hooks for encoder layers
encoder_hook_handles = []



for name, layer in model.enc_tower.named_modules():
    handle = layer.register_forward_hook(encoder_hook_fn)
    encoder_hook_handles.append(handle)



#def run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, scaled_noise, adv_gen):
def run_time_plots_and_saves(step, total_loss, deviation, mase_dev, normalized_attacked, scaled_noise, adv_gen):

    with torch.no_grad():
        print(f"Step {step}, Loss: {total_loss.item()}, deviation: {deviation}, recon mse: {mase_dev}")
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
        plt.savefig(""+filter_location+"/run_time/NVAE_attack_.png")

    optimized_noise = scaled_noise

    torch.save(inter_model.state_dict(), ""+filter_location+"/nvae_filter_norm_bound_"+str(desired_norm_l_inf)+".torch")



def save_images_in_row(images, save_path):
    """
    Saves 15 images in a single row.

    Args:
        images (torch.Tensor): Tensor of shape [15, 3, 64, 64]
        save_path (str): Path to save the image
    """
    # Convert to NumPy and move to CPU if necessary
    images = images.detach().cpu().numpy()

    # Normalize images to [0,1] if they are in range [-1,1]
    images = (images - images.min()) / (images.max() - images.min())

    # Create a figure
    fig, axarr = plt.subplots(1, 15, figsize=(15, 5))  # 15 columns, 1 row

    for i in range(15):
        img = np.transpose(images[i], (1, 2, 0))  # Convert from (C, H, W) -> (H, W, C)
        axarr[i].imshow(img)
        axarr[i].axis('off')  # Hide axes

    plt.subplots_adjust(wspace=0.1, hspace=0)  # Adjust spacing
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # Save the figure
    plt.close(fig)



attck_types = ["hlatent_l2", "hlatent_wasserstein", "hlatent_SKL", "hlatent_cos", "combi_l2_cond", "combi_wasserstein", "combi_SKL", "combi_cos", ]


#if(attck_type == "combi_l2_cond"):
noise_addition = torch.load(""+uni_noise_path+"/NVAE_attack_type"+str(attck_types[4])+"_norm_bound_"+str(desired_norm_l_inf)+"feature_"+str(select_feature)+"_source_segment_"+str(source_segment)+"_.pt")

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
        #print("source_im.shape", source_im.shape)
        #current_L_inf_norm = torch.norm(noise_addition, p=float('inf'))
        #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))

        normalized_attacked = (source_im + (noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))))
        normalized_attacked = (normalized_attacked-normalized_attacked.min())/(normalized_attacked.max()-normalized_attacked.min())

        xx = source_im
        x = normalized_attacked
        both_recons = []
        #for x in [source_im, normalized_attacked]:

        filter_loss = 0
        l_ct = 0
        x = inter_model.all_encs[l_ct](x)   ###########inter_model
        s = model.stem(2 * x - 1.0)
        l_ct+=1
        ss = model.stem(2 * xx - 1.0)
        filter_loss +=criterion(s, ss)


        for cell in model.pre_process:
            s = inter_model.all_encs[l_ct](s)   ###########inter_model
            l_ct+=1
            s = cell(s)

            ss = cell(ss)
            filter_loss +=criterion(s, ss)


        combiner_cells_enc = []
        combiner_cells_s = []
        combiner_cells_ss = []

        #print("model.enc_tower", model.enc_tower)

        for cell in model.enc_tower:
            #print("cell.cell_type", cell.cell_type)
            if cell.cell_type == 'combiner_enc':
                #print("cell.cell_type", cell.cell_type)
                #print("cell", cell)
                combiner_cells_enc.append(cell)
                combiner_cells_s.append(s)
                combiner_cells_ss.append(ss)

            else:
                #s = inter_model.all_encs[l_ct](s)   ###########inter_model
                s = cell(s)
                l_ct += 1
                ss = cell(ss)
                #filter_loss +=criterion(s, ss)

        combiner_cells_enc.reverse()
        combiner_cells_s.reverse()

        idx_dec = 0
        ftr = model.enc0(s)                            # this reduces the channel dimension
        param0 = model.enc_sampler[idx_dec](ftr)
        
        mu_q, log_sig_q = torch.chunk(param0, 2, dim=1)
        dist = Normal(mu_q, log_sig_q)   # for the first approx. posterior
        z, _ = dist.sample()
        log_q_conv = dist.log_p(z)

        # apply normalizing flows
        nf_offset = 0
        #print('model.num_flows', model.num_flows)
        #print("model.nf_cells", model.nf_cells)
        for n in range(model.num_flows):
            z, log_det = model.nf_cells[n](z, ftr)
            log_q_conv -= log_det
        nf_offset += model.num_flows
        all_q = [dist]
        all_lat_rep = [z]

        all_log_q = [log_q_conv]


        # To make sure we do not pass any deterministic features from x to decoder.
        s = 0

        # prior for z0
        dist = Normal(mu=torch.zeros_like(z), log_sigma=torch.zeros_like(z))
        log_p_conv = dist.log_p(z)
        all_p = [dist]
        all_log_p = [log_p_conv]


        idx_dec = 0

        s = model.prior_ftr0.unsqueeze(0) # random tensor of shape of the encoder tower output
        batch_size = z.size(0)
        s = s.expand(batch_size, -1, -1, -1)
        #print("what is model.dec_sampler", model.dec_sampler) # They are neural nets
        #print("What is model.nf_cells", model.nf_cells) # They are again neural nets
        for cell in model.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    # form prior
                    param = model.dec_sampler[idx_dec - 1](s)
                    mu_p, log_sig_p = torch.chunk(param, 2, dim=1)

                    # form encoder
                    ftr = combiner_cells_enc[idx_dec - 1](combiner_cells_s[idx_dec - 1], s)
                    param = model.enc_sampler[idx_dec](ftr)

                    mu_q, log_sig_q = torch.chunk(param, 2, dim=1)

                    dist = Normal(mu_p + mu_q, log_sig_p + log_sig_q) if model.res_dist else Normal(mu_q, log_sig_q)
                    z, _ = dist.sample()

                    log_q_conv = dist.log_p(z)

                    # apply NF

                    for n in range(model.num_flows):
                        z, log_det = model.nf_cells[nf_offset + n](z, ftr)

                        log_q_conv -= log_det

                    nf_offset += model.num_flows

                    all_log_q.append(log_q_conv)

                    all_lat_rep.append(z)
                    all_q.append(dist)

                    # evaluate log_p(z)
                    dist = Normal(mu_p, log_sig_p)
                    log_p_conv = dist.log_p(z)
                    all_p.append(dist)
                    all_log_p.append(log_p_conv)

                # 'combiner_dec'
                s = cell(s, z)
                idx_dec += 1
            else:

                s = cell(s)

            
        if model.vanilla_vae:
            s = model.stem_decoder(z)

        for cell in model.post_process:
            s = cell(s)

        logits = model.image_conditional(s)

        # compute kl
        kl_all = []
        kl_diag = []
        log_p, log_q = 0., 0.


        for q, p, log_q_conv, log_p_conv in zip(all_q, all_p, all_log_q, all_log_p):
            #print("model.with_nf", model.with_nf)
            
            if model.with_nf:
                kl_per_var = log_q_conv - log_p_conv
            else:
                kl_per_var = q.kl(p)

            kl_diag.append(torch.mean(torch.sum(kl_per_var, dim=[2, 3]), dim=0))
            kl_all.append(torch.sum(kl_per_var, dim=[1, 2, 3]))
            log_q += torch.sum(log_q_conv, dim=[1, 2, 3])
            log_p += torch.sum(log_p_conv, dim=[1, 2, 3])


        #logits, log_q, log_p, kl_all, kl_diag, all_lat_rep

        reconstructions = model.decoder_output(logits)
        reconstructions = reconstructions.sample()


        #total_loss = criterion(both_recons[0], both_recons[1])
        total_loss = filter_loss
        both_recons = []
        #print("total_loss", total_loss)
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #if step % 400 == 0:
        
    with torch.no_grad():
        #scaled_noise = noise_addition * (desired_norm_l_inf / torch.norm(noise_addition, p=float('inf')))
        #l_inf_distortion = torch.norm(scaled_noise, p=float('inf'))
        #l2_distortion = torch.norm(scaled_noise, p=2)

        #layerwise_outputs.clear()
        #adv_logits, _, _, _, _, _ = model(normalized_attacked)
        #adv_gen = model.decoder_output(adv_logits)
        #adv_gen = adv_gen.sample()
        #layerwise_outputs.clear()

        deviation = torch.norm(reconstructions - source_im, p=2)

        mase_dev = torch.mean((reconstructions - normalized_attacked) ** 2)

        #get_em = run_time_plots_and_saves(step, total_loss, l2_distortion, l_inf_distortion, deviation, mase_dev, normalized_attacked, scaled_noise, adv_gen)
        get_em = run_time_plots_and_saves(step, total_loss, deviation, mase_dev, normalized_attacked, noise_addition, reconstructions)

