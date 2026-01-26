

'''
conda deactivate
cd illcond
conda activate /home/luser/anaconda3/envs/inn
python betaVAEandTcVAE/TCvaetrainingWithCelebA.py  --which_gpu 1 --beta_value 5.0 --data_directory /home/luser/autoencoder_attacks/train_aautoencoders/data_cel1 --batch_size 64 --epochs 200 --lr 1e-4 --run_time_plot_dir runtimePlots --checkpoint_storage vae_checkpoints
'''


import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn, optim
from torchvision import datasets, transforms

import zipfile

import shutil
import os
import pandas as pd
import math

from utils.distributions import log_Bernoulli, log_Gaus_diag


import argparse

parser = argparse.ArgumentParser(description='VAE celebA training')

parser.add_argument('--which_gpu', type=int, default=0, help='Index of the GPU to use (0-N)')
parser.add_argument('--beta_value', type=float, default=5.0, help='Beta VAE beta value')
parser.add_argument('--data_directory', type=str, default=0, help='data directory')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--epochs', type=int, default=200, help='training batch size')
parser.add_argument('--lr', type=float, default=1e-6, help='Beta VAE beta value')
parser.add_argument('--run_time_plot_dir', type=str, default="betaVAEandTcVAE/a_training_runtime", help='run time plots directory')
parser.add_argument('--checkpoint_storage', type=str, default="betaVAEandTcVAE/checkpoints", help='run time plots directory')



args = parser.parse_args()

which_gpu = args.which_gpu
beta_value = args.beta_value
data_directory = args.data_directory
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
run_time_plot_dir = args.run_time_plot_dir
checkpoint_storage = args.checkpoint_storage



device = ("cuda:"+str(which_gpu)+"" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training



data_directory1 = ''+data_directory+'/smile/'
data_directory2 = ''+data_directory+'/no_smile/'
img_list = os.listdir(data_directory1)
img_list.extend(os.listdir(data_directory2))
transform = transforms.Compose([
          transforms.Resize((64, 64)),
          transforms.ToTensor()
          ])
celeba_data = datasets.ImageFolder(data_directory, transform=transform)
train_set, test_set = torch.utils.data.random_split(celeba_data, [int(len(img_list) * 0.8), len(img_list) - int(len(img_list) * 0.8)])
train_data_size = len(train_set)
test_data_size = len(test_set)
print('train_data_size', train_data_size)
print('test_data_size', test_data_size)
trainLoader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)
testLoader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)


import torch
import torch.nn as nn
import torch.nn.functional as F

from vae_train import VAE, VAE_big

likelihood_selection =  ['bernoulli', 'gaussian']

likelihood = likelihood_selection[0]

if likelihood == 'bernoulli':
    log_lik = lambda x, x_mean, x_logvar, dim: log_Bernoulli(x, x_mean, dim=dim)
elif likelihood == 'gaussian':
    log_lik = log_Gaus_diag

image_channels = 3

print('image_channels', image_channels)

#model = VAE(image_channels=image_channels).to(device)

model = VAE_big(device, image_channels=image_channels).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

beta_value = 10.0

def reconstrcution_kld_vae(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE, KLD

def reconstruction_error(x, x_mean, x_logvar):
    return log_Gaus_diag(x, x_mean, x_logvar, 1)


def reparametrize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
    return eps.mul(std).add_(mu)


'''def reparameterize(self, mu, logvar):
    std = logvar.mul(0.5).exp_()
    # return torch.normal(mu, std)
    esp = torch.randn(*mu.size()).to(self.device)
    z = mu + std * esp
    return z.to(self.device)'''

def bottleneck(self, h):
    mu, logvar = self.fc1(h), self.fc2(h)
    z = self.reparameterize(mu.to(self.device), logvar.to(self.device))
    return z, mu, logvar
    
def representation(self, x):
    return self.bottleneck(self.encoder(x))[0]

'''def forward(self, x):
    h = self.encoder(x)
    z, mu, logvar = self.bottleneck(h.to(self.device))
    z = self.fc3(z)
    #print('z.shape', z.shape)
    return self.decoder(z), mu, logvar'''

def calc_entropies(z_sample, z_q_mean, z_q_logvar):
    MB, z_dim = z_sample.shape
    z_sample = z_sample.view(MB, 1, z_dim)
    z_q_mean = z_q_mean.view(1, MB, z_dim)
    z_q_logvar = z_q_logvar.view(1, MB, z_dim)
    log_qz_i = -0.5 * (math.log(2.0*math.pi) +
                        z_q_logvar +
                        torch.pow(z_sample - z_q_mean, 2) / (torch.exp(z_q_logvar) + 1e-10))  # MB x MB x z_dim
    marginal_entropies = (math.log(MB) - torch.logsumexp(log_qz_i, dim=0)) # MB x z_dim
    log_qz = log_qz_i.sum(2) # MB x MB
    joint_entropy = math.log(MB) - torch.logsumexp(log_qz, dim=0)  # MB
    return marginal_entropies, joint_entropy


#epochs = 200

#tc_vae_beta = 5.0

train_loss = []

for epoch in range(epochs):
   
    total_train_loss = 0
    # training our model
    for idx, (image, label) in enumerate(trainLoader):
        images, label = image.to(device), label.to(device)

        z_q_mean, z_q_logvar = model.fc1(model.encoder(images.to(device))), model.fc2(model.encoder(images.to(device)))
        z_q = reparametrize(z_q_mean, z_q_logvar)
        z_q_dec = model.fc3(z_q)
        x_mean = model.decoder(z_q_dec)
        x_logvar = -torch.ones_like(x_mean)*math.log(2*math.pi)

        log_q_zx = log_Gaus_diag(z_q, z_q_mean, z_q_logvar, dim=1)  # MB

        z_q_mean1 =  z_q_mean*0.0   # This is the mean of the prior distribution
        z_q_logvar1 = z_q_logvar*0.0 # This is the log variance of the prior distribution : Zero logvariance means that the prior is a standard normal distribution : means thayt variance is 1 . : I mean mean is 0 and variance is 1
        log_pz = log_Gaus_diag(z_q, z_q_mean1, z_q_logvar1, dim=1)  # MB

        MB = batch_size

        recon_images, mu, logvar, z = model(images.to(device))

        BCE, KLD = reconstrcution_kld_vae(recon_images.to(device), images.to(device), mu.to(device), logvar.to(device))

        marginal_entropies, joint_entropy = calc_entropies(z_q, z_q_mean, z_q_logvar)

        MI = log_q_zx + joint_entropy  # MB
        TC = (marginal_entropies.sum(1) - joint_entropy)  # MB
        TC = torch.mean(TC)
        MI = torch.mean(MI)

        loss = BCE + MI + beta_value * TC + KLD

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print('loss', loss)
    print("Epoch : ", epoch)


    with torch.no_grad():
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(images[0].permute(1, 2, 0).cpu().numpy())
        ax[0].set_title('Input Image')
        ax[0].axis('off')

        ax[1].imshow(recon_images[0].cpu().detach().permute(1, 2, 0).cpu().numpy())
        ax[1].set_title('Reconstructed Image')
        ax[1].axis('off')

        plt.show()
        plt.savefig(""+run_time_plot_dir+"/tcVAE_epoch_"+str(epoch)+"_.png")


    #torch.save(model.state_dict(), ''+checkpoint_storage+'/celebA_CNN_TCVAE'+str(beta_value)+'_big_trainSize'+str(train_data_size)+'_epochs'+str(epoch)+'.torch')
