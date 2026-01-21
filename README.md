# GRILL: Restoring Gradient Signal in Ill-Conditioned Layers for More Effective Adversarial Attacks on Autoencoders


# Code for NVAE attacks

We consider the official implementation of NVAE from https://github.com/NVlabs/NVAE. We take the pretrained weights from  the oficial publishers and implement adversarial attacks

#### clone the nvae official repository using the code below: 

<pre>
```
git clone https://github.com/NVlabs/NVAE.git
```
</pre>



Follow the instructions from https://github.com/NVlabs/NVAE and download the checkpoints for celebA 64 dataset from https://drive.google.com/drive/folders/14DWGte1E7qnMTbAs6b87vtJrmEU9luKn 

Command Arguments: 
1. `desired_norm_l_inf`:  L-infinity bound on the added adversarial noise
2. `attck_type` : Choose the attack method from `la_l2, la_wass, la_cos, grill_l2, grill_wass, grill_cos`. Descriptions for these methods are given in our paper
3. `nvae_checkpoint_path` : Address of the downloaded trained NVAE model weights from the publishers of https://arxiv.org/abs/2007.03898 , code: https://github.com/NVlabs/NVAE
4. `your_data_directory` : address of the FFHQ images directory
5. `uni_noise_path` : Directory where the optimized noise is saved
6. `which_gpu` : Enter the index of the GPU you want to use 



#### To create the environment and install dependencies for adversarial attacks on NVAAE

<pre>
```
conda deactivate
cd alma
python -m venv nvaeenv
source nvaeenv/bin/activate
pipenv install -r requirements.txt
```
</pre>


#### To select the GPU visibility activate the environment and open the illcond directory


<pre>
export CUDA_VISIBLE_DEVICES=4
cd NVAE/
source nvaeenv1/bin/activate
cd ..
cd illcond/
</pre>



#### To run universal adversarial attacks on NVAE

To save condition numbers to use in GRILL

<pre>
python nvae/NvaeSaveConditionNumbers.py --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
</pre>

To run all attacks

<pre>
```
python nvae/NvaeAllUniversalAttacks.py --attck_type "la_l2_kf" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeAllUniversalAttacks.py --attck_type "la_wass_kf_cr" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeAllUniversalAttacks.py --attck_type "la_cos_kf_cr" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeAllUniversalAttacks.py --attck_type "grill_l2_kf" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeAllUniversalAttacks.py --attck_type "grill_wass_kf" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeAllUniversalAttacks.py --attck_type "grill_cos_kf" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
```
</pre>

### To get output distortion distributions for universal attacks on NVAEs

<pre>
```
python nvae/NvaeDeviationsDistributionsAcrossPerturbations.py 
```
</pre>



#### To run sample specific attacks on NVAE

<pre>
```
python nvae/NvaeAllSampleSpecificAttacks.py --attck_type "la_l2_kf_SS" --desired_norm_l_inf 0.02 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeAllSampleSpecificAttacks.py --attck_type "la_wass_kf_cr_SS" --desired_norm_l_inf 0.02 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeAllSampleSpecificAttacks.py --attck_type "la_cos_kf_cr_SS" --desired_norm_l_inf 0.02 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeAllSampleSpecificAttacks.py --attck_type "grill_l2_kf_SS" --desired_norm_l_inf 0.02 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeAllSampleSpecificAttacks.py --attck_type "grill_wass_kf_SS" --desired_norm_l_inf 0.02 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeAllSampleSpecificAttacks.py --attck_type "grill_cos_kf_SS" --desired_norm_l_inf 0.02 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint

```
</pre>