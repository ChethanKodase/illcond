# GRILL: Restoring Gradient Signal in Ill-Conditioned Layers for More Effective Adversarial Attacks on Autoencoders


# Code for NVAE attacks

We consider the official implementation of NVAE from https://github.com/NVlabs/NVAE. We take the pretrained weights from  the oficial publishers and implement adversarial attacks


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


#### To get NVAE encoder condition number plots


<pre>
```
python nvae/NvaeConditionNumberSingularValuesPlotsForTheEncoder.py
```
</pre>



#### To run universal adversarial attacks on NVAE


To save condition numbers to use in GRILL



<pre>
```
python nvae/NvaeSaveConditionNumbers.py --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
```
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

### To get output distortion distributions across perturbations for universal attacks on NVAEs

<pre>
```
python nvae/NvaeDeviationsDistributionsAcrossPerturbations.py 
```
</pre>


#### To run universal adaptive attacks on NVAE

<pre>
```
python nvae/NvaeAllUniversalAdaptiveAttacks.py --attck_type "la_l2_kf_mcmc" --desired_norm_l_inf 0.037 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeAllUniversalAdaptiveAttacks.py --attck_type "grill_wass_kf_mcmc" --desired_norm_l_inf 0.09 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
```
</pre>

#### To get output distortion distributions across perturbations plots for universal adaptive attacks on NVAE

python nvae/NvaeDeviationsDistributionsAcrossPerturbationsForAdaptiveAttacks.py


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

##### To get sample specific attack qualitative results 

<pre>
```
python nvae/NvaeSampleSpecificQualitativePlotsForCLassicAttacks.py --desired_norm_l_inf 0.03 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeSampleSpecificQualitativePlotsForCLassicAttacks.py --desired_norm_l_inf 0.02 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeSampleSpecificQualitativePlotsForCLassicAttacks.py --desired_norm_l_inf 0.01 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
```
</pre>

#### TO Run Layerwise gradient restoration effects ablation 

<pre>
```
python nvae/NvaeAllUniversalAttacks.py --attck_type "grill_wass_kf_allSum" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeAllUniversalAttacks.py --attck_type "grill_wass_kf_30pRev" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeAllUniversalAttacks.py --attck_type "grill_wass_kf_50pRev" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeAllUniversalAttacks.py --attck_type "grill_wass_kf_70pRev" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeAllUniversalAttacks.py --attck_type "grill_wass_kf_90pRev" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeAllUniversalAttacks.py --attck_type "grill_wass_kf" --desired_norm_l_inf 0.05 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoints
```
</pre>

#### To plot the above ablation results

<pre>
```
python nvae/NvaeLayerFractionsAblation.py
```
</pre>


#### To run layer weighting ablations on NVAE
<pre>
```
python nvae/NvaeAllUniversalAdaptiveAttacks.py --attck_type "grill_l2_mcmc" --desired_norm_l_inf 0.035 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoin
python nvae/NvaeAllUniversalAdaptiveAttacks.py --attck_type "grill_l2_mcmc_eqwts" --desired_norm_l_inf 0.035 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
python nvae/NvaeAllUniversalAdaptiveAttacks.py --attck_type "grill_l2_mcmc_rndwts" --desired_norm_l_inf 0.035 --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint
```
</pre>


#### To get the convergence plots for NVAE
<pre>
```
python nvae/NvaeGetConvergencePlots.py
```
</pre>

#### To get nvae layerlosses tracking plots

<pre>
```
python nvae/NvaeLayerLossesPlotsSymlog.py 
```
</pre>