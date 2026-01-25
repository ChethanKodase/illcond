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



#### To create the environment and install dependencies for adversarial attacks on NVAE

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

#### To get sample wise qualitative results of all universal attacks


<pre>
```
python nvae/NvaeAllUniversalAttacksSampleWiseQualitativePlots.py --data_directory ../data_cel1 --nvae_checkpoint_path ../NVAE/pretrained_checkpoint --uni_noise_path ../NVAE/attack_run_time_univ/attack_noise --desired_norm_l_inf 0.025
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


# Code for Qwen 2.5 attacks


Here we perform evaluation of adversarial robustness of Qwen 2.5.


Create a conda environment :

`conda create -n QwenAttack python=3.10 -y`

Activate :

`conda activate QwenAttack`

Run :

`export PYTHONNOUSERSITE=1`

Install torch and torchvision :

`python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision`


Install other packages :

```
python -m pip install \
  "transformers>=4.45.0" \
  accelerate \
  huggingface_hub \
  pillow \
  sentencepiece \
  tiktoken \
  einops \
  "protobuf<5"

```

Install hugging face hub:

`pip install huggingface_hub`


Login with HF token: 

`hf auth login` 


Make a directory inside the repo:

`mkdir Qwen2.5-VL-7B-Instruct`

Paste the address of the above directory in the local_dir in the below command

```

python - << 'EOF'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
    local_dir="/home/luser/vlmAttack/Qwen2.5-VL-7B-Instruct",
    local_dir_use_symlinks=False,
)
print("Download complete.")
EOF


```



#### To do inference of the Qwen 2.5 model 

Select the GPU and activate the environment 

```
export CUDA_VISIBLE_DEVICES=6
conda deactivate
cd illcond/
conda activate QwenAttack
export PYTHONNOUSERSITE=1
```


`python QwenAttack/Qwen2_5_Inference.py`

#### to To perform sample specific adversarial attacks on Qwen 2.5 and save the results : 

```
python QwenAttack/QwenUntargetedAttacks.py --attck_type grill_cos --desired_norm_l_inf 0.03 --learningRate 0.001 --sampleName astronauts --numSteps 10000
python QwenAttack/QwenUntargetedAttacks.py --attck_type OA_cos --desired_norm_l_inf 0.03 --learningRate 0.001 --sampleName astronauts --numSteps 10000
python QwenAttack/QwenUntargetedAttacks.py --attck_type grill_l2 --desired_norm_l_inf 0.03 --learningRate 0.001 --sampleName astronauts --numSteps 10000
python QwenAttack/QwenUntargetedAttacks.py --attck_type OA_l2 --desired_norm_l_inf 0.03 --learningRate 0.001 --sampleName astronauts --numSteps 10000
python QwenAttack/QwenUntargetedAttacks.py --attck_type grill_wass --desired_norm_l_inf 0.03 --learningRate 0.001 --sampleName astronauts --numSteps 10000
python QwenAttack/QwenUntargetedAttacks.py --attck_type OA_wass --desired_norm_l_inf 0.03 --learningRate 0.001 --sampleName astronauts --numSteps 10000
```

Repeat the same for other values of $L_\inf$ norms and other data samples by updating --desired_norm_l_inf and --sampleName . Use blackHole, boat, cheetah, light, walker and nature which are already available as images in the repository.


##### To plot the layerwise singular values and condition number 

`python QwenAttack/qwen2p5Conditioning.py`




# Code for DiffAE attacks

Follow https://github.com/phizaz/diffae to download the checkpoints and FFHQ dataset.

#### Install the conda environment required and activate:


<pre>
```
conda env create -f environment2.yml
cd alma
conda activate your_diffae_environment
```
</pre>

#### To run universal attacks on DiffAE

<pre>
```
cd illcond
conda activate dt2
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.18 --attck_type la_l2_kfAdamNoScheduler1 --which_gpu 5 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.18 --attck_type la_wass_kfAdamNoScheduler1 --which_gpu 3 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.18 --attck_type la_cos_kfAdamNoScheduler1 --which_gpu 4 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.18 --attck_type grill_l2_kfAdamNoScheduler1 --which_gpu 6 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.18 --attck_type grill_wass_kfAdamNoScheduler1 --which_gpu 2 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.18 --attck_type grill_cos_kfAdamNoScheduler1 --which_gpu 7 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
```
</pre>

#### To run adaptive attacks on DiffAE

<pre>
```
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.20 --attck_type grill_cos_kfAdamNoScheduler1_mcmc --which_gpu 6 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.20 --attck_type la_cos_kfAdamNoScheduler1_mcmc --which_gpu 7 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
```
</pre>


#### To run layerweight ablations 


<pre>
```
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.33 --attck_type grill_cos_pr1 --which_gpu 1 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.33 --attck_type grill_cos_pr_rnd1 --which_gpu 2 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.33 --attck_type grill_cos_pr_unif1 --which_gpu 3 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
```
</pre>

#### To get histogram plots for ablations run 

<pre>
```
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.33 --attck_type la_cos_pr --which_gpu 4 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallUniversalAttacks.py --desired_norm_l_inf 0.33 --attck_type grill_cos_pr_rnd1 --which_gpu 2 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
```
</pre>

#### To run sample specific attacks on DiffAE

<pre>
```
python diffae/DiffAEallSampleSpecificAttacks.py --desired_norm_l_inf 0.5 --attck_type la_l2_kfAdamNoScheduler1_SS --which_gpu 0 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallSampleSpecificAttacks.py--desired_norm_l_inf 0.03 --attck_type la_wass_kfAdamNoScheduler1_SS --which_gpu 1 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallSampleSpecificAttacks.py--desired_norm_l_inf 0.03 --attck_type la_cos_kfAdamNoScheduler1_SS --which_gpu 2 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallSampleSpecificAttacks.py--desired_norm_l_inf 0.03 --attck_type grill_l2_kfAdamNoScheduler1_SS --which_gpu 3 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallSampleSpecificAttacks.py--desired_norm_l_inf 0.03 --attck_type grill_wass_kfAdamNoScheduler1_SS --which_gpu 4 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
python diffae/DiffAEallSampleSpecificAttacks.py--desired_norm_l_inf 0.03 --attck_type grill_cos_kfAdamNoScheduler1_SS --which_gpu 5 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad
```
</pre>



#### To save universal attacks output distortion quantitatively


<pre>
```
python diffae/DiffAEoutputDistortionStorageAfterAttack.py --desired_norm_l_inf 0.18 --which_gpu 1 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad --noise_directory diffae/noise_storage
```
</pre>

#### To save universal attacks output distortion quantitatively for adaptive attacks


<pre>
```
python diffae/DiffAEoutputDistortionStorageAfterAdaptiveAttacks.py --desired_norm_l_inf 0.18 --which_gpu 1 --diffae_checkpoint ../diffae/checkpoints --ffhq_images_directory ../diffae/imgs_align_uni_ad --noise_directory diffae/noise_storage
```
</pre>


#### To get output distortion plots for univeral attacks across perturbation budgets

<pre>
```
python diffae/DiffAEDistortionDistributionPlotsForClassicUniversalAttacks.py --epsilon_list 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.30 0.31
```
</pre>

#### To get output distortion plots for univeral adaptive attacks across perturbation budgets

<pre>
```
python diffae/DiffAEOutPutDistortionDistributionPlotsForUniversalAdaptiveAttacks.py --epsilon_list 0.21 0.24 0.25 0.30 0.33 
```
</pre>

#### To get DiffAE sample specific attacks qualitative plots

<pre>
```
python diffae/DiffAESampleSpecificAttackQualitativePlots.py --desired_norm_l_inf 0.18 --which_gpu 1 --diffae_checkpoint diffae/checkpoints --ffhq_images_directory diffae/imgs_align_uni_ad
```
</pre>


#### To get layerwise DiffAE condition number and singular values plots

<pre>
```
python diffae/DiffAEconditionAnalysisPlots.py --which_gpu 7 --diffae_checkpoint diffae/checkpoints
```
</pre>

#### To get DIffAE convergence plots

<pre>
```
python diffae/DiffAEConvergencePlots.py
```
</pre>

# Code for attacks on MAE

#### To set up the environment for attack on MAE


<pre>
```
conda env create -f environment3.yml
```
</pre>




#### To get imagenet dataset for adversarial attacks on MAEs:

Download the imagenet subset of 10k images from hugging face (https://huggingface.co/datasets/Oztobuzz/ImageNet_10k/tree/main/data ).

There are 4 files train-00000-of-00004.parquet, train-00001-of-00004.parquet, train-00002-of-00004.parquet, train-00003-of-00004.parquet

Make a directory called mae/imagenetparaquet and add these files .

Make a directory called mae/imagenetDataSubset and run the below code

<pre>
```
python mae/datasetSaver.py
```
</pre>


#### To run universal attacks on MAE

We consider masked autoencoder implementation and pretrained weights from https://github.com/facebookresearch/mae 

<pre>
```
export CUDA_VISIBLE_DEVICES=0
cd mae/demo
conda activate mae5
python mae/MaeUniversalAttack.py --attck_type "oa_l2_kf_mcmc" --desired_norm_l_inf 0.07 --set_mask_ratio 0.75 --learningRate 0.01
```
</pre>

Replace --attck_type with any of :  "la_l2_kf", "la_wass_kf", "la_cos_kf", "oa_l2_kf", "oa_wass_kf", "oa_cos_kf", "lgr_l2_kf", "lgr_wass_kf", "lgr_cos_kf", "grill_l2_kf_only_decodings", "grill_wass_kf_only_decodings", "grill_cos_kf_only_decodings" to perform universal attacks. 
Change --desired_norm_l_inf as per the requirement. ideally between 0.05 and 0.1. The default value of mask ratio will be set to --set_mask_ratio 0.75 based on the reulsts from the paper https://arxiv.org/abs/2111.06377 . 


#### To run universal adaptive attacks on MAE

<pre>
```
python mae/MaeUniversalAdaptiveAttack.py --attck_type "oa_l2_kf_mcmc" --desired_norm_l_inf 0.09 --set_mask_ratio 0.75 --learningRate 0.01
python mae/MaeUniversalAdaptiveAttack.py --attck_type "grill_l2_kf_only_decodings_mcmc" --desired_norm_l_inf 0.05 --set_mask_ratio 0.75 --learningRate 0.01
```
</pre>

#### To get quantitative damage distribution plots for MAE for classic and adaptive and universal contexts 

<pre>
```
python mae/maeAttackQuantitativeMeanStdVarPlotsRunningTest.py --set_mask_ratio 0.75 --learningRate 0.01
python mae/maeAttackQuantitativeMeanStdVarPlotsRunningTestMCMC.py --set_mask_ratio 0.75 --learningRate 0.01
```
</pre>


#### To get qualitative outputs for universal classic and adaptive attacks with and without HMC defense 

<pre>
```
python mae/maeAttackQualitativeImagePlotting.py --attck_type "grill_cos_kf_only_decodings" --desired_norm_l_inf 0.09 --set_mask_ratio 0.75 --learningRate 0.01
python mae/maeAttackQualitativeImagePlotting.py --attck_type "grill_cos_kf_only_decodings_mcmc" --desired_norm_l_inf 0.09 --set_mask_ratio 0.75 --learningRate 0.01

python mae/maeAttackQualitativeImagePlotting.py --attck_type "grill_l2_kf_only_decodings" --desired_norm_l_inf 0.09 --set_mask_ratio 0.75 --learningRate 0.01
python maeAttackQualitativeImagePlotting.py --attck_type "grill_l2_kf_only_decodings_mcmc" --desired_norm_l_inf 0.09 --set_mask_ratio 0.75 --learningRate 0.01

python mae/maeAttackQualitativeImagePlotting.py --attck_type "oa_l2_kf" --desired_norm_l_inf 0.09 --set_mask_ratio 0.75 --learningRate 0.01
python mae/maeAttackQualitativeImagePlotting.py --attck_type "oa_l2_kf_mcmc" --desired_norm_l_inf 0.09 --set_mask_ratio 0.75 --learningRate 0.01
```
</pre>


#### To get plots of condition numbers and singular values
<pre>
```
python mae/maeConditionAnalysis.py
```
</pre>


#### To do sample specific attacks on MAEs

<pre>
```
python mae/MaeSampleSpecificAttacks.py --attck_type "la_cos_kf_SS" --desired_norm_l_inf 0.9 --set_mask_ratio 0.75 --learningRate 0.01
```
</pre>


Use any of the following:  "la_l2_kf_SS", "la_wass_kf_SS", "la_cos_kf_SS", "oa_l2_kf_SS", "oa_wass_kf_SS", "oa_cos_kf_SS", "lgr_l2_kf", "lgr_wass_kf", "lgr_cos_kf", "grill_l2_kf_only_decodings_SS", "grill_wass_kf_only_decodings_SS", "grill_cos_kf_only_decodings_SS" as --attck_type. 


# Code for attacks on Gemma 3

Download Gemma 3-4b from hugging face and save the downloaded files in `illcond/gemma_attack/Gemma3-4b` and set up the environment using the below the commands:


<pre>
```
conda create -n gemma3 python=3.10 -y
export PYTHONNOUSERSITE=1
conda activate gemma3
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --index-url https://download.pytorch.org/whl/cu121 "torch>=2.6.0" "torchvision>=0.21.0"
python -c "import torch; print(torch.__version__, 'cuda', torch.cuda.is_available())"
python -m pip install --index-url https://download.pytorch.org/whl/cu121   torch==2.5.1+cu121 torchvision==0.20.1+cu121
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -m pip install --upgrade "transformers==4.50.3"
python -c "import transformers; print('transformers', transformers.__version__)"
```
</pre>

#### To do inference on Gemma 3: 

In `gemma_attack/gemma3Inference.py`: 

Input the address of the image path to `IMAGE_PATH` for example  `IMAGE_PATH = "gemma_attack/outputsStorage/walker/adv_ORIG_attackType_grill_wass_lr_0.001_eps_0.02.png"`
Input the text prompt to `QUESTION` for example `QUESTION = "What is shown in this image?"`

The run the below code for inference : 

`python gemma_attack/gemma3Inference.py`


#### To perform adversarial attacks on Gemma 3:


<pre>
```
export CUDA_VISIBLE_DEVICES=0
conda activate gemma3
cd illcond
python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 10000 --attackSample light
```
</pre>


Repeat the same for other values of $L_\inf$ norms and other data samples by updating --desired_norm_l_inf and --attackSample . Use blackHole, boat, cheetah, light, walker and nature which are already available as images in the repository.


#### TO get plots of Gemma 3 layerwise condition numbers and singular values:

Run : `python gemma_attack/gemma3Conditioning.py `
