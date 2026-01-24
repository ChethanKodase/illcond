



'''

export CUDA_VISIBLE_DEVICES=0
conda activate gemma3
cd illcond
python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.5 --learningRate 0.001 --num_steps 10000 --attackSample light
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.01 --learningRate 0.001 --num_steps 10000 --attackSample light

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 10000 --attackSample light
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 10000 --attackSample light

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.03 --learningRate 0.001 --num_steps 10000 --attackSample light
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.03 --learningRate 0.001 --num_steps 10000 --attackSample light

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.04 --learningRate 0.001 --num_steps 10000 --attackSample light
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.04 --learningRate 0.001 --num_steps 10000 --attackSample light

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.05 --learningRate 0.001 --num_steps 10000 --attackSample light
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.05 --learningRate 0.001 --num_steps 10000 --attackSample light

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.06 --learningRate 0.001 --num_steps 10000 --attackSample light
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.06 --learningRate 0.001 --num_steps 10000 --attackSample light

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.07 --learningRate 0.001 --num_steps 10000 --attackSample light
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.07 --learningRate 0.001 --num_steps 10000 --attackSample light

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.08 --learningRate 0.001 --num_steps 10000 --attackSample light
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.08 --learningRate 0.001 --num_steps 10000 --attackSample light

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.09 --learningRate 0.001 --num_steps 10000 --attackSample light
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.09 --learningRate 0.001 --num_steps 10000 --attackSample light

##########################################################################################################################################################################################################################################################################################

export CUDA_VISIBLE_DEVICES=1
conda activate gemma3
cd gemma_attack
python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.01 --learningRate 0.001 --num_steps 10000 --attackSample astronauts
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.01 --learningRate 0.001 --num_steps 10000 --attackSample astronauts

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 10000 --attackSample astronauts
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 10000 --attackSample astronauts

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.03 --learningRate 0.001 --num_steps 10000 --attackSample astronauts
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.03 --learningRate 0.001 --num_steps 10000 --attackSample astronauts

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.04 --learningRate 0.001 --num_steps 10000 --attackSample astronauts
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.04 --learningRate 0.001 --num_steps 10000 --attackSample astronauts

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.05 --learningRate 0.001 --num_steps 10000 --attackSample astronauts
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.05 --learningRate 0.001 --num_steps 10000 --attackSample astronauts

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.06 --learningRate 0.001 --num_steps 10000 --attackSample astronauts
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.06 --learningRate 0.001 --num_steps 10000 --attackSample astronauts

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.07 --learningRate 0.001 --num_steps 10000 --attackSample astronauts
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.07 --learningRate 0.001 --num_steps 10000 --attackSample astronauts

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.08 --learningRate 0.001 --num_steps 10000 --attackSample astronauts
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.08 --learningRate 0.001 --num_steps 10000 --attackSample astronauts

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.09 --learningRate 0.001 --num_steps 10000 --attackSample astronauts
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.09 --learningRate 0.001 --num_steps 10000 --attackSample astronauts



##########################################################################################################################################################################################################################################################################################

export CUDA_VISIBLE_DEVICES=2
conda activate gemma3
cd gemma_attack
python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.01 --learningRate 0.001 --num_steps 10000 --attackSample blackHole
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.01 --learningRate 0.001 --num_steps 10000 --attackSample blackHole

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 10000 --attackSample blackHole
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 10000 --attackSample blackHole

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.03 --learningRate 0.001 --num_steps 10000 --attackSample blackHole
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.03 --learningRate 0.001 --num_steps 10000 --attackSample blackHole

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.04 --learningRate 0.001 --num_steps 10000 --attackSample blackHole
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.04 --learningRate 0.001 --num_steps 10000 --attackSample blackHole

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.05 --learningRate 0.001 --num_steps 10000 --attackSample blackHole
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.05 --learningRate 0.001 --num_steps 10000 --attackSample blackHole

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.06 --learningRate 0.001 --num_steps 10000 --attackSample blackHole
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.06 --learningRate 0.001 --num_steps 10000 --attackSample blackHole

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.07 --learningRate 0.001 --num_steps 10000 --attackSample blackHole
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.07 --learningRate 0.001 --num_steps 10000 --attackSample blackHole

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.08 --learningRate 0.001 --num_steps 10000 --attackSample blackHole
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.08 --learningRate 0.001 --num_steps 10000 --attackSample blackHole

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.09 --learningRate 0.001 --num_steps 10000 --attackSample blackHole
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.09 --learningRate 0.001 --num_steps 10000 --attackSample blackHole



##########################################################################################################################################################################################################################################################################################

export CUDA_VISIBLE_DEVICES=3
conda activate gemma3
cd gemma_attack
python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.01 --learningRate 0.001 --num_steps 10000 --attackSample boat
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.01 --learningRate 0.001 --num_steps 10000 --attackSample boat

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 10000 --attackSample boat
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 10000 --attackSample boat

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.03 --learningRate 0.001 --num_steps 10000 --attackSample boat
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.03 --learningRate 0.001 --num_steps 10000 --attackSample boat

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.04 --learningRate 0.001 --num_steps 10000 --attackSample boat
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.04 --learningRate 0.001 --num_steps 10000 --attackSample boat

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.05 --learningRate 0.001 --num_steps 10000 --attackSample boat
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.05 --learningRate 0.001 --num_steps 10000 --attackSample boat

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.06 --learningRate 0.001 --num_steps 10000 --attackSample boat
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.06 --learningRate 0.001 --num_steps 10000 --attackSample boat

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.07 --learningRate 0.001 --num_steps 10000 --attackSample boat
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.07 --learningRate 0.001 --num_steps 10000 --attackSample boat

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.08 --learningRate 0.001 --num_steps 10000 --attackSample boat
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.08 --learningRate 0.001 --num_steps 10000 --attackSample boat

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.09 --learningRate 0.001 --num_steps 10000 --attackSample boat
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.09 --learningRate 0.001 --num_steps 10000 --attackSample boat



##########################################################################################################################################################################################################################################################################################

export CUDA_VISIBLE_DEVICES=4
conda activate gemma3
cd gemma_attack
python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.01 --learningRate 0.001 --num_steps 10000 --attackSample cheetah
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.01 --learningRate 0.001 --num_steps 10000 --attackSample cheetah

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 10000 --attackSample cheetah
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 10000 --attackSample cheetah

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.03 --learningRate 0.001 --num_steps 10000 --attackSample cheetah
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.03 --learningRate 0.001 --num_steps 10000 --attackSample cheetah

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.04 --learningRate 0.001 --num_steps 10000 --attackSample cheetah
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.04 --learningRate 0.001 --num_steps 10000 --attackSample cheetah

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.05 --learningRate 0.001 --num_steps 10000 --attackSample cheetah
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.05 --learningRate 0.001 --num_steps 10000 --attackSample cheetah

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.06 --learningRate 0.001 --num_steps 10000 --attackSample cheetah
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.06 --learningRate 0.001 --num_steps 10000 --attackSample cheetah

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.07 --learningRate 0.001 --num_steps 10000 --attackSample cheetah
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.07 --learningRate 0.001 --num_steps 10000 --attackSample cheetah

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.08 --learningRate 0.001 --num_steps 10000 --attackSample cheetah
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.08 --learningRate 0.001 --num_steps 10000 --attackSample cheetah

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.09 --learningRate 0.001 --num_steps 10000 --attackSample cheetah
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.09 --learningRate 0.001 --num_steps 10000 --attackSample cheetah

##########################################################################################################################################################################################################################################################################################


export CUDA_VISIBLE_DEVICES=5
conda activate gemma3
cd gemma_attack
python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.009 --learningRate 0.001 --num_steps 10000 --attackSample walker
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.009 --learningRate 0.001 --num_steps 10000 --attackSample walker

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.008 --learningRate 0.001 --num_steps 10000 --attackSample walker
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.008 --learningRate 0.001 --num_steps 10000 --attackSample walker

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.007 --learningRate 0.001 --num_steps 10000 --attackSample walker
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.007 --learningRate 0.001 --num_steps 10000 --attackSample walker

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.006 --learningRate 0.001 --num_steps 10000 --attackSample walker
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.006 --learningRate 0.001 --num_steps 10000 --attackSample walker

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 10000 --attackSample walker
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 10000 --attackSample walker

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 10000 --attackSample walker
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 10000 --attackSample walker

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 10000 --attackSample walker
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 10000 --attackSample walker

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 10000 --attackSample walker
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 10000 --attackSample walker

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.001 --learningRate 0.001 --num_steps 10000 --attackSample walker
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.001 --learningRate 0.001 --num_steps 10000 --attackSample walker



export CUDA_VISIBLE_DEVICES=7
conda activate gemma3
cd gemma_attack
python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.009 --learningRate 0.001 --num_steps 10000 --attackSample boat
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.009 --learningRate 0.001 --num_steps 10000 --attackSample boat

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.008 --learningRate 0.001 --num_steps 10000 --attackSample boat
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.008 --learningRate 0.001 --num_steps 10000 --attackSample boat

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.007 --learningRate 0.001 --num_steps 10000 --attackSample boat
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.007 --learningRate 0.001 --num_steps 10000 --attackSample boat

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.006 --learningRate 0.001 --num_steps 10000 --attackSample boat
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.006 --learningRate 0.001 --num_steps 10000 --attackSample boat

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 10000 --attackSample boat
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 10000 --attackSample boat

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 10000 --attackSample boat
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 10000 --attackSample boat

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 10000 --attackSample boat
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 10000 --attackSample boat

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 10000 --attackSample boat
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 10000 --attackSample boat

python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.001 --learningRate 0.001 --num_steps 10000 --attackSample boat
python gemma_attack/gemma3Attack1.py --attck_type OA_wass --desired_norm_l_inf 0.001 --learningRate 0.001 --num_steps 10000 --attackSample boat

##########################################################################################################################################################################################################################################################################################


'''


# gemma3Attack1.py
# Original-image-space adversarial attack for Gemma-3 (no squeeze)
# Supports: grill_l2, grill_cos, OA_l2, OA_cos
#
# Run:
#   export CUDA_VISIBLE_DEVICES=2
#   conda activate gemma3
#   cd /data1/chethan/gemma_attack
#   python gemma_attack/gemma3Attack1.py --attck_type grill_wass --desired_norm_l_inf 0.06 --learningRate 0.001 --num_steps 10000

import os
import sys
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)

criterion = nn.MSELoss()


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

# ----------------------------
# Losses: GRILL + OA
# ----------------------------
def get_grill_l2(outputs, outputsN):
    loss = 0.0
    for h, hn in zip(outputs.hidden_states, outputsN.hidden_states):
        loss = loss + criterion(h, hn)
    return loss * criterion(h, hn)


def get_grill_wass(outputs, outputsN):
    loss = 0.0
    for h, hn in zip(outputs.hidden_states, outputsN.hidden_states):
        loss = loss + wasserstein_distance(h, hn)
    return loss * wasserstein_distance(h, hn)


def get_grill_cos(outputs, outputsN):
    loss = 0.0
    for h, hn in zip(outputs.hidden_states, outputsN.hidden_states):
        loss = loss + (1.0 - cos(h, hn)) ** 2
    return loss * (1.0 - cos(outputs.logits, outputsN.logits)) ** 2


def get_oa_l2(outputs, outputsN):
    return criterion(outputs.logits, outputsN.logits)


def get_oa_wass(outputs, outputsN):
    return wasserstein_distance(outputs.logits, outputsN.logits)


def get_oa_cos(outputs, outputsN):
    return (1.0 - cos(outputs.logits, outputsN.logits)) ** 2


# ----------------------------
# Utilities: image <-> tensor
# ----------------------------
def pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    """PIL RGB -> torch float tensor in [0,1], shape (1,3,H,W)"""
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    return t


def tensor01_to_pil(t01: torch.Tensor) -> Image.Image:
    """torch tensor [0,1], shape (1,3,H,W) or (3,H,W) -> PIL RGB"""
    if t01.dim() == 4:
        t01 = t01[0]
    t01 = t01.detach().cpu().clamp(0, 1)
    arr = (t01.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ----------------------------
# Differentiable preprocessing (approx Gemma)
# ----------------------------
def _get_target_hw(image_processor):
    """
    Try to infer model target H,W from HF image_processor.
    We handle common formats:
      - ip.size = {"height": H, "width": W}
      - ip.size = {"shortest_edge": S}
      - ip.size = S (int)
      - ip.crop_size likewise
    """
    ip = image_processor
    target_h = target_w = None

    crop = getattr(ip, "crop_size", None)
    if isinstance(crop, dict):
        target_h = crop.get("height", None)
        target_w = crop.get("width", None)
    elif isinstance(crop, int):
        target_h = target_w = crop

    if target_h is None or target_w is None:
        size = getattr(ip, "size", None)
        if isinstance(size, dict):
            if "height" in size and "width" in size:
                target_h = size["height"]
                target_w = size["width"]
            elif "shortest_edge" in size:
                target_h = target_w = size["shortest_edge"]
        elif isinstance(size, int):
            target_h = target_w = size

    if target_h is None or target_w is None:
        # fallback for many gemma/vlm configs
        target_h = target_w = 896

    return int(target_h), int(target_w)


def resize_keep_aspect_center_crop(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Differentiable:
      - scale so that resized image >= target in both dims
      - center crop to (target_h, target_w)
    """
    _, _, H, W = x.shape
    scale = max(target_h / H, target_w / W)
    newH = int(round(H * scale))
    newW = int(round(W * scale))

    x_resized = F.interpolate(x, size=(newH, newW), mode="bilinear", align_corners=False)

    top = max((newH - target_h) // 2, 0)
    left = max((newW - target_w) // 2, 0)
    x_crop = x_resized[:, :, top:top + target_h, left:left + target_w]

    # pad if needed (unlikely)
    pad_h = target_h - x_crop.shape[2]
    pad_w = target_w - x_crop.shape[3]
    if pad_h > 0 or pad_w > 0:
        x_crop = F.pad(x_crop, (0, max(pad_w, 0), 0, max(pad_h, 0)))

    return x_crop


def normalize_like_processor(x01: torch.Tensor, image_processor) -> torch.Tensor:
    mean = torch.tensor(image_processor.image_mean, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    std = torch.tensor(image_processor.image_std, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    return (x01 - mean) / std


def gemma_preprocess_differentiable(x01: torch.Tensor, processor) -> torch.Tensor:
    """
    Differentiable approximation of the processor's image pipeline.
    Produces pixel_values like the processor would (shape 1x3xH'xW').
    """
    ip = processor.image_processor
    th, tw = _get_target_hw(ip)
    x = resize_keep_aspect_center_crop(x01, th, tw)
    x = normalize_like_processor(x, ip)
    return x


# ----------------------------
# Build template inputs ONCE (IMPORTANT)
# Ensures image placeholder tokens exist in input_ids
# ----------------------------
def build_template_inputs(processor, question: str, pil_image: Image.Image, device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Pass an image ONCE so the processor inserts the correct special image token(s)
    template = processor(text=[prompt], images=[pil_image], return_tensors="pt")
    template = {k: v.to(device) if torch.is_tensor(v) else v for k, v in template.items()}
    return template


# ----------------------------
# Generation helper (uses template, swaps pixel_values)
# ----------------------------
def run_generation_with_pixel_values(model, processor, template_inputs, pixel_values, max_new_tokens=128):
    model.eval()
    inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
    inputs["pixel_values"] = pixel_values

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # deterministic
        )

    input_ids = inputs["input_ids"]
    gen_only = out_ids[:, input_ids.shape[1]:]
    return processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]


# ----------------------------
# ORIGINAL-SPACE Adam attack
# ----------------------------
def adam_attack_original_space(
    model,
    processor,
    template_inputs,
    x_orig01,               # (1,3,H0,W0) in [0,1]
    attck_type: str,
    num_steps: int,
    lr: float,
    epsilon: float,         # L_inf bound in ORIGINAL pixel space [0,1]
    device,
    save_conv_path: str,
):
    """
    Optimize delta in ORIGINAL image pixel space (no squeeze):
        x_adv01 = clamp(x_orig01 + delta, 0, 1)
        ||delta||_inf <= epsilon

    Each step:
        pixel_values_adv = preprocess(x_adv01)  (differentiable)
        pixel_values_clean = preprocess(x_orig01)
        outputs_adv vs outputs_clean for GRILL/OA losses
    """
    x_orig01 = x_orig01.detach().to(device)

    # delta init
    delta = 0.001 * torch.randn_like(x_orig01, device=device)
    delta.requires_grad_(True)

    opt = torch.optim.Adam([delta], lr=lr)

    losses_list = [0.0]
    best_loss = -1e18
    best_delta = delta.detach().clone()

    model.train()

    # Precompute clean pixel_values once (no grads needed)
    with torch.no_grad():
        pv_clean_fixed = gemma_preprocess_differentiable(x_orig01, processor)

    for step in range(num_steps):
        # original-space adv image with L_inf constraint
        x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
        x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

        # preprocess adv (must be differentiable)
        pv_adv = gemma_preprocess_differentiable(x_adv01, processor)

        adv_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
        adv_inputs["pixel_values"] = pv_adv
        adv_inputs["labels"] = template_inputs["input_ids"]
        adv_inputs["use_cache"] = False

        clean_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
        clean_inputs["pixel_values"] = pv_clean_fixed
        clean_inputs["labels"] = template_inputs["input_ids"]
        clean_inputs["use_cache"] = False

        outputs = model(**adv_inputs, output_hidden_states=True, return_dict=True)
        with torch.no_grad():
            outputsN = model(**clean_inputs, output_hidden_states=True, return_dict=True)

        if attck_type == "grill_l2":
            loss = get_grill_l2(outputs, outputsN)
        elif attck_type == "grill_wass":
            loss = get_grill_wass(outputs, outputsN)
        elif attck_type == "grill_cos":
            loss = get_grill_cos(outputs, outputsN)
        elif attck_type == "OA_l2":
            loss = get_oa_l2(outputs, outputsN)
        elif attck_type == "OA_wass":
            loss = get_oa_wass(outputs, outputsN)
        elif attck_type == "OA_cos":
            loss = get_oa_cos(outputs, outputsN)
        else:
            raise ValueError("attck_type must be one of: grill_l2, grill_cos, OA_l2, OA_cos")

        attack_loss = -loss  # maximize loss

        opt.zero_grad(set_to_none=True)
        attack_loss.backward()
        opt.step()

        # keep delta inside [-epsilon, epsilon]
        with torch.no_grad():
            delta.data.clamp_(-epsilon, epsilon)

        lv = float(loss.item())
        if (step + 1) % 10 == 0 or step == 0:
            print(f"[step {step+1}/{num_steps}] loss={lv:.6f}")

        if lv > best_loss:
            best_loss = lv
            best_delta = delta.detach().clone()
            losses_list.append(lv)
            np.save(save_conv_path, np.array(losses_list, dtype=np.float32))

        # cleanup
        del outputs, outputsN, loss, attack_loss, pv_adv, adv_inputs, clean_inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with torch.no_grad():
        x_adv01_final = (x_orig01 + best_delta).clamp(0.0, 1.0)
        x_adv01_final = torch.max(torch.min(x_adv01_final, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

    return x_adv01_final


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Gemma-3 ORIGINAL-image-space adversarial attack (no squeeze)")
    parser.add_argument("--attck_type", type=str, default="grill_l2",
                        help="grill_l2 | grill_cos | OA_l2 | OA_cos")
    parser.add_argument("--desired_norm_l_inf", type=float, default=0.03,
                        help="epsilon L_inf in ORIGINAL pixel space [0..1]. Try 0.01~0.08")
    parser.add_argument("--learningRate", type=float, default=1e-3,
                        help="Adam learning rate")
    parser.add_argument("--num_steps", type=int, default=2000,
                        help="Number of Adam steps")
    parser.add_argument("--attackSample", type=str, default="nature",
                    help="which sample")
    args = parser.parse_args()

    attck_type = args.attck_type
    epsilon = float(args.desired_norm_l_inf)
    lr = float(args.learningRate)
    num_steps = int(args.num_steps)
    attackSample = str(args.attackSample)

    # ---- CONFIG (change if needed)
    #attackSample = "nature"
    MODEL_PATH = "gemma_attack/Gemma3-4b"
    IMAGE_PATH = f"gemma_attack/dataSamples/{attackSample}68.jpg"
    #IMAGE_PATH = f"gemma_attack/dataSamples/interference68.jpeg"
    
    QUESTION = "What is shown in this image?"
    MAX_NEW_TOKENS = 128

    os.makedirs("gemma_attack/outputsStorage", exist_ok=True)
    os.makedirs(f"gemma_attack/outputsStorage/convergence/{attackSample}", exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"device={device}, dtype={dtype}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, padding_side="left")

    print("Loading model...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    # Load original image (keep original resolution)
    pil = Image.open(IMAGE_PATH).convert("RGB")
    x_orig01 = pil_to_tensor01(pil).to(device)

    # Build template inputs ONCE (inserts image tokens in input_ids)
    template_inputs = build_template_inputs(processor, QUESTION, pil, device)

    # Clean output: preprocess original (differentiable) then generate
    pv_clean = gemma_preprocess_differentiable(x_orig01, processor)

    print("\n=== CLEAN OUTPUT ===")
    clean_text = run_generation_with_pixel_values(model, processor, template_inputs, pv_clean, max_new_tokens=MAX_NEW_TOKENS)
    print(clean_text)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Attack
    conv_path = f"gemma_attack/outputsStorage/convergence/{attackSample}/gemma_ORIG_attack_{attck_type}_lr_{lr}_eps_{epsilon}.npy"
    x_adv01 = adam_attack_original_space(
        model=model,
        processor=processor,
        template_inputs=template_inputs,
        x_orig01=x_orig01,
        attck_type=attck_type,
        num_steps=num_steps,
        lr=lr,
        epsilon=epsilon,
        device=device,
        save_conv_path=conv_path,
    )

    # Save ORIGINAL-resolution adversarial image (no squeeze)
    adv_img_path = f"gemma_attack/outputsStorage/{attackSample}/adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}.png"
    tensor01_to_pil(x_adv01).save(adv_img_path)
    print(f"\nSaved ORIGINAL-resolution adversarial image to: {adv_img_path}")

    # Adv output (preprocess adv then generate)
    pv_adv = gemma_preprocess_differentiable(x_adv01, processor)
    print("\n=== ADVERSARIAL OUTPUT ===")
    adv_text = run_generation_with_pixel_values(model, processor, template_inputs, pv_adv, max_new_tokens=MAX_NEW_TOKENS)
    print(adv_text)

    # Save text outputs
    out_txt = f"gemma_attack/outputsStorage/{attackSample}/output_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}.txt"
    with open(out_txt, "w") as f:
        f.write("=== CLEAN OUTPUT ===\n")
        f.write(clean_text + "\n\n")
        f.write("=== ADVERSARIAL OUTPUT ===\n")
        f.write(adv_text + "\n")

    print(f"\nSaved outputs to: {out_txt}")
    print(f"Saved convergence to: {conv_path}")


if __name__ == "__main__":
    main()

