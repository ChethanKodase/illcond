import torch
import numpy as np
import torch.nn as nn


def get_layer_pert_recon(model):
    g_cond_nums = []
    for i, block in enumerate(model.ema_model.encoder.input_blocks):  

        b_cond_nums = []
        for name, param in block.named_parameters():
            if "weight" in name:
                original_param_wt = param.clone()
                if (len(original_param_wt.shape)==4):
                    W_matrix = original_param_wt.view(original_param_wt.shape[0], -1)  # Flatten kernels into a 2D matrix
                    U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
                    condition_number = S.max() / S.min()
                    b_cond_nums.append(condition_number.item())
                else:
                    b_cond_nums.append(1.0)

        b_cond_nums = np.array(b_cond_nums)
        b_mean_cond = np.mean(b_cond_nums)
        g_cond_nums.append(b_mean_cond)

    for i, block in enumerate(model.ema_model.encoder.middle_block):
        b_cond_nums = []
        for name, param in block.named_parameters():
            if "weight" in name:
                original_param_wt = param.clone()
                if (len(original_param_wt.shape)==4):
                    W_matrix = original_param_wt.view(original_param_wt.shape[0], -1)  # Flatten kernels into a 2D matrix
                    U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
                    condition_number = S.max() / S.min()
                    b_cond_nums.append(condition_number.item())
                else:
                    b_cond_nums.append(1.0)

        b_cond_nums = np.array(b_cond_nums)
        b_mean_cond = np.mean(b_cond_nums)
        g_cond_nums.append(b_mean_cond)

    for i, block in enumerate(model.ema_model.encoder.out):
        b_cond_nums = []
        for name, param in block.named_parameters():
            if "weight" in name:
                original_param_wt = param.clone()
                if (len(original_param_wt.shape)==4):
                    W_matrix = original_param_wt.view(original_param_wt.shape[0], -1)  # Flatten kernels into a 2D matrix
                    U, S, Vt = torch.linalg.svd(W_matrix, full_matrices=False)
                    condition_number = S.max() / S.min()
                    b_cond_nums.append(condition_number.item())
                else:
                    condition_number = 1.0
                    b_cond_nums.append(condition_number)
            else:
                b_cond_nums.append(1.0)
        if isinstance(block, (nn.SiLU, nn.AdaptiveAvgPool2d, nn.Flatten)):
            b_cond_nums = [1.0]
        b_cond_nums = np.array(b_cond_nums)
        b_mean_cond = np.mean(b_cond_nums)
        g_cond_nums.append(b_mean_cond)

    cond_nums_array = np.array(g_cond_nums)

    cond_nums_array[cond_nums_array>50.0]=50.0
    cond_nums_normalized = (cond_nums_array) / np.sum(cond_nums_array)

    return cond_nums_normalized