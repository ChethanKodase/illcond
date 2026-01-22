


'''

conda activate vlmAttack
export PYTHONNOUSERSITE=1
python vlm_attack.py --attck_type OA_l2 --desired_norm_l_inf 0.1 --learningRate 0.001





'''

#!/usr/bin/env python
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch.nn as nn
import torch.nn.functional as F


import random
import os


def set_seed(seed: int = 0):
    # Python RNG
    random.seed(seed)

    # Numpy RNG
    np.random.seed(seed)

    # PyTorch CPU RNG
    torch.manual_seed(seed)

    # PyTorch CUDA RNGs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Ensure hash randomness does not affect reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

# call it:
set_seed(42)


import argparse

parser = argparse.ArgumentParser(description='Adversarial attack on VLM')
parser.add_argument('--attck_type', type=str, default="lip", help='Segment index')
parser.add_argument('--desired_norm_l_inf', type=float, default="lip", help='Segment index')
parser.add_argument('--learningRate', type=float, default="lip", help='Segment index')

args = parser.parse_args()


attck_type = args.attck_type
desired_norm_l_inf = args.desired_norm_l_inf
learningRate = args.learningRate

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
MODEL_PATH = "Qwen2.5-VL-7B-Instruct"
IMAGE_PATH = "dataSamples/dogs68.jpg"




DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16  # you can also try torch.float16

criterion = nn.MSELoss()


def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()

# -------------------------------------------------------------------
# BUILD MULTIMODAL INPUTS (uses PIL internally, but outside grad path)
# -------------------------------------------------------------------
def build_inputs(processor, image_path: str, question: str):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    return inputs.to(DEVICE)


# -------------------------------------------------------------------
# CLEAN GENERATION (no gradients)
# -------------------------------------------------------------------
def run_clean_generation(model, processor, inputs, max_new_tokens: int = 128):
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    input_ids = inputs["input_ids"]
    gen_only = output_ids[:, input_ids.shape[1]:]
    texts = processor.batch_decode(
        gen_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return texts[0]



def getOALoss(outputs,outputsN):
    return criterion(outputs.logits, outputsN.logits)


# -------------------------------------------------------------------
# ADAM ATTACK IN PATCH (pixel_values) SPACE
# -------------------------------------------------------------------
def adam_attack_on_patches(
    model,
    original_inputs,
    num_steps: int = 3,   # keep small; bump up if it runs fine
    lr: float = 0.05,
    epsilon: float = 0.2,
):
    """
    Iterative adversarial attack on `pixel_values` using Adam.

    We optimize a noise tensor `delta` such that:
        adv_pixel = orig_pixel_values + delta
    and clamp delta to [-epsilon, epsilon] in this patch space.

    Objective: maximize language modeling loss (untargeted attack).
    """

    # Clone inputs to avoid modifying original dict
    inputs = {
        k: (v.clone().to(DEVICE) if torch.is_tensor(v) else v)
        for k, v in original_inputs.items()
    }

    if "pixel_values" not in inputs:
        raise ValueError("pixel_values not found in inputs. Check build_inputs().")

    orig_pixel_values = inputs["pixel_values"].detach()

    # Initialize noise (delta) as zeros
    #delta = torch.zeros_like(orig_pixel_values, device=DEVICE, dtype=orig_pixel_values.dtype)
    delta = 0.01 * torch.randn_like(orig_pixel_values, device=DEVICE, dtype=orig_pixel_values.dtype)

    delta.requires_grad_(True)

    optimizer = torch.optim.Adam([delta], lr=lr)

    # All other inputs remain fixed
    static_keys = [k for k in inputs.keys() if k != "pixel_values"]
    static_inputs = {k: inputs[k] for k in static_keys}


    model.train()
    lossesList = [0.0]
    for step in range(num_steps):
        # Build adversarial pixel_values within epsilon-ball
        adv_pixel = orig_pixel_values + delta
        adv_pixel = torch.max(
            torch.min(adv_pixel, orig_pixel_values + epsilon),
            orig_pixel_values - epsilon,
        )

        # Build model inputs for this step
        step_inputs = dict(static_inputs)
        step_inputs["pixel_values"] = adv_pixel
        step_inputs["labels"] = static_inputs["input_ids"]
        step_inputs["use_cache"] = False  # critical to save memory

        step_inputsN = dict(static_inputs)
        step_inputsN["pixel_values"] = orig_pixel_values
        step_inputsN["labels"] = static_inputs["input_ids"]
        step_inputsN["use_cache"] = False  # critical to save memory

        # Forward
        outputs = model(**step_inputs)
        outputsN = model(**step_inputsN)

        if attck_type=="OA_l2":
            loss = getOALoss(outputs, outputsN)


        '''loss = 0
        for hiddenState, hiddenStateN in zip(outputs.hidden_states,outputsN.hidden_states):
            loss = loss + criterion(hiddenState, hiddenStateN)

        loss = loss * criterion(outputs.logits, outputsN.logits)'''

        #loss = outputs.loss  # we want to MAXIMIZE this

        # Attack objective: maximize loss -> minimize (-loss)
        attack_loss = -loss

        optimizer.zero_grad(set_to_none=True)
        attack_loss.backward()
        optimizer.step()

        # Project delta back into [-epsilon, epsilon] after Adam update
        with torch.no_grad():
            delta.data.clamp_(-epsilon, epsilon)

        print(f"[Adam step {step+1}/{num_steps}] loss = {loss.item():.4f}")
        print("lossesList", lossesList)
        #print("max(lossesList)", max(lossesList))
        if max(lossesList) < loss.item():
            #print("ever came here ? ")
            consideredDelta = delta
            # Explicitly free step locals & cached memory
            lossesList.append(loss.item())
            np.save(f"outputsStorage/convergence/vlm_attack_type__given_attackType_{attck_type}_lr_{learningRate}_epsilon_{desired_norm_l_inf}_.npy", lossesList)

        del outputs, loss, attack_loss, adv_pixel, step_inputs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Final adversarial pixel_values

    with torch.no_grad():
        adv_pixel_final = orig_pixel_values + consideredDelta
        adv_pixel_final = torch.max(
            torch.min(adv_pixel_final, orig_pixel_values + epsilon),
            orig_pixel_values - epsilon,
        )

    return adv_pixel_final


# -------------------------------------------------------------------
# UNPATCHIFY & SAVE ADVERSARIAL IMAGE
# -------------------------------------------------------------------
def save_adv_image_from_pixel_values(
    adv_pixel_values: torch.Tensor,
    inputs,
    processor,
    save_path: str = "adv_image.png",
):
    """
    Approximate inverse of the Qwen2-VL image preprocessing:

    pixel_values (num_patches, D) -> normalized video tensor -> RGB image (H, W, 3).

    This reconstructs the model-resolution image (after resize),
    not the original 5184x3888 resolution.
    """
    ip = processor.image_processor

    # Bring to CPU & float32 for safe reshaping / math
    pv = adv_pixel_values.detach().cpu().float()

    # Read grid (T, H_grid, W_grid)
    grid_thw = inputs["image_grid_thw"][0].cpu().tolist()
    grid_t, grid_h, grid_w = grid_thw  # T_grid, H_grid, W_grid

    # Hyperparameters from image processor
    patch_size = ip.patch_size           # typically 14
    temporal_patch_size = ip.temporal_patch_size  # typically 2
    merge_size = ip.merge_size           # typically 2
    channel = 3

    num_patches, D = pv.shape
    assert num_patches == grid_t * grid_h * grid_w, \
        f"num_patches mismatch: {num_patches} vs {grid_t*grid_h*grid_w}"
    assert D == channel * temporal_patch_size * patch_size * patch_size, \
        f"patch dim mismatch: {D} vs {channel*temporal_patch_size*patch_size*patch_size}"

    # Step 1: undo final flatten -> 9D tensor in permuted space
    patches = pv.view(
        grid_t,
        grid_h // merge_size,
        grid_w // merge_size,
        merge_size,
        merge_size,
        channel,
        temporal_patch_size,
        patch_size,
        patch_size,
    )

    # Step 2: invert the permute from the official preprocessing
    # Forward permute was: (0, 3, 6, 4, 7, 2, 1, 5, 8)
    # So we map back to original dims: (0, 1, 2, 3, 4, 5, 6, 7, 8)
    patches_orig = patches.permute(0, 6, 5, 1, 3, 7, 2, 4, 8).contiguous()
    # patches_orig shape:
    # (grid_t, temporal_patch_size, channel,
    #  grid_h // merge_size, merge_size, patch_size,
    #  grid_w // merge_size, merge_size, patch_size)

    # Step 3: collapse spatial/merge dims back into H', W'
    H_resized = (grid_h // merge_size) * merge_size * patch_size
    W_resized = (grid_w // merge_size) * merge_size * patch_size

    vid = patches_orig.view(
        grid_t,
        temporal_patch_size,
        channel,
        H_resized,
        W_resized,
    )
    # vid shape: (grid_t, temporal_patch_size, C, H, W)

    # Step 4: collapse temporal grid into full frames dimension
    vid = vid.view(grid_t * temporal_patch_size, channel, H_resized, W_resized)
    # For images, Qwen2-VL uses 2 identical frames; take the first one
    frame0 = vid[0]  # (C, H, W), still normalized

    # Step 5: denormalize (inverse of CLIP normalization)
    mean = torch.tensor(ip.image_mean, dtype=torch.float32).view(channel, 1, 1)
    std = torch.tensor(ip.image_std, dtype=torch.float32).view(channel, 1, 1)

    img = frame0 * std + mean  # back to [0,1] approx
    img = img.clamp(0.0, 1.0)

    # Step 6: to uint8 and save
    img_np = (img.numpy().transpose(1, 2, 0) * 255.0).round().clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    pil_img.save(save_path)
    print(f"Saved adversarial image (model-resolution) to: {save_path}")


# -------------------------------------------------------------------
# RUN GENERATION WITH ADVERSARIAL PATCHES
# -------------------------------------------------------------------
def run_adversarial_generation(
    model,
    processor,
    original_inputs,
    adv_pixel_values,
    max_new_tokens: int = 128,
):
    model.eval()
    adv_inputs = {
        k: (v.clone().to(DEVICE) if torch.is_tensor(v) else v)
        for k, v in original_inputs.items()
    }
    adv_inputs["pixel_values"] = adv_pixel_values

    with torch.no_grad():
        output_ids = model.generate(
            **adv_inputs,
            max_new_tokens=max_new_tokens,
        )

    input_ids = adv_inputs["input_ids"]
    gen_only = output_ids[:, input_ids.shape[1]:]
    texts = processor.batch_decode(
        gen_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return texts[0]


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    print("Loading model and processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        device_map=None,
    ).to(DEVICE)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    question = "What is shown in this image?"

    print("Building inputs (one-time preprocessing with PIL)...")
    inputs = build_inputs(processor, IMAGE_PATH, question)

    print("\n=== CLEAN (NO ATTACK) OUTPUT ===")
    clean_text = run_clean_generation(model, processor, inputs)
    print(clean_text)

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("\nRunning Adam-based iterative attack on patch tensor (pixel_values)...")
    adv_pixel_values = adam_attack_on_patches(
        model,
        inputs,
        num_steps=2,   # start small; if OK, try 4–5
        lr=learningRate,
        epsilon=desired_norm_l_inf # it was 0.05 before
    )

    # --- NEW: save adversarial image reconstructed from adv_pixel_values ---
    save_adv_image_from_pixel_values(
        adv_pixel_values,
        inputs,
        processor,
        save_path=f"outputsStorage/adv_imagegiven_attackType_{attck_type}_lr_{learningRate}_epsilon_{desired_norm_l_inf}_.png",
    )

    print("\n=== ADVERSARIAL OUTPUT (PATCH-SPACE, ADAM ATTACK) ===")
    adv_text = run_adversarial_generation(model, processor, inputs, adv_pixel_values)
    print(adv_text)

    # --------------------------------------------------------
    # SAVE TO TEXT FILE
    # --------------------------------------------------------
    filename = f"outputsStorage/output_given_attackType_{attck_type}_lr_{learningRate}_epsilon_{desired_norm_l_inf}_.txt"
    with open(filename, "w") as f:
        f.write("=== CLEAN OUTPUT ===\n")
        f.write(clean_text + "\n\n")
        f.write("=== ADVERSARIAL OUTPUT ===\n")
        f.write(adv_text + "\n")

    print(f"\nSaved outputs to: {filename}")


if __name__ == "__main__":
    main()