
'''




export CUDA_VISIBLE_DEVICES=3
conda deactivate
cd interpretAttacks/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
for ATTACK_SAMPLE in $(seq 2 50); do
    python qwen/QwenUntargeted_GRILL_cos_flops.py --attck_type grill_cos --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done



17943893878625

'''

#!/usr/bin/env python
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import argparse
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from torch.profiler import profile, ProfilerActivity


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


criterion = nn.MSELoss()


# ----------------------------
# Loss utilities
# ----------------------------
def cos(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()


def cosVis(a, b):
    a = torch.flatten(a)
    b = torch.flatten(b)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()


def wasserstein_distance(tensor_a, tensor_b):
    a = torch.flatten(tensor_a)
    b = torch.flatten(tensor_b)
    a_sorted, _ = torch.sort(a)
    b_sorted, _ = torch.sort(b)
    return torch.mean(torch.abs(a_sorted - b_sorted))


def get_bsa_loss(outputs, outputsN):
    loss = 0.0
    for h, hn in zip(outputs.hidden_states, outputsN.hidden_states):
        cos_per_token = F.cosine_similarity(h.squeeze(0), hn.squeeze(0), dim=1)
        loss = loss + cos_per_token.sum()
    return loss


def get_bsa_flat_loss(outputs, outputsN):
    loss = 0.0
    for h, hn in zip(outputs.hidden_states, outputsN.hidden_states):
        loss = loss + (1.0 - cos(h, hn)) ** 2
    return -1.0 * loss


def get_bsa_vision_loss(acts, actsN):
    loss = 0.0
    for h, hn in zip(acts, actsN):
        cos_per_token = F.cosine_similarity(h, hn, dim=-1)
        loss = loss + cos_per_token.sum()
    return loss


def get_bsa_flat_vision_loss(acts, actsN):
    loss = 0.0
    for h, hn in zip(acts, actsN):
        loss = loss + (1.0 - cosVis(h, hn)) ** 2
    return -1.0 * loss


def getGrillCosLoss(outputs,outputsN):
    loss = 0
    for hiddenState, hiddenStateN in zip(outputs.hidden_states,outputsN.hidden_states):
        loss = loss + (1.0-cos(hiddenState, hiddenStateN))**2
    return loss * (1.0-cos(hiddenState, hiddenStateN))**2

def getGrillCosLossVis(outputs,outputsN):
    loss = 0
    for hiddenState, hiddenStateN in zip(outputs, outputsN):
        loss = loss + (1.0-cos(hiddenState, hiddenStateN))**2
    return loss * (1.0-cos(hiddenState, hiddenStateN))**2


def getGrillCosLossLanVisComb(acts, actsN, outputs,outputsN):
    loss = 0
    for hiddenState, hiddenStateN in zip(acts, actsN):
        loss = loss + (1.0-cos(hiddenState, hiddenStateN))**2

    for hiddenState, hiddenStateN in zip(outputs.hidden_states,outputsN.hidden_states):
        loss = loss + (1.0-cos(hiddenState, hiddenStateN))**2

    return loss * (1.0-cos(hiddenState, hiddenStateN))**2



# ----------------------------
# PIL / tensor helpers
# ----------------------------
def pil_to_tensor01(pil_img):
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor01_to_pil(t01):
    if t01.dim() == 4:
        t01 = t01[0]
    t01 = t01.detach().cpu().clamp(0, 1)
    arr = (
        t01.permute(1, 2, 0).numpy() * 255.0
    ).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ----------------------------
# Differentiable Qwen preprocessing
# ----------------------------
def _get_qwen_resize_hw(image_processor, H, W):
    patch_size = int(getattr(image_processor, "patch_size", 14))
    merge_size = int(getattr(image_processor, "merge_size", 2))
    factor = patch_size * merge_size

    min_pixels = int(getattr(image_processor, "min_pixels", 56 * 56))
    max_pixels = int(getattr(image_processor, "max_pixels", 28 * 28 * 1280))

    def round_by_factor(x, f):
        return int(round(x / f) * f)

    def floor_by_factor(x, f):
        return int(np.floor(x / f) * f)

    def ceil_by_factor(x, f):
        return int(np.ceil(x / f) * f)

    h_bar = max(factor, round_by_factor(H, factor))
    w_bar = max(factor, round_by_factor(W, factor))

    if h_bar * w_bar > max_pixels:
        beta = np.sqrt((H * W) / max_pixels)
        h_bar = max(factor, floor_by_factor(H / beta, factor))
        w_bar = max(factor, floor_by_factor(W / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = np.sqrt(min_pixels / (H * W))
        h_bar = max(factor, ceil_by_factor(H * beta, factor))
        w_bar = max(factor, ceil_by_factor(W * beta, factor))

    return int(h_bar), int(w_bar)


def qwen_preprocess_differentiable(x01, processor):
    ip = processor.image_processor
    _, C, H, W = x01.shape
    assert C == 3

    patch_size = int(ip.patch_size)
    temporal_patch_size = int(ip.temporal_patch_size)
    merge_size = int(ip.merge_size)

    target_h, target_w = _get_qwen_resize_hw(ip, H, W)

    x = F.interpolate(
        x01,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )

    mean = torch.tensor(ip.image_mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(ip.image_std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std

    x = x.repeat(temporal_patch_size, 1, 1, 1)

    grid_t = x.shape[0] // temporal_patch_size
    grid_h = target_h // patch_size
    grid_w = target_w // patch_size

    patches = x.view(
        grid_t,
        temporal_patch_size,
        3,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )

    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()

    pixel_values = patches.view(
        grid_t * grid_h * grid_w,
        3 * temporal_patch_size * patch_size * patch_size,
    )

    image_grid_thw = torch.tensor(
        [[grid_t, grid_h, grid_w]],
        dtype=torch.long,
        device=x01.device,
    )

    return pixel_values, image_grid_thw


# ----------------------------
# Qwen inputs
# ----------------------------
def build_template_inputs(processor, question, pil_image, device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    template = processor(
        text=[prompt],
        images=[pil_image],
        return_tensors="pt",
    )

    return {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in template.items()
    }


def run_generation_with_pixel_values(
    model,
    processor,
    template_inputs,
    pixel_values,
    image_grid_thw,
    max_new_tokens=128,
):
    model.eval()

    inputs = {
        k: v.clone() if torch.is_tensor(v) else v
        for k, v in template_inputs.items()
    }

    inputs["pixel_values"] = pixel_values
    inputs["image_grid_thw"] = image_grid_thw

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_ids = inputs["input_ids"]
    gen_only = out_ids[:, input_ids.shape[1]:]

    return processor.batch_decode(
        gen_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]


# ----------------------------
# Vision hooks: this is the important part
# ----------------------------
def run_get_image_features_with_vision_hooks(model, pixel_values, image_grid_thw):
    acts = []
    handles = []

    if hasattr(model, "model") and hasattr(model.model, "visual") and hasattr(model.model.visual, "blocks"):
        blocks = model.model.visual.blocks
    elif hasattr(model, "visual") and hasattr(model.visual, "blocks"):
        blocks = model.visual.blocks
    else:
        raise RuntimeError("Could not find Qwen vision blocks.")

    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        if torch.is_tensor(out):
            acts.append(out)

    for block in blocks:
        handles.append(block.register_forward_hook(hook_fn))

    feat = model.get_image_features(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )

    for h in handles:
        h.remove()

    if isinstance(feat, tuple):
        feat = feat[0]

    return feat, acts


# ----------------------------
# ORIGINAL-image-space BSA attack
# ----------------------------
def adam_attack_original_space(
    model,
    processor,
    template_inputs,
    x_orig01,
    attck_type,
    num_steps,
    lr,
    epsilon,
    device,
    save_conv_path,
):
    x_orig01 = x_orig01.detach().to(device)

    delta = 0.001 * torch.randn_like(x_orig01, device=device)
    delta.requires_grad_(True)

    optimizer = torch.optim.Adam([delta], lr=lr)

    losses_list = [0.0]
    best_loss = 1e18
    best_delta = delta.detach().clone()

    model.eval()
    model.config.use_cache = False
    model.config.output_hidden_states = True
    model.config.return_dict = True

    with torch.no_grad():
        pv_clean, grid_clean = qwen_preprocess_differentiable(x_orig01, processor)

        clean_inputs = {
            k: v.clone() if torch.is_tensor(v) else v
            for k, v in template_inputs.items()
        }

        clean_inputs["pixel_values"] = pv_clean
        clean_inputs["image_grid_thw"] = grid_clean
        clean_inputs["labels"] = template_inputs["input_ids"]
        clean_inputs["use_cache"] = False

        outputsN = model(
            **clean_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        _, actsN = run_get_image_features_with_vision_hooks(
            model,
            pv_clean,
            grid_clean,
        )

        print("Number of language hidden states:", len(outputsN.hidden_states))
        print("Number of vision hidden states:", len(actsN))

    adv_inputs = {
        k: v.clone() if torch.is_tensor(v) else v
        for k, v in template_inputs.items()
    }

    adv_inputs["labels"] = template_inputs["input_ids"]
    adv_inputs["use_cache"] = False

    for step in range(num_steps):
        if step == 0:
            print("entered zero ?")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_flops=True,
                record_shapes=True
            ) as prof:

                x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
                x_adv01 = torch.max(
                    torch.min(x_adv01, x_orig01 + epsilon),
                    x_orig01 - epsilon,
                ).clamp(0.0, 1.0)

                pv_adv, grid_adv = qwen_preprocess_differentiable(x_adv01, processor)

                adv_inputs["pixel_values"] = pv_adv
                adv_inputs["image_grid_thw"] = grid_adv

                outputs = model(
                    **adv_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )

                _, acts = run_get_image_features_with_vision_hooks(
                    model,
                    pv_adv,
                    grid_adv,
                )


                loss = -1 *  (getGrillCosLoss(outputs, outputsN) + getGrillCosLossVis(acts, actsN) )

                #loss = -1 *  getGrillCosLossLanVisComb(acts, actsN, outputs, outputsN)



                optimizer.zero_grad(set_to_none=True)
                loss.backward()


                lv = float(loss.item())

                if step == 0 or (step + 1) % 10 == 0:
                    print(f"[Adam step {step + 1}/{num_steps}] loss={lv:.6f}")

                if lv < best_loss:
                    best_loss = lv
                    best_delta = delta.detach().clone()
                    losses_list.append(lv)
                    np.save(save_conv_path, np.array(losses_list, dtype=np.float32))

                optimizer.step()

                with torch.no_grad():
                    delta.data.clamp_(-epsilon, epsilon)

                del outputs, acts, loss, pv_adv, grid_adv
                
            print(prof.key_averages().table(sort_by="flops", row_limit=20))

            total_flops_step0 = sum(
                evt.flops for evt in prof.key_averages() if evt.flops is not None
            )
            print(f"FLOPs for profiled attack step: {total_flops_step0}")
            print(f"Estimated FLOPs for all {num_steps} steps: {total_flops_step0 * num_steps}")
            if torch.cuda.is_available():
                print(f"Peak GPU allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
                print(f"Peak GPU reserved:  {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")

        else:
            x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
            x_adv01 = torch.max(
                torch.min(x_adv01, x_orig01 + epsilon),
                x_orig01 - epsilon,
            ).clamp(0.0, 1.0)

            pv_adv, grid_adv = qwen_preprocess_differentiable(x_adv01, processor)

            adv_inputs["pixel_values"] = pv_adv
            adv_inputs["image_grid_thw"] = grid_adv

            outputs = model(
                **adv_inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            _, acts = run_get_image_features_with_vision_hooks(
                model,
                pv_adv,
                grid_adv,
            )


            loss = -1 *  (getGrillCosLoss(outputs, outputsN) + getGrillCosLossVis(acts, actsN) )

            #loss = -1 *  getGrillCosLossLanVisComb(acts, actsN, outputs, outputsN)



            optimizer.zero_grad(set_to_none=True)
            loss.backward()


            lv = float(loss.item())

            if step == 0 or (step + 1) % 10 == 0:
                print(f"[Adam step {step + 1}/{num_steps}] loss={lv:.6f}")

            if lv < best_loss:
                best_loss = lv
                best_delta = delta.detach().clone()
                losses_list.append(lv)
                np.save(save_conv_path, np.array(losses_list, dtype=np.float32))

            optimizer.step()

            with torch.no_grad():
                delta.data.clamp_(-epsilon, epsilon)

            del outputs, acts, loss, pv_adv, grid_adv



    with torch.no_grad():
        x_adv01_final = (x_orig01 + best_delta).clamp(0.0, 1.0)
        x_adv01_final = torch.max(
            torch.min(x_adv01_final, x_orig01 + epsilon),
            x_orig01 - epsilon,
        ).clamp(0.0, 1.0)

    return x_adv01_final, best_delta


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL original-image-space BSA attack"
    )

    parser.add_argument("--attck_type", type=str, default="bsa")
    parser.add_argument("--desired_norm_l_inf", type=float, default=0.005)
    parser.add_argument("--learningRate", type=float, default=0.001)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--numSteps", type=int, default=None)
    parser.add_argument("--attackSample", type=str, default="nature")

    args = parser.parse_args()

    attck_type = args.attck_type
    epsilon = float(args.desired_norm_l_inf)
    lr = float(args.learningRate)
    num_steps = int(args.numSteps) if args.numSteps is not None else int(args.num_steps)
    attackSample = str(args.attackSample)

    MODEL_PATH = "../illcond/QwenAttack/Qwen2.5-VL-7B-Instruct"
    IMAGE_PATH = f"llava_attack/dataSamplesForQuant/{attackSample}.JPEG"

    QUESTION = "What is shown in this image?"
    MAX_NEW_TOKENS = 128

    os.makedirs("qwen/outputsStorageImagenet", exist_ok=True)
    os.makedirs(f"qwen/outputsStorageImagenet/advOutputs/{attackSample}", exist_ok=True)
    os.makedirs(f"qwen/outputsStorageImagenet/convergence/{attackSample}", exist_ok=True)

    conv_path = (
        f"qwen/outputsStorageImagenet/convergence/{attackSample}/"
        f"qwen_ORIG_attack_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.npy"
    )

    adv_img_path = (
        f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.png"
    )

    adv_noise_path = (
        f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.pt"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"device={device}, dtype={dtype}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=dtype,
        device_map=None,
    ).to(device)

    model.eval()
    model.config.use_cache = False

    pil = Image.open(IMAGE_PATH).convert("RGB")
    x_orig01 = pil_to_tensor01(pil).to(device)

    template_inputs = build_template_inputs(
        processor,
        QUESTION,
        pil,
        device,
    )

    pv_clean, grid_clean = qwen_preprocess_differentiable(x_orig01, processor)

    print("\n=== CLEAN OUTPUT ===")
    clean_text = run_generation_with_pixel_values(
        model,
        processor,
        template_inputs,
        pv_clean,
        grid_clean,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(clean_text)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\nRunning BSA attack...")
    x_adv01, best_pert = adam_attack_original_space(
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


if __name__ == "__main__":
    main()