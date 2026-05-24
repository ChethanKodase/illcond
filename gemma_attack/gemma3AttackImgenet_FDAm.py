



'''



export CUDA_VISIBLE_DEVICES=4
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 50 300); do
    python gemma_attack/gemma3AttackImgenet_FDAm.py --attck_type fdam --desired_norm_l_inf 0.0009 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 50 300); do
    python gemma_attack/gemma3AttackImgenet_FDAm.py --attck_type fdam --desired_norm_l_inf 0.0008 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 50 300); do
    python gemma_attack/gemma3AttackImgenet_FDAm.py --attck_type fdam --desired_norm_l_inf 0.0007 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 50 300); do
    python gemma_attack/gemma3AttackImgenet_FDAm.py --attck_type fdam --desired_norm_l_inf 0.0006 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 50 300); do
    python gemma_attack/gemma3AttackImgenet_FDAm.py --attck_type fdam --desired_norm_l_inf 0.0005 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done


'''

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


def get_grill_wass(outputs, outputsN, startPos, endPos):
    loss = 0.0
    for h, hn in zip(outputs.hidden_states[startPos:endPos], outputsN.hidden_states[startPos:endPos]):
        loss = loss + wasserstein_distance(h, hn)
    return loss


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
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
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
            do_sample=False,
        )

    input_ids = inputs["input_ids"]
    gen_only = out_ids[:, input_ids.shape[1]:]
    return processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]


# ----------------------------
# FDA helpers
# ----------------------------
def _drop_cls_if_present(h: torch.Tensor) -> torch.Tensor:
    """
    Vision hidden states are usually [B, T, D] with CLS token at position 0.
    FDA should act on patch/image tokens, not the CLS token.
    """
    if h.dim() == 3 and h.shape[1] > 1:
        return h[:, 1:, :]
    return h


def build_fda_masks_from_clean(hidden_states_clean, layer_start=1):
    """
    Build FDA support / non-support masks from clean hidden states.

    FDA defines support using activations greater than a central tendency C.
    The paper found spatial mean across channels to work best.
    For ViT-style [B, T, D] tensors, we treat:
      - T = spatial locations (patch tokens)
      - D = channels
    and use token-wise mean across channels as C.
    """
    masks = []
    selected_indices = []

    for idx in range(layer_start, len(hidden_states_clean)):
        h = hidden_states_clean[idx].detach().float()
        h = _drop_cls_if_present(h)

        if h.numel() == 0:
            continue

        # C_i(h,w) analogue for transformer tokens:
        # mean across channels for each token/spatial location
        C = h.mean(dim=-1, keepdim=True)

        support = h > C
        nonsupport = h < C

        masks.append((support, nonsupport))
        selected_indices.append(idx)

    return masks, selected_indices


def fda_loss_from_adv(hidden_states_adv, masks, selected_indices, eps=1e-12, layer_weights=None):
    """
    FDA objective:
        L(li) = log ||non-support adv||_2 - log ||support adv||_2
        Objective = - sum_i L(li)

    Returns the scalar objective to MINIMIZE.
    """
    total = 0.0
    used = 0

    if layer_weights is None:
        layer_weights = [1.0] * len(selected_indices)

    for w, idx, (support, nonsupport) in zip(layer_weights, selected_indices, masks):
        h_adv = hidden_states_adv[idx].float()
        h_adv = _drop_cls_if_present(h_adv)

        support_vals = h_adv[support]
        nonsupport_vals = h_adv[nonsupport]

        support_norm = torch.norm(support_vals, p=2)
        nonsupport_norm = torch.norm(nonsupport_vals, p=2)

        layer_obj = torch.log(nonsupport_norm + eps) - torch.log(support_norm + eps)

        # Paper minimizes -sum_i L(li)
        total = total - (float(w) * layer_obj)
        used += 1

    return total / max(used, 1)


def build_default_layer_weights(num_selected_layers):
    """
    Slightly emphasize deeper layers, which often carry higher-level semantics.
    """
    if num_selected_layers <= 0:
        return []
    weights = np.linspace(1.0, 2.0, num_selected_layers, dtype=np.float32)
    return weights.tolist()


# ----------------------------
# ORIGINAL-SPACE Adam attack
# ----------------------------
def adam_attack_original_space(
    model,
    processor,
    template_inputs,
    x_orig01,
    attck_type: str,
    num_steps: int,
    lr: float,
    epsilon: float,
    device,
    save_conv_path: str
):
    """
    Optimize delta in ORIGINAL image pixel space (no squeeze):
        x_adv01 = clamp(x_orig01 + delta, 0, 1)
        ||delta||_inf <= epsilon

    FDA version:
      - uses vision tower hidden states only
      - builds support masks from clean hidden states once
      - optimizes FDA objective over selected layers
    """
    x_orig01 = x_orig01.detach().to(device=device, dtype=torch.float32)

    delta = 0.001 * torch.randn_like(x_orig01, device=device, dtype=torch.float32)
    delta.requires_grad_(True)

    opt = torch.optim.Adam([delta], lr=lr)

    losses_list = [0.0]
    best_loss = 1e18
    best_delta = delta.detach().clone()

    model.eval()

    vision_model = model.vision_tower.vision_model
    vision_model.eval()

    if attck_type != "fdam":
        raise ValueError(f"This FDA script expects --attck_type fdam, got: {attck_type}")

    with torch.no_grad():
        pv_clean_fixed = gemma_preprocess_differentiable(x_orig01, processor)
        pv_clean_fixed = pv_clean_fixed.to(device=device, dtype=next(model.parameters()).dtype)

        vision_outN = vision_model(
            pixel_values=pv_clean_fixed,
            output_hidden_states=True,
            return_dict=True,
        )

        hiddStateLenVis = len(vision_outN.hidden_states)
        print(" Number of vision hidden states is: ", hiddStateLenVis)

        # skip earliest embedding-like state; use block outputs
        fda_masks, selected_indices = build_fda_masks_from_clean(
            vision_outN.hidden_states,
            layer_start=1
        )

        layer_weights = build_default_layer_weights(len(selected_indices))

        print(" Number of selected FDA layers is: ", len(selected_indices))
        if len(selected_indices) > 0:
            print(" Selected FDA layer indices: ", selected_indices)

    for step in range(num_steps):
        x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
        x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

        pv_adv = gemma_preprocess_differentiable(x_adv01, processor)
        pv_adv = pv_adv.to(device=device, dtype=next(model.parameters()).dtype)

        vision_out = vision_model(
            pixel_values=pv_adv,
            output_hidden_states=True,
            return_dict=True,
        )

        loss = fda_loss_from_adv(
            hidden_states_adv=vision_out.hidden_states,
            masks=fda_masks,
            selected_indices=selected_indices,
            layer_weights=layer_weights,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            delta.data.clamp_(-epsilon, epsilon)

        lv = float(loss.item())
        if (step + 1) % 10 == 0 or step == 0:
            print(f"[step {step+1}/{num_steps}] loss={lv:.6f}")

        if lv < best_loss:
            best_loss = lv
            best_delta = delta.detach().clone()
            losses_list.append(lv)
            np.save(save_conv_path, np.array(losses_list, dtype=np.float32))

        del vision_out, loss, pv_adv

    with torch.no_grad():
        x_adv01_final = (x_orig01 + best_delta).clamp(0.0, 1.0)
        x_adv01_final = torch.max(torch.min(x_adv01_final, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

    return x_adv01_final, best_delta


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

    MODEL_PATH = "../illcond/gemma_attack/Gemma3-4b"
    IMAGE_PATH = f"gemma_attack/dataSamplesForQuant/{attackSample}.JPEG"

    QUESTION = "What is shown in this image?"
    MAX_NEW_TOKENS = 128

    os.makedirs("gemma_attack/outputsStorageImagenet", exist_ok=True)
    os.makedirs(f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}", exist_ok=True)
    os.makedirs(f"gemma_attack/outputsStorageImagenet/convergence/{attackSample}", exist_ok=True)

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

    pil = Image.open(IMAGE_PATH).convert("RGB")
    x_orig01 = pil_to_tensor01(pil).to(device=device, dtype=torch.float32)

    template_inputs = build_template_inputs(processor, QUESTION, pil, device)

    pv_clean = gemma_preprocess_differentiable(x_orig01, processor).to(dtype=dtype)

    print("\n=== CLEAN OUTPUT ===")
    clean_text = run_generation_with_pixel_values(model, processor, template_inputs, pv_clean, max_new_tokens=MAX_NEW_TOKENS)
    print(clean_text)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    conv_path = f"gemma_attack/outputsStorageImagenet/convergence/{attackSample}/gemma_ORIG_attack_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.npy"
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
        save_conv_path=conv_path
    )

    adv_img_path = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.png"
    adv_noise_path = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.pt"

    tensor01_to_pil(x_adv01).save(adv_img_path)
    print(f"\nSaved ORIGINAL-resolution adversarial image to: {adv_img_path}")

    torch.save(best_pert.detach().cpu(), adv_noise_path)

    pv_adv = gemma_preprocess_differentiable(x_adv01, processor).to(dtype=dtype)
    print("\n=== ADVERSARIAL OUTPUT ===")
    adv_text = run_generation_with_pixel_values(model, processor, template_inputs, pv_adv, max_new_tokens=MAX_NEW_TOKENS)
    print(adv_text)

    cleanOutTxt = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/cleanOutput.txt"
    with open(cleanOutTxt, "w") as f:
        f.write(clean_text + "\n\n")

    advOutTxt = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.txt"
    with open(advOutTxt, "w") as f:
        f.write(adv_text + "\n")

    print(f"\nSaved outputs to: {advOutTxt}")
    print(f"Saved convergence to: {conv_path}")


if __name__ == "__main__":
    main()