

'''


export CUDA_VISIBLE_DEVICES=0
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 50 250); do
    python gemma_attack/gemma3AttackImgenet_EGA1_flops.py --attck_type ega_flops --desired_norm_l_inf 0.05 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 50 250); do
    python gemma_attack/gemma3AttackImgenet_EGA1_flops.py --attck_type ega_flops --desired_norm_l_inf 0.04 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 50 250); do
    python gemma_attack/gemma3AttackImgenet_EGA1_flops.py --attck_type ega_flops --desired_norm_l_inf 0.03 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 50 250); do
    python gemma_attack/gemma3AttackImgenet_EGA1_flops.py --attck_type ega_flops --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 50 250); do
    python gemma_attack/gemma3AttackImgenet_EGA1_flops.py --attck_type ega_flops --desired_norm_l_inf 0.01 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done


'''



import os
import argparse
import random
import numpy as np

import torch
import torch.nn.functional as F

from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from torch.profiler import profile, ProfilerActivity


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


def pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def tensor01_to_pil(t01: torch.Tensor) -> Image.Image:
    if t01.dim() == 4:
        t01 = t01[0]
    t01 = t01.detach().cpu().clamp(0, 1)
    arr = (t01.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _get_target_hw(image_processor):
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
    ip = processor.image_processor
    th, tw = _get_target_hw(ip)
    x = resize_keep_aspect_center_crop(x01, th, tw)
    x = normalize_like_processor(x, ip)
    return x


def build_prompt_and_template(processor, question: str, pil_image: Image.Image, device):
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
    return prompt, template


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


def build_teacher_forced_inputs(processor, prompt: str, pil_image: Image.Image, answer_text: str, device):
    prompt_inputs = processor(text=[prompt], images=[pil_image], return_tensors="pt")
    full_text = prompt + answer_text
    full_inputs = processor(text=[full_text], images=[pil_image], return_tensors="pt")

    input_ids = full_inputs["input_ids"].to(device)
    attention_mask = full_inputs["attention_mask"].to(device)
    pixel_values = full_inputs["pixel_values"].to(device)

    labels = input_ids.clone()
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels.to(device),
    }
    return out, prompt_len


def token_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def select_top_entropy_positions(entropy: torch.Tensor, valid_mask: torch.Tensor, ratio: float = 0.2) -> torch.Tensor:
    selected = torch.zeros_like(valid_mask, dtype=torch.bool)
    B, _ = entropy.shape

    for b in range(B):
        idx = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        k = max(1, int(idx.numel() * ratio))
        vals = entropy[b, idx]
        topk = torch.topk(vals, k=k, largest=True).indices
        selected_idx = idx[topk]
        selected[b, selected_idx] = True

    return selected


def adam_attack_original_space_ega(
    model,
    processor,
    teacher_inputs,
    x_orig01,
    num_steps: int,
    lr: float,
    epsilon: float,
    device,
    save_conv_path: str,
    ega_ratio: float = 0.2,
    mask_refresh_every: int = 50,
):
    x_orig01 = x_orig01.detach().to(device)

    delta = 0.001 * torch.randn_like(x_orig01, device=device)
    delta.requires_grad_(True)

    opt = torch.optim.Adam([delta], lr=lr)

    losses_list = []
    best_metric = -1e18
    best_delta = delta.detach().clone()

    model.train()

    with torch.no_grad():
        pv_clean_fixed = gemma_preprocess_differentiable(x_orig01, processor)

        clean_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in teacher_inputs.items()}
        clean_inputs["pixel_values"] = pv_clean_fixed
        clean_inputs["use_cache"] = False

        outputs_clean = model(**clean_inputs, output_hidden_states=False, return_dict=True)
        clean_entropy = token_entropy_from_logits(outputs_clean.logits)
        valid_mask = (clean_inputs["labels"] != -100)
        ega_mask = select_top_entropy_positions(clean_entropy, valid_mask, ratio=ega_ratio)

        print("outputs_clean.logits.shape:", outputs_clean.logits.shape)
        print("num selected EGA positions:", int(ega_mask.sum().item()))

    adv_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in teacher_inputs.items()}
    adv_inputs["use_cache"] = False

    for step in range(num_steps):
        if step == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_flops=True,
                record_shapes=True
            ) as prof:
                x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
                x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

                pv_adv = gemma_preprocess_differentiable(x_adv01, processor)
                adv_inputs["pixel_values"] = pv_adv

                outputs_adv = model(**adv_inputs, output_hidden_states=False, return_dict=True)

                if mask_refresh_every > 0 and step > 0 and (step % mask_refresh_every == 0):
                    with torch.no_grad():
                        current_entropy = token_entropy_from_logits(outputs_adv.logits.detach())
                        ega_mask = select_top_entropy_positions(current_entropy, valid_mask, ratio=ega_ratio)

                adv_entropy = token_entropy_from_logits(outputs_adv.logits)
                loss_ega = adv_entropy[ega_mask].mean()

                # Optional: add a little NLL pressure for stronger output damage
                # loss_total = -(loss_ega + 0.5 * outputs_adv.loss)

                loss_total = -loss_ega

                opt.zero_grad(set_to_none=True)
                loss_total.backward()
                opt.step()

                with torch.no_grad():
                    delta.data.clamp_(-epsilon, epsilon)

                current_metric = float(loss_ega.item())
                losses_list.append(current_metric)

                if (step + 1) % 10 == 0 or step == 0:
                    print(
                        f"[step {step+1}/{num_steps}] "
                        f"ega_entropy={current_metric:.6f} "
                        f"n_selected={int(ega_mask.sum().item())}"
                    )

                if current_metric > best_metric:
                    best_metric = current_metric
                    best_delta = delta.detach().clone()
                    np.save(save_conv_path, np.array(losses_list, dtype=np.float32))

                del outputs_adv, adv_entropy, loss_ega, loss_total, pv_adv
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
            x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

            pv_adv = gemma_preprocess_differentiable(x_adv01, processor)
            adv_inputs["pixel_values"] = pv_adv

            outputs_adv = model(**adv_inputs, output_hidden_states=False, return_dict=True)

            if mask_refresh_every > 0 and step > 0 and (step % mask_refresh_every == 0):
                with torch.no_grad():
                    current_entropy = token_entropy_from_logits(outputs_adv.logits.detach())
                    ega_mask = select_top_entropy_positions(current_entropy, valid_mask, ratio=ega_ratio)

            adv_entropy = token_entropy_from_logits(outputs_adv.logits)
            loss_ega = adv_entropy[ega_mask].mean()

            # Optional: add a little NLL pressure for stronger output damage
            # loss_total = -(loss_ega + 0.5 * outputs_adv.loss)

            loss_total = -loss_ega

            opt.zero_grad(set_to_none=True)
            loss_total.backward()
            opt.step()

            with torch.no_grad():
                delta.data.clamp_(-epsilon, epsilon)

            current_metric = float(loss_ega.item())
            losses_list.append(current_metric)

            if (step + 1) % 10 == 0 or step == 0:
                print(
                    f"[step {step+1}/{num_steps}] "
                    f"ega_entropy={current_metric:.6f} "
                    f"n_selected={int(ega_mask.sum().item())}"
                )

            if current_metric > best_metric:
                best_metric = current_metric
                best_delta = delta.detach().clone()
                np.save(save_conv_path, np.array(losses_list, dtype=np.float32))

            del outputs_adv, adv_entropy, loss_ega, loss_total, pv_adv



    with torch.no_grad():
        x_adv01_final = (x_orig01 + best_delta).clamp(0.0, 1.0)
        x_adv01_final = torch.max(torch.min(x_adv01_final, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

    return x_adv01_final, best_delta


def main():
    parser = argparse.ArgumentParser(description="Gemma-3 ORIGINAL-image-space EGA attack")
    parser.add_argument("--attck_type", type=str, default="ega")
    parser.add_argument("--desired_norm_l_inf", type=float, default=0.05)
    parser.add_argument("--learningRate", type=float, default=1e-3)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--attackSample", type=str, default="1")
    parser.add_argument("--ega_ratio", type=float, default=0.2)
    parser.add_argument("--mask_refresh_every", type=int, default=50)
    args = parser.parse_args()

    attck_type = args.attck_type
    if attck_type != "ega_flops":
        raise ValueError("This script is only for --attck_type ega_flops")

    epsilon = float(args.desired_norm_l_inf)
    lr = float(args.learningRate)
    num_steps = int(args.num_steps)
    attackSample = str(args.attackSample)
    ega_ratio = float(args.ega_ratio)
    mask_refresh_every = int(args.mask_refresh_every)

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
    x_orig01 = pil_to_tensor01(pil).to(device)

    prompt, template_inputs = build_prompt_and_template(processor, QUESTION, pil, device)

    pv_clean = gemma_preprocess_differentiable(x_orig01, processor)

    print("\n=== CLEAN OUTPUT ===")
    clean_text = run_generation_with_pixel_values(
        model, processor, template_inputs, pv_clean, max_new_tokens=MAX_NEW_TOKENS
    )
    print(clean_text)

    teacher_inputs, prompt_len = build_teacher_forced_inputs(
        processor=processor,
        prompt=prompt,
        pil_image=pil,
        answer_text=clean_text,
        device=device,
    )
    print("prompt_len:", prompt_len)
    print("teacher_inputs[input_ids].shape:", teacher_inputs["input_ids"].shape)
    print("teacher_inputs[labels].shape:", teacher_inputs["labels"].shape)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    conv_path = (
        f"gemma_attack/outputsStorageImagenet/convergence/{attackSample}/"
        f"gemma_ORIG_attack_ega_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_ratio_{ega_ratio}.npy"
    )

    x_adv01, best_pert = adam_attack_original_space_ega(
        model=model,
        processor=processor,
        teacher_inputs=teacher_inputs,
        x_orig01=x_orig01,
        num_steps=num_steps,
        lr=lr,
        epsilon=epsilon,
        device=device,
        save_conv_path=conv_path,
        ega_ratio=ega_ratio,
        mask_refresh_every=mask_refresh_every,
    )

    adv_img_path = (
        f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_ratio_{ega_ratio}.png"
    )
    adv_noise_path = (
        f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_ratio_{ega_ratio}.pt"
    )

    tensor01_to_pil(x_adv01).save(adv_img_path)
    print(f"\nSaved ORIGINAL-resolution adversarial image to: {adv_img_path}")

    torch.save(best_pert.detach().cpu(), adv_noise_path)

    pv_adv = gemma_preprocess_differentiable(x_adv01, processor)
    print("\n=== ADVERSARIAL OUTPUT ===")
    adv_text = run_generation_with_pixel_values(
        model, processor, template_inputs, pv_adv, max_new_tokens=MAX_NEW_TOKENS
    )
    print(adv_text)

    cleanOutTxt = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/cleanOutput.txt"
    with open(cleanOutTxt, "w") as f:
        f.write(clean_text + "\n\n")

    advOutTxt = (
        f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
        f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_ratio_{ega_ratio}.txt"
    )
    with open(advOutTxt, "w") as f:
        f.write(adv_text + "\n")

    print(f"\nSaved outputs to: {advOutTxt}")
    print(f"Saved convergence to: {conv_path}")


if __name__ == "__main__":
    main()