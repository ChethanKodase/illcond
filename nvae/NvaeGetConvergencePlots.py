


'''


conda deactivate
conda deactivate
conda deactivate
export CUDA_VISIBLE_DEVICES=4
cd NVAE/
source nvaeenv1/bin/activate
cd ..
cd illcond/
python nvae/NvaeGetConvergencePlots.py

'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# --- params / paths ---
attck_types = ["grill_wass_kf", "la_l2_kf"]
desired_norm_l_inf = 0.05

base_load_dir = "nvae/deviation_store"
save_dir = "nvae/NvaeConvergencePlots"
os.makedirs(save_dir, exist_ok=True)

target_length = 40  # desired final length

def load_and_pad(path, target_len):
    """Load a 1D numpy array from path. If shorter than target_len,
    repeat the last value until target_len. If longer, trim to target_len."""
    arr = np.load(path)
    if arr.ndim != 1:
        arr = arr.ravel()
    n = len(arr)
    if n == 0:
        raise ValueError(f"Loaded array from {path} is empty.")
    if n < target_len:
        last_val = arr[-1]
        pad_len = target_len - n
        arr = np.concatenate([arr, np.full(pad_len, last_val, dtype=arr.dtype)])
    elif n > target_len:
        arr = arr[:target_len]
    # if equal, nothing to do
    return arr

# --- load files (and pad/trim) ---
files = {}
for attack in attck_types:
    fname = f"NVAE_attack_type{attack}_norm_bound_{desired_norm_l_inf}_.npy"
    path = os.path.join(base_load_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file not found: {path}")
    arr = load_and_pad(path, target_length)
    files[attack] = arr
    print(f"Loaded {attack}: original length -> padded/trimmed to {len(arr)}")

# --- plotting (paper-ready, matching previous style) ---
plt.figure(figsize=(6, 5))

# GRILL-wass → teal
plt.plot(files["grill_wass_kf"], label="GRILL-wass", color="teal", linewidth=2.5)

# LA-L2 → blue
plt.plot(files["la_l2_kf"], label="LA-L2", color="blue", linewidth=2.5)

plt.xlabel("steps", fontsize=22)
plt.ylabel("L-2 distance", fontsize=22)

plt.grid(True)
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

plt.yticks(fontsize=22)
plt.xticks(fontsize=22, rotation=45)

plt.legend(fontsize=18)
plt.tight_layout()

save_path = os.path.join(save_dir, f"NVAE_norm_{desired_norm_l_inf}_convergence.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Plot saved to: {save_path}")