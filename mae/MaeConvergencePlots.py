


'''

export CUDA_VISIBLE_DEVICES=2
cd mae/demo
conda activate mae5
python mae/MaeConvergencePlots.py

'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

attck_types = ["oa_l2_kf", "grill_l2_kf_only_decodings"]
desired_norm_l_inf = 0.07

# ---- quick sanity prints (kept from your script) ----
for attack in attck_types:
    print()
    print(attack)
    all_adv_div_list = np.load(
        f"mae/deviation_store/MAE_attack_type{attack}_norm_bound_{desired_norm_l_inf}_.npy"
    )
    print(all_adv_div_list[-10:-1])

# ---- keep your augmentation exactly (do not change logic) ----
data = {}
TARGET_LEN = 100

for attack in attck_types:
    file_path = f"mae/deviation_store/MAE_attack_type{attack}_norm_bound_{desired_norm_l_inf}_.npy"
    all_adv_div_list = np.load(file_path)[1:]  # keep your [1:] as-is

    cur_len = len(all_adv_div_list)
    if cur_len < TARGET_LEN:
        last_value = all_adv_div_list[-1]
        pad_count = TARGET_LEN - cur_len
        pad_values = np.full(pad_count, last_value)
        all_adv_div_list = np.concatenate([all_adv_div_list, pad_values])

    print("len(all_adv_div_list)", len(all_adv_div_list))

    # keep your normalization
    data[attack] = all_adv_div_list

# ---- plotting (paper-ready style, consistent with your other plots) ----
save_dir = "mae/MaeConvergencePlots"
os.makedirs(save_dir, exist_ok=True)

label_map = {
    "oa_l2_kf": "OA l2",
    "grill_l2_kf_only_decodings": "GRILL l2",
}
color_map = {
    "oa_l2_kf": "purple",
    "grill_l2_kf_only_decodings": "lime",
}

plt.figure(figsize=(6, 5))

for attack in attck_types:
    y_vals = data[attack]              # already augmented to TARGET_LEN
    x_steps = np.arange(len(y_vals))   # 0..99
    plt.plot(
        x_steps,
        y_vals,
        label=label_map[attack],
        color=color_map[attack],
        linewidth=2.5
    )

plt.xlabel("steps", fontsize=22)
plt.ylabel("L-2 distance", fontsize=22)

plt.grid(True)
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

plt.yticks(fontsize=22)
plt.xticks(fontsize=22, rotation=45)

plt.legend(fontsize=18)
plt.tight_layout()

save_path = os.path.join(save_dir, f"MAE_norm_{desired_norm_l_inf}_convergence.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Plot saved to: {save_path}")