


'''

cd illcond
conda activate dt2
python diffae/DiffAEConvergencePlots.py



'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

attck_types = [
    "la_cos_kfAdamNoScheduler1",
    "grill_cos_kfAdamNoScheduler1"
]

legend_map = {
    "la_cos_kfAdamNoScheduler1": ("LA-cos", "red"),
    "grill_cos_kfAdamNoScheduler1": ("GRILL-cos", "gold"),
}

desired_norm_l_inf = 0.30

load_dir = "diffae/attack_run_time_univ/adv_div_convergence"
save_dir = "diffae/convergenceJan26"
os.makedirs(save_dir, exist_ok=True)

plt.figure(figsize=(6, 5))

for attack in attck_types:
    adv_div_list = np.load(
        f"{load_dir}/"
        f"DiffAE_attack_type{attack}_norm_bound_{desired_norm_l_inf}_.npy"
    )

    # 15 values uniformly distributed over 100 steps
    x_steps = np.linspace(0, 100, len(adv_div_list))

    label, color = legend_map[attack]
    plt.plot(
        x_steps,
        adv_div_list,
        label=label,
        color=color,
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

save_path = os.path.join(
    save_dir,
    f"DiffAE_norm_{desired_norm_l_inf}_convergence.png"
)
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Plot saved to: {save_path}")