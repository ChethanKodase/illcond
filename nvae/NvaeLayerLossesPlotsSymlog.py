
'''

conda deactivate
conda deactivate
conda deactivate
export CUDA_VISIBLE_DEVICES=7
cd NVAE/
source nvaeenv1/bin/activate
cd ..
cd illcond/
python nvae/NvaeLayerLossesPlotsSymlog.py 


'''


import numpy as np
import matplotlib.pyplot as plt



import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- your existing settings ---
attck_type = "grill_wass_kf_layerLosses"
#attck_type = "grill_l2_kf_layerLosses"
#attck_type = "la_l2_kf_layerLosses"


desired_norm_l_inf = 0.05

allStepLayerLossesArray = np.load(
    "nvae/stepLayerLossstore//NVAE_attack_type"
    + str(attck_type)
    + "_norm_bound_"
    + str(desired_norm_l_inf)
    + "_.npy"
)

# --- plot settings ---
plt.figure(figsize=(13, 8))

num_steps = len(allStepLayerLossesArray)
#step_indices = list(range(0, num_steps, 30))  # plot every 20th step (as you do)

#step_indices = [0, 5, 10, 99]
step_indices = [99, 10, 5, 0]
print("step_indices", step_indices)

# Dark, clearly distinguishable colors (perceptually uniform)
cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0, 1, len(step_indices)))

print("colors", colors)
colors = ["red", "blue", "yellow", "green"]

# Plot curves
for idx, i in enumerate(step_indices):
    plt.plot(
        allStepLayerLossesArray[i],
        color=colors[idx],
        linewidth=2.5,
        alpha=0.9,
        label=f"Step {i}",
    )


# Symlog scale (handles zeros). Use a larger linthresh for your range (~0..17000)
plt.yscale("symlog", linthresh=10.0)
plt.ylim(bottom=0)


# Labels and ticks
plt.xlabel("Layer Index", fontsize=28)
plt.ylabel("Layer Loss (L2, symlog scale)", fontsize=28)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)



# Make y-axis labels look nice (avoid scientific clutter)
ax = plt.gca()
ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

# Grid
plt.grid(True, which="both", linestyle="--", alpha=0.3)

# Legend outside the plot (so it doesn't cover curves)
plt.legend(
    fontsize=16,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
    title="Attack Step",
    title_fontsize=18,
)

# Save
out_path = f"nvae/allLayerLossPlots/{attck_type}ALL_steps_symlog.png"
plt.tight_layout()
plt.savefig(out_path, bbox_inches="tight", dpi=300)
plt.close()

print("Saved:", out_path)

