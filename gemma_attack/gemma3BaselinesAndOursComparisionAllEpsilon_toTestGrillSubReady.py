
'''

export CUDA_VISIBLE_DEVICES=0
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3BaselinesAndOursComparisionAllEpsilon_toTestGrillSubReady.py --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --AttackStartLayer_vis 11 --whichMLP na --whichMLP_vis na --numSamplesConsidered 45



export CUDA_VISIBLE_DEVICES=0
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3BaselinesAndOursComparisionAllEpsilon_toTestGrillSubReady.py --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --AttackStartLayer_vis 11 --whichMLP na --whichMLP_vis na --numSamplesConsidered 300


'''

from bert_score import score
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FuncFormatter

# ============================================================
# IJCAI-style plotting settings (PNG only)
# ============================================================

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

    # Bigger readable fonts
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,

    # Thicker lines
    "lines.linewidth": 2.8,
    "axes.linewidth": 1.0,
})

# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Gemma-3 ORIGINAL-image-space adversarial attack"
)

parser.add_argument("--learningRate", type=float, default=1e-3)
parser.add_argument("--num_steps", type=int, default=2000)

parser.add_argument("--AttackStartLayer", type=int, default=0)
parser.add_argument("--AttackStartLayer_vis", type=int, default=0)

parser.add_argument("--numLayerstAtAtime", type=int, default=2)

parser.add_argument("--whichMLP", type=str, default="fc1")
parser.add_argument("--whichMLP_vis", type=str, default="fc1")

parser.add_argument("--numSamplesConsidered", type=int, default=50)

parser.add_argument("--perturbationScale", type=str, default="tinyTiny")

args = parser.parse_args()

# ============================================================
# Parameters
# ============================================================

lr = float(args.learningRate)
num_steps = int(args.num_steps)

AttackStartLayer = int(args.AttackStartLayer)
AttackStartLayer_vis = int(args.AttackStartLayer_vis)

numLayerstAtAtime = int(args.numLayerstAtAtime)

whichMLP = str(args.whichMLP)
whichMLP_vis = str(args.whichMLP_vis)

perturbationScale = str(args.perturbationScale)

numSamplesConsidered = int(args.numSamplesConsidered)

chosenLanLayers = [0]
chosenVisLayers = [11]

towardsNull = 0.1
ega_ratio = 0.2

# ============================================================
# Epsilon settings
# ============================================================

epsilonTypes = perturbationScale

if epsilonTypes == "tiny":
    allEpsilons = [0.001, 0.002, 0.003, 0.004, 0.005]
    all_attck_types = ["bsa", "dra", "fdam", "ssp", "ega", "grill_cos"]

elif epsilonTypes == "tinyTiny":
    allEpsilons = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
    all_attck_types = ["bsa", "dra", "fdam", "ssp", "ega", "grill_cos"]

else:
    allEpsilons = [0.01, 0.02, 0.03, 0.04, 0.05]
    all_attck_types = ["bsa", "dra", "fdam", "ssp", "ega", "grill_cos"]

# ============================================================
# Storage
# ============================================================

precisionMeanForAttacksSeries = []
precisionStdForAttacksSeries = []

recallMeanForAttacksSeries = []
recallStdForAttacksSeries = []

f1MeanForAttacksSeries = []
f1StdForAttacksSeries = []

# ============================================================
# Compute BERT scores
# ============================================================

for epsilon in allEpsilons:

    precisionMeanForAttacks = []
    precisionStdForAttacks = []

    recallMeanForAttacks = []
    recallStdForAttacks = []

    f1MeanForAttacks = []
    f1StdForAttacks = []

    for attck_type in all_attck_types:

        sampleAggP = []
        sampleAggR = []
        sampleAggF1 = []

        for attackSample in range(1, numSamplesConsidered):

            if attck_type == "saa":

                advOutputPath = (
                    f"gemma_attack/outputsStorageImagenet/advOutputs/"
                    f"{attackSample}/"
                    f"advOutput_attackType_{attck_type}_lr_{lr}_"
                    f"eps_{epsilon}_AttackStartLayer_{AttackStartLayer}_"
                    f"numLayerstAtAtime_{numLayerstAtAtime}_"
                    f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
                    f"{whichMLP}.txt"
                )

            elif attck_type == "saav":

                advOutputPath = (
                    f"gemma_attack/outputsStorageImagenet/advOutputs/"
                    f"{attackSample}/"
                    f"advOutput_attackType_{attck_type}_lr_{lr}_"
                    f"eps_{epsilon}_AttackStartLayer_{AttackStartLayer_vis}_"
                    f"numLayerstAtAtime_{numLayerstAtAtime}_"
                    f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
                    f"{whichMLP_vis}.txt"
                )

            elif attck_type == "ega":

                advOutputPath = (
                    f"gemma_attack/outputsStorageImagenet/advOutputs/"
                    f"{attackSample}/"
                    f"advOutput_attackType_{attck_type}_lr_{lr}_"
                    f"eps_{epsilon}_num_steps_{num_steps}_"
                    f"ratio_{ega_ratio}.txt"
                )

            else:

                advOutputPath = (
                    f"gemma_attack/outputsStorageImagenet/advOutputs/"
                    f"{attackSample}/"
                    f"advOutput_attackType_{attck_type}_lr_{lr}_"
                    f"eps_{epsilon}_num_steps_{num_steps}_.txt"
                )

            with open(advOutputPath, "r") as f:
                advOutput = [f.read().strip()]

            cleanOutputPath = (
                f"gemma_attack/outputsStorageImagenet/advOutputs/"
                f"{attackSample}/cleanOutput.txt"
            )

            with open(cleanOutputPath, "r") as f:
                cleanOutput = [f.read().strip()]

            P, R, F1 = score(
                advOutput,
                cleanOutput,
                lang="en",
                model_type="roberta-large",
                rescale_with_baseline=True
            )

            sampleAggP.append(P.item())
            sampleAggR.append(R.item())
            sampleAggF1.append(F1.item())

        # ====================================================
        # Aggregate
        # ====================================================

        sampleAggP = np.array(sampleAggP)
        sampleAggR = np.array(sampleAggR)
        sampleAggF1 = np.array(sampleAggF1)

        precisionMeanForAttacks.append(sampleAggP.mean())
        precisionStdForAttacks.append(sampleAggP.std())

        recallMeanForAttacks.append(sampleAggR.mean())
        recallStdForAttacks.append(sampleAggR.std())

        f1MeanForAttacks.append(sampleAggF1.mean())
        f1StdForAttacks.append(sampleAggF1.std())

    # ========================================================
    # Store series
    # ========================================================

    precisionMeanForAttacksSeries.append(
        np.array(precisionMeanForAttacks)
    )

    precisionStdForAttacksSeries.append(
        np.array(precisionStdForAttacks)
    )

    recallMeanForAttacksSeries.append(
        np.array(recallMeanForAttacks)
    )

    recallStdForAttacksSeries.append(
        np.array(recallStdForAttacks)
    )

    f1MeanForAttacksSeries.append(
        np.array(f1MeanForAttacks)
    )

    f1StdForAttacksSeries.append(
        np.array(f1StdForAttacks)
    )

# ============================================================
# Convert to numpy
# ============================================================

precisionMeanForAttacksSeries = np.array(
    precisionMeanForAttacksSeries
)

precisionStdForAttacksSeries = np.array(
    precisionStdForAttacksSeries
)

recallMeanForAttacksSeries = np.array(
    recallMeanForAttacksSeries
)

recallStdForAttacksSeries = np.array(
    recallStdForAttacksSeries
)

f1MeanForAttacksSeries = np.array(
    f1MeanForAttacksSeries
)

f1StdForAttacksSeries = np.array(
    f1StdForAttacksSeries
)

# ============================================================
# Labels
# ============================================================

AllAttckTypes = [
    "BSA",
    "DRA",
    "FDA",
    "SSPA",
    "EGA",
    "GRILL-cos"
]

# ============================================================
# Scientific notation formatter
# ============================================================

def format_func(x, pos):

    if x == 0:
        return "0"

    exponent = int(np.floor(np.log10(abs(x))))
    coeff = x / (10 ** exponent)

    return r"${:.1f}\times10^{{{}}}$".format(coeff, exponent)

# ============================================================
# Plot function
# ============================================================

def plot_metric(means, stds, ylabel, save_name):

    # Wider + taller: prevents vertical squeezing
    fig, ax = plt.subplots(figsize=(4.8, 7.8))

    for i in range(means.shape[1]):

        mean = means[:, i]
        std = stds[:, i]

        ax.plot(
            allEpsilons,
            mean,
            label=AllAttckTypes[i],
            linewidth=3.2,
            marker='o',
            markersize=5.5,
            markeredgewidth=1.1
        )

        ax.fill_between(
            allEpsilons,
            mean - std,
            mean + std,
            alpha=0.16,
            linewidth=0
        )

    ax.set_xlabel(r"$c$", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    ax.set_xticks(allEpsilons)
    ax.xaxis.set_major_formatter(FuncFormatter(format_func))

    ax.tick_params(axis='x', rotation=35, labelsize=13, width=1.2, length=5)
    ax.tick_params(axis='y', labelsize=13, width=1.2, length=5)

    ax.grid(True, linewidth=0.7, alpha=0.35)

    legend = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
        fontsize=14,
        handlelength=2.7,
        columnspacing=1.2,
        handletextpad=0.5
    )

    for legline in legend.get_lines():
        legline.set_linewidth(4.0)

    for spine in ax.spines.values():
        spine.set_linewidth(1.3)

    # Important: leave enough top space for legend
    fig.subplots_adjust(
        left=0.20,
        right=0.98,
        bottom=0.26,
        top=0.78
    )

    save_path = os.path.join(save_dir, save_name + ".png")

    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches='tight',
        pad_inches=0.05
    )

    plt.close()

# ============================================================
# Save directory
# ============================================================

save_dir = "gemma_attack/AllPlots/grillComparisionSeries"

os.makedirs(save_dir, exist_ok=True)

base_name = (
    f"num_steps_{num_steps}_"
    f"AttackStartLayer_{AttackStartLayer}_"
    f"towardsNull_{towardsNull}_"
    f"numSamplesConsidered_{numSamplesConsidered}_"
    f"{whichMLP}_{whichMLP_vis}_"
    f"{chosenLanLayers}_{chosenVisLayers}_"
    f"{epsilonTypes}"
)

# ============================================================
# Precision Plot
# ============================================================

plot_metric(
    precisionMeanForAttacksSeries,
    precisionStdForAttacksSeries,
    "BERT Precision",
    "PrecisionComparisionSeries_" + base_name
)

# ============================================================
# Recall Plot
# ============================================================

plot_metric(
    recallMeanForAttacksSeries,
    recallStdForAttacksSeries,
    "BERT Recall",
    "RecallComparisionSeries_" + base_name
)

# ============================================================
# F1 Plot
# ============================================================

plot_metric(
    f1MeanForAttacksSeries,
    f1StdForAttacksSeries,
    "BERT F1 Score",
    "F1ComparisionSeries_" + base_name
)

# ============================================================
# Print Results
# ============================================================

print("\nPrecision Means")
print(precisionMeanForAttacksSeries)

print("\nPrecision STDs")
print(precisionStdForAttacksSeries)

print("\nRecall Means")
print(recallMeanForAttacksSeries)

print("\nRecall STDs")
print(recallStdForAttacksSeries)

print("\nF1 Means")
print(f1MeanForAttacksSeries)

print("\nF1 STDs")
print(f1StdForAttacksSeries)