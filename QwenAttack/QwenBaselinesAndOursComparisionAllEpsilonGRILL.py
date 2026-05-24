
'''


export CUDA_VISIBLE_DEVICES=3
conda deactivate
cd interpretAttacks/
conda activate gemma3
python qwen/QwenBaselinesAndOursComparisionAllEpsilonGRILL.py \
    --learningRate 0.001 \
    --num_steps 1000 \
    --AttackStartLayer 0 \
    --numLayerstAtAtime 1 \
    --whichMLP gate_proj \
    --whichMLPVis gate_proj \
    --chosenLanLayers 2 \
    --chosenVisLayers 0 1 2 4 5 6 7 8 9 14 24 \
    --numSamplesConsidered 38

'''

from bert_score import score
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FuncFormatter


# ============================================================
# IJCAI-style plotting settings
# ============================================================

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,

    "lines.linewidth": 2.8,
    "axes.linewidth": 1.0,
})


parser = argparse.ArgumentParser(
    description="Qwen BERTScore comparison across perturbation budgets"
)

parser.add_argument("--learningRate", type=float, default=1e-3)
parser.add_argument("--num_steps", type=int, default=2000)
parser.add_argument("--AttackStartLayer", type=int, default=0)
parser.add_argument("--AttackStartLayer_vis", type=int, default=0)
parser.add_argument("--numLayerstAtAtime", type=int, default=2)

parser.add_argument("--whichMLP", type=str, default="fc1")
parser.add_argument("--whichMLPVis", type=str, default="fc1")

parser.add_argument("--numSamplesConsidered", type=int, default=50)

parser.add_argument(
    "--chosenLanLayers",
    type=int,
    nargs="+",
    default=None,
    help="Example: --chosenLanLayers 0 1 2"
)

parser.add_argument(
    "--chosenVisLayers",
    type=int,
    nargs="+",
    default=None,
    help="Example: --chosenVisLayers 11 12 13"
)

args = parser.parse_args()

lr = float(args.learningRate)
num_steps = int(args.num_steps)
AttackStartLayer = int(args.AttackStartLayer)
AttackStartLayer_vis = int(args.AttackStartLayer_vis)
numLayerstAtAtime = int(args.numLayerstAtAtime)
whichMLP = str(args.whichMLP)
whichMLPVis = str(args.whichMLPVis)
numSamplesConsidered = int(args.numSamplesConsidered)

chosenLanLayers = args.chosenLanLayers
chosenVisLayers = args.chosenVisLayers

towardsNull = 0.5
ega_ratio = 0.2

#allEpsilons = [0.001, 0.002, 0.003, 0.004, 0.005]

allEpsilons = [0.001, 0.002, 0.004, 0.005]


all_attck_types = ["bsa", "dra", "fdam", "ssp", "ega", "grill_cos"]
AllAttckTypes = ["BSA", "DRA", "FDA", "SSPA", "EGA", "GRILL-cos"]


precisionMeanForAttacksSeries = []
precisionStdForAttacksSeries = []

recallMeanForAttacksSeries = []
recallStdForAttacksSeries = []

f1MeanForAttacksSeries = []
f1StdForAttacksSeries = []


for epsilon in allEpsilons:

    precisionMeanForAttacks = []
    precisionStdForAttacks = []

    recallMeanForAttacks = []
    recallStdForAttacks = []

    f1MeanForAttacks = []
    f1StdForAttacks = []

    print(f"\nEvaluating epsilon = {epsilon}")

    for attck_type in all_attck_types:

        sampleAggP = []
        sampleAggR = []
        sampleAggF1 = []

        print("attack type:", attck_type)

        for attackSample in range(1, numSamplesConsidered):

            if attck_type == "saa_loop":
                advOutputPath = (
                    f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
                    f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
                    f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
                    f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
                    f"{whichMLP}_{whichMLPVis}_{chosenLanLayers}_{chosenVisLayers}.txt"
                )

            elif attck_type == "ega":
                advOutputPath = (
                    f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
                    f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
                    f"num_steps_{num_steps}_ratio_{ega_ratio}.txt"
                )

            else:
                advOutputPath = (
                    f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
                    f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
                    f"num_steps_{num_steps}_.txt"
                )

            cleanOutputPath = (
                f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
                f"cleanOutput.txt"
            )

            if not os.path.exists(advOutputPath):
                print("Missing adversarial output:", advOutputPath)
                continue

            if not os.path.exists(cleanOutputPath):
                print("Missing clean output:", cleanOutputPath)
                continue

            with open(advOutputPath, "r") as f:
                advOutput = [f.read().strip()]

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

        sampleAggP = np.array(sampleAggP)
        sampleAggR = np.array(sampleAggR)
        sampleAggF1 = np.array(sampleAggF1)

        precisionMeanForAttacks.append(sampleAggP.mean())
        precisionStdForAttacks.append(sampleAggP.std())

        recallMeanForAttacks.append(sampleAggR.mean())
        recallStdForAttacks.append(sampleAggR.std())

        f1MeanForAttacks.append(sampleAggF1.mean())
        f1StdForAttacks.append(sampleAggF1.std())

    precisionMeanForAttacksSeries.append(precisionMeanForAttacks)
    precisionStdForAttacksSeries.append(precisionStdForAttacks)

    recallMeanForAttacksSeries.append(recallMeanForAttacks)
    recallStdForAttacksSeries.append(recallStdForAttacks)

    f1MeanForAttacksSeries.append(f1MeanForAttacks)
    f1StdForAttacksSeries.append(f1StdForAttacks)


precisionMeanForAttacksSeries = np.array(precisionMeanForAttacksSeries)
precisionStdForAttacksSeries = np.array(precisionStdForAttacksSeries)

recallMeanForAttacksSeries = np.array(recallMeanForAttacksSeries)
recallStdForAttacksSeries = np.array(recallStdForAttacksSeries)

f1MeanForAttacksSeries = np.array(f1MeanForAttacksSeries)
f1StdForAttacksSeries = np.array(f1StdForAttacksSeries)


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

def plot_metric(means, stds, ylabel, filename_prefix):

    save_dir = "qwen/AllPlots/comparisionSeriesGRILL"
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.8, 7.8))

    for i in range(means.shape[1]):

        mean = means[:, i]
        std = stds[:, i]

        ax.plot(
            allEpsilons,
            mean,
            label=AllAttckTypes[i],
            linewidth=3.2,
            marker="o",
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

    ax.tick_params(
        axis="x",
        rotation=35,
        labelsize=13,
        width=1.2,
        length=5
    )

    ax.tick_params(
        axis="y",
        labelsize=13,
        width=1.2,
        length=5
    )

    ax.grid(True, linewidth=0.7, alpha=0.35)

    legend = ax.legend(
        loc="upper center",
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

    fig.subplots_adjust(
        left=0.20,
        right=0.98,
        bottom=0.26,
        top=0.78
    )

    save_path = os.path.join(
        save_dir,
        f"{filename_prefix}_num_steps_{num_steps}_"
        f"AttackStartLayer_{AttackStartLayer}_"
        f"towardsNull_{towardsNull}_"
        f"numSamplesConsidered_{numSamplesConsidered}_"
        f"{whichMLP}_{whichMLPVis}_"
        f"{chosenLanLayers}_{chosenVisLayers}_"
        f"eps_0001_0002_0003.png"
    )

    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.05
    )

    plt.close()

    print(f"Saved: {save_path}")


plot_metric(
    precisionMeanForAttacksSeries,
    precisionStdForAttacksSeries,
    "BERT Precision",
    "PrecisionComparisionSeries"
)

print("\nprecision")
print("means:", precisionMeanForAttacksSeries)
print("stds:", precisionStdForAttacksSeries)


plot_metric(
    recallMeanForAttacksSeries,
    recallStdForAttacksSeries,
    "BERT Recall",
    "RecallComparisionSeries"
)

print("\nrecall")
print("means:", recallMeanForAttacksSeries)
print("stds:", recallStdForAttacksSeries)


plot_metric(
    f1MeanForAttacksSeries,
    f1StdForAttacksSeries,
    "BERT F1 Score",
    "F1ComparisionSeries"
)

print("\nf1 score")
print("means:", f1MeanForAttacksSeries)
print("stds:", f1StdForAttacksSeries)