"""Figures 1 and S1 for the manuscript (conceptual diagrams).

    python make_figs_concept.py "D:/claude_projects/R3/final-sub-25-30-aug-2026/figs"

Requires only matplotlib. Writes .tif (600 dpi), .png (600 dpi) and .pdf (vector).
"""
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else \
    r"D:\claude_projects\R3\final-sub-25-30-aug-2026\figs"

BLUE, GREEN, ORANGE, GREY = "#0072B2", "#009E73", "#E69F00", "#6E6E6E"
FS = 8.5
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FS,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext, kw in (("tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
                    ("png", {"dpi": 600}),
                    ("pdf", {})):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"),
                    bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="white", ec=GREY, lw=1.1, fs=FS, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3, linespacing=1.35, fontweight=weight)


def arrow(ax, p0, p1, color=GREY, style="-|>", lw=1.1, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11,
                                 color=color, linewidth=lw, linestyle=ls,
                                 shrinkA=2, shrinkB=2, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def blank(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# --------------------------------------------------------------------------
# Figure 1. Conceptual separation of the two constructs and the two criterion
# sources. This is the claim the study tests, drawn once.
# --------------------------------------------------------------------------
def figure1():
    fig, ax = blank(7.0, 3.5)

    box(ax, 0.02, 0.60, 0.20, 0.20, "Learner\nprompt", fc="#EAF3FA", ec=BLUE)
    box(ax, 0.02, 0.16, 0.20, 0.20, "Model\nresponse", fc="#F2F2F2")

    arrow(ax, (0.12, 0.60), (0.12, 0.37), color=GREY, ls="--")
    ax.text(0.135, 0.485, "generates", fontsize=FS - 1.2, color=GREY,
            ha="left", va="center", rotation=90)

    # measurement arms
    box(ax, 0.34, 0.60, 0.26, 0.20,
        "Prompt Quality Agent\n(response-independent)", fc="#EAF3FA", ec=BLUE)
    box(ax, 0.34, 0.30, 0.26, 0.20,
        "Fixed rubric\n(prompt-independent)", fc="#E8F5F0", ec=GREEN)
    box(ax, 0.34, 0.02, 0.26, 0.20,
        "Extracted criteria\n(prompt-derived)", fc="#FDF2E0", ec=ORANGE)

    arrow(ax, (0.22, 0.70), (0.34, 0.70), color=BLUE)
    arrow(ax, (0.22, 0.26), (0.34, 0.40), color=GREEN)
    arrow(ax, (0.22, 0.22), (0.34, 0.12), color=GREEN)
    arrow(ax, (0.12, 0.60), (0.34, 0.12), color=ORANGE, ls=":", rad=-0.28)
    ax.text(0.245, 0.055, "criteria taken\nfrom the prompt", fontsize=FS - 1.5,
            color=ORANGE, ha="center", va="center")

    # outcomes
    box(ax, 0.70, 0.60, 0.28, 0.20, "Prompt quality  $Q$", fc="white", ec=BLUE)
    box(ax, 0.70, 0.30, 0.28, 0.20, "$S_{\\mathrm{fixed}}$", fc="white", ec=GREEN)
    box(ax, 0.70, 0.02, 0.28, 0.20, "$S_{\\mathrm{cal}}$", fc="white", ec=ORANGE)

    for y, c in ((0.70, BLUE), (0.40, GREEN), (0.12, ORANGE)):
        arrow(ax, (0.60, y), (0.70, y), color=c)

    # the two tested associations
    arrow(ax, (0.84, 0.60), (0.84, 0.50), color=GREEN, style="<|-|>", lw=1.4)
    ax.text(0.865, 0.55, "RQ1", fontsize=FS, color=GREEN,
            ha="left", va="center", fontweight="bold")
    arrow(ax, (0.965, 0.60), (0.965, 0.22), color=ORANGE, style="<|-|>",
          lw=1.4, rad=-0.42)
    ax.text(0.995, 0.41, "RQ2", fontsize=FS, color=ORANGE,
            ha="left", va="center", fontweight="bold")

    ax.text(0.0, 0.93,
            "Response compliance is measured twice on the same response.\n"
            "Only the source of the scoring criteria differs.",
            fontsize=FS - 0.5, va="top", ha="left", color="black")
    save(fig, "Figure1_conceptual_separation")


# --------------------------------------------------------------------------
# Figure S1. Corpus construction and the analysis pipeline, with the counts
# actually obtained.
# --------------------------------------------------------------------------
def figureS1():
    fig, ax = blank(7.0, 5.6)

    box(ax, 0.03, 0.885, 0.27, 0.09, "WildChat-1M", fc="#F2F2F2")
    box(ax, 0.365, 0.885, 0.27, 0.09, "LMSYS-Chat-1M", fc="#F2F2F2")
    box(ax, 0.70, 0.885, 0.27, 0.09, "IFEval (541)", fc="#EAF3FA", ec=BLUE)

    box(ax, 0.03, 0.735, 0.60, 0.10,
        "First user turn  ->  length filter (< 1000 words)\n"
        "->  pedagogical-indicator filter  ->  de-identification")
    arrow(ax, (0.165, 0.885), (0.165, 0.835))
    arrow(ax, (0.50, 0.885), (0.50, 0.835))
    arrow(ax, (0.835, 0.885), (0.835, 0.665), color=BLUE)
    ax.text(0.85, 0.775, "constraints\nretained", fontsize=FS - 1.5,
            color=BLUE, ha="left", va="center")

    box(ax, 0.03, 0.585, 0.94, 0.08,
        "Master benchmark  n = 4,275   (WildChat 1,734  |  LMSYS 2,000  |  IFEval 541)")
    arrow(ax, (0.33, 0.735), (0.33, 0.665))

    box(ax, 0.20, 0.445, 0.60, 0.08,
        "Random sample under fixed seed  n = 500\n"
        "0 inference failures  ->  500 analysed", fc="#EAF3FA", ec=BLUE)
    arrow(ax, (0.50, 0.585), (0.50, 0.525))

    box(ax, 0.03, 0.295, 0.29, 0.09,
        "IFEval  n = 74\n73 verified, 116 instructions", fc="#E8F5F0", ec=GREEN)
    box(ax, 0.355, 0.295, 0.29, 0.09, "LMSYS  n = 235", fc="#F2F2F2")
    box(ax, 0.68, 0.295, 0.29, 0.09, "WildChat  n = 191", fc="#F2F2F2")
    for x in (0.175, 0.50, 0.825):
        arrow(ax, (0.50, 0.445), (x, 0.385), rad=0.0)

    box(ax, 0.03, 0.135, 0.21, 0.09,
        "Verified\naccuracy", fc="#E8F5F0", ec=GREEN)
    box(ax, 0.27, 0.135, 0.21, 0.09, "Divergence\n(RQ1, RQ2)", fc="#EAF3FA", ec=BLUE)
    box(ax, 0.51, 0.135, 0.21, 0.09, "Perturbation\n(RQ3)", fc="#FDF2E0", ec=ORANGE)
    box(ax, 0.75, 0.135, 0.22, 0.09, "Ablation\nn = 120", fc="#FDF2E0", ec=ORANGE)

    # Verified accuracy draws on the IFEval subset only; the other three
    # analyses draw on all 500 items, so they hang from a shared bus.
    arrow(ax, (0.175, 0.295), (0.135, 0.225), color=GREEN)
    ax.plot([0.375, 0.86], [0.262, 0.262], color=GREY, linewidth=1.0, zorder=1)
    for x, c in ((0.375, BLUE), (0.615, ORANGE), (0.86, ORANGE)):
        ax.plot([x, x], [0.262, 0.255], color=GREY, linewidth=1.0, zorder=1)
        arrow(ax, (x, 0.255), (x, 0.225), color=c)
    ax.plot([0.50, 0.50], [0.295, 0.262], color=GREY, linewidth=1.0, zorder=1)
    ax.text(0.885, 0.268, "all 500 items", fontsize=FS - 1.5, color=GREY,
            ha="left", va="bottom")

    box(ax, 0.15, 0.015, 0.70, 0.075,
        "Known-groups validation: 100 IFEval prompts paired with degraded variants",
        fc="#EAF3FA", ec=BLUE)
    arrow(ax, (0.135, 0.135), (0.30, 0.06), color=BLUE, ls="--", rad=0.25)

    save(fig, "FigureS1_pipeline")


if __name__ == "__main__":
    figure1()
    figureS1()
    print(f"Figures 1 and S1 written to {OUT}")
