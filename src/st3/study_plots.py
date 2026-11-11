"""Scientific figures for the local EXP-002 artifact directory."""

import textwrap

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

COLORS = {"A": "#657483", "B": "#bf6b15", "C": "#007d77"}


def render_spectrum(folder, spectrum):
    rank = spectrum["rank"]
    fig, (values, energy) = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    values.plot(rank, spectrum["singular_value"], "o-", markersize=3, color=COLORS["C"])
    values.set(xlabel="Component", ylabel="Singular value", title="Model C interaction spectrum")
    energy.plot(rank, spectrum["cumulative_energy"], color=COLORS["C"], label="Cumulative squared energy")
    energy.plot(rank, spectrum["relative_error"], color=COLORS["B"], label="Relative Frobenius error")
    energy.set(xlabel="Retained rank", ylabel="Fraction", ylim=(0, 1.03), title="Post-hoc rank truncation")
    energy.legend(loc="center right", fontsize=8)
    fig.savefig(folder / "singular_spectrum.png", dpi=180)
    plt.close(fig)


def render_study(folder, pairs, paintings, language, intervals):
    scored = paintings[paintings["status"] == "scored"]
    fig, (distribution, ci) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    distribution.hist(scored["contrast_C_minus_B"], bins=14, color=COLORS["C"], edgecolor="white")
    distribution.axvline(0, color="#333333", linestyle="--", linewidth=1)
    distribution.set(xlabel="Painting-level Contrast difference (C − B)", ylabel="Paintings",
                     title=f"Paired differences on {len(scored)} paintings")
    for y, name in enumerate(("contrast_C_minus_A", "contrast_C_minus_B")):
        result = intervals[name]
        point = result["difference"]
        low, high = result["ci95"]
        ci.hlines(y, low, high, color=COLORS["C"], linewidth=2)
        ci.scatter(point, y, color=COLORS["C"], zorder=3)
    ci.axvline(0, color="#333333", linestyle="--", linewidth=1)
    ci.set(yticks=[0, 1], yticklabels=["C − A", "C − B"], ylim=(-0.5, 1.5),
           xlabel="Mean Contrast difference", title="Paired bootstrap 95% percentile intervals")
    fig.savefig(folder / "contrast_reliability.png", dpi=180)
    plt.close(fig)

    ordered = language.sort_values("log_loss_C_minus_B")
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(7.5, 7.5), layout="constrained")
    for offset, name in zip((-.18, 0, .18), ("A", "B", "C")):
        ax.scatter(ordered[f"log_loss_{name}"], y + offset, color=COLORS[name], s=24, label=name)
    labels = [f"{row.language} ({row.pairs})" + ("" if row.macro_eligible else " *")
              for row in ordered.itertuples()]
    ax.set(yticks=y, yticklabels=labels, xlabel="Log Loss (bits per annotator)",
           title="Development Log Loss by language and model")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#dddddd", linewidth=.7)
    ax.legend(title="Model", loc="lower right")
    ax.text(0, -.085, "* Fewer than 20 pairs, excluded from macro Log Loss", transform=ax.transAxes, fontsize=9)
    fig.savefig(folder / "language_log_loss.png", dpi=180)
    plt.close(fig)

    delta = scored["contrast_C_minus_B"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True, layout="constrained")
    for ax, index in zip(axes, (delta.idxmax(), delta.idxmin())):
        row = scored.loc[index]
        observations = pairs[pairs["painting"] == row["painting"]].sort_values(["p", "language"])
        x = np.arange(len(observations))
        for offset, column, label, color in ((-.18, "p", "Observed", "#242424"), (0, "p_B", "B", COLORS["B"]),
                                              (.18, "p_C", "C", COLORS["C"])):
            ax.scatter(x + offset, observations[column], label=label, color=color, s=30)
        title = row["painting"].replace("_", " ").replace("-", " ")
        ax.set(title=f"{textwrap.fill(title, 42)}\nC − B Contrast {row['contrast_C_minus_B']:+.3f}",
               xticks=x, ylim=(-.05, 1.05), xlabel="Target language")
        ax.set_xticklabels(observations["language"], rotation=50, ha="right")
        ax.grid(axis="y", color="#dddddd", linewidth=.7)
    axes[0].set_ylabel("Positive sentiment share")
    axes[0].legend(loc="upper left", ncols=3)
    fig.savefig(folder / "painting_examples.png", dpi=180)
    plt.close(fig)
