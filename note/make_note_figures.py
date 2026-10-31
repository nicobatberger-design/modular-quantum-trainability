"""Figures of the research note, from the JSON results at the repository root.

Run from the repository root: python note/make_note_figures.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DEPTH_STYLE = {40: (BLUE, "o"), 80: (ORANGE, "s"), 160: (AQUA, "^")}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Cambria", "DejaVu Serif"], "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 1.4, "lines.markersize": 5, "savefig.dpi": 300,
})


def load(name):
    with open(name, encoding="utf-8") as f:
        return json.load(f)


def depth_data():
    d = load("results_depth_test.json")
    p160 = load("results_depth_160.json")
    h = load("hardening_results.json")
    per = {int(L): list(v["gstar_per_seed"]) for L, v in d["data"].items()}
    per[160] = list(p160["gstar_per_seed"])
    for k, v in h["H2_depth_refine"].items():
        per[int(k.split("_")[1])] = list(v["per_seed"])
    for k, v in h["H2_extra_seeds"].items():
        per[int(k.split("_")[1])] += list(v["per_seed_6_10"])
    return per


def figure_1():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)

    per = depth_data()
    Ls = np.array(sorted(per))
    mean = np.array([np.mean(per[L]) for L in Ls])
    std = np.array([np.std(per[L]) for L in Ls])
    deep = Ls >= 40
    slope, intercept = np.polyfit(np.log(Ls[deep]), np.log(mean[deep]), 1)
    a.errorbar(Ls[~deep], mean[~deep], yerr=std[~deep], fmt="o", color=MUTED, mfc="white", capsize=2, elinewidth=0.8,
               label="L < 40, not fitted")
    a.errorbar(Ls[deep], mean[deep], yerr=std[deep], fmt="o", color=BLUE, capsize=2, elinewidth=0.8,
               label="L = 40 to 160, fitted")
    xs = np.array([34, 190])
    a.plot(xs, np.exp(intercept) * xs**slope, color=BLUE, linewidth=1.1, label=f"fit: slope −{abs(slope):.2f}")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("depth L")
    a.set_ylabel("g*  (N = 10, mean and per-seed s.d.)")
    a.set_xticks([6, 10, 20, 40, 80, 160])
    a.set_xticklabels(["6", "10", "20", "40", "80", "160"])
    a.minorticks_off()
    a.legend(loc="lower left", fontsize=8)
    a.set_title("(a) Optimal coupling against depth", loc="left", fontsize=9, color=INK)

    res = load("results_followup_B.json")
    get = {(r["L"], r["x"]): r for r in res}
    x = [0.25, 0.5, 0.75, 1.0, 1.44, 2.0, 3.0, 4.0, 6.0, 9.0]
    for L, (color, marker) in DEPTH_STYLE.items():
        v0 = get[(L, 0.0)]["var_gradient"]
        smax = max(get[(L, k)]["entropy_boundary_cut"] for k in x + ["monolith"])
        var = [get[(L, k)]["var_gradient"] / v0 for k in x]
        ent = [get[(L, k)]["entropy_boundary_cut"] / smax for k in x]
        b.plot(x, var, color=color, marker=marker, label=f"L = {L}")
        b.plot(x, ent, color=color, marker=marker, linestyle="--", mfc="white")
    b.set_xscale("log")
    b.set_xticks([0.25, 0.5, 1, 2, 4, 9])
    b.set_xticklabels(["0.25", "0.5", "1", "2", "4", "9"])
    b.minorticks_off()
    b.set_ylim(0, 1)
    b.set_xlabel("x = L g²")
    b.set_ylabel("fraction  (N = 10)")
    b.text(0.27, 0.70, "gradient variance\n(solid)", color=INK, fontsize=8, va="top")
    b.text(0.27, 0.36, "entropy, boundary cut\n(dashed)", color=INK, fontsize=8, va="top")
    b.legend(loc="upper center", ncol=3, fontsize=8, columnspacing=1.0, handlelength=1.6)
    b.set_title("(b) Both quantities against x", loc="left", fontsize=9, color=INK)

    fig.savefig("note/fig1_depth_and_scaling.png")
    plt.close(fig)


def figure_2():
    fig, ax = plt.subplots(figsize=(7.2, 2.75), constrained_layout=True)
    res = load("results_truncation.json")
    d16 = load("results_followup_D.json")
    bulk = load("results_followup_A.json")

    def line(r, key="truncated"):
        ns = sorted(int(k) for k in r[key])
        return ns, [r[key][str(n)]["nrmse_gradient"][0] for n in ns]

    edge = [r for r in res if abs(r["g"] - 1.2 / np.sqrt(r["L"])) < 1e-9]
    markers = {(12, 40): "o", (12, 80): "s", (12, 160): "^", (14, 80): "D"}
    for r in edge:
        ns, ys = line(r)
        ax.plot(ns, ys, color=BLUE, marker=markers[(r["N"], r["L"])], label=f"g₀, edge qubit, N={r['N']}, L={r['L']}")
    for r, m in zip(d16, ("v", "P")):
        ns, ys = line(r)
        ax.plot(ns, ys, color=BLUE, marker=m, linestyle=":", label=f"g₀, edge qubit, N=16, L={r['L']}")
    for r, m in zip(bulk, ("o", "^")):
        ns, ys = line(r, "window")
        ax.plot(ns[1:], ys[1:], color=MAGENTA, marker=m, label=f"g₀, middle qubit, N=14, L={r['L']}")

    strong = [r for r in res if r["L"] == 80 and r["N"] == 12 and r["g"] > 0.2]
    for r, color, name in zip(strong, (ORANGE, AQUA), ("2g₀", "4g₀")):
        ns, ys = line(r)
        ax.plot(ns, ys, color=color, marker="o", label=f"{name}, edge qubit, N=12, L=80")
    mono = [r for r in res if r["g"] > 1.5][0]
    ns, ys = line(mono)
    ax.plot(ns, ys, color=YELLOW, marker="o", label="g = π/2, edge qubit, N=12, L=40")

    ax.axhline(1, color=MUTED, linewidth=0.9, linestyle="--")
    ax.text(11.9, 1.12, "no better than answering 0", color=MUTED, fontsize=8, ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(5e-5, 20)
    ax.set_xlim(1.7, 12.3)
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_xlabel("qubits kept in the truncated circuit, n")
    ax.set_ylabel("gradient NRMSE")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.savefig("note/fig2_truncation.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_1()
    figure_2()
    per = depth_data()
    Ls = np.array(sorted(per))
    print("depths", list(Ls))
