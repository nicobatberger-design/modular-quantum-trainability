"""Compare the follow-up results with PREDICTIONS_2026-10-31.md."""
import json

import numpy as np


def load(name):
    with open(name, encoding="utf-8") as f:
        return json.load(f)


print("=== truncation_test (gradient NRMSE) ===")
for r in load("results_truncation.json"):
    row = {n: round(v["nrmse_gradient"][0], 4) for n, v in r["truncated"].items()}
    print(f"N={r['N']} L={r['L']} g={r['g']:.3f} var={r['var_gradient']:.5f} {row}")

print("\n=== A. Bulk observable, N=14, Z on qubit 6 ===")
for r in load("results_followup_A.json"):
    w = r["window"]
    a6, a10 = w["6"]["nrmse_gradient"], w["10"]["nrmse_gradient"]
    print(f"L={r['L']} n=6 {a6[0]:.3f} [{a6[1]:.3f}, {a6[2]:.3f}] predicted 0.10-0.35: {0.10 <= a6[0] <= 0.35}")
    print(f"L={r['L']} n=10 {a10[0]:.3f} [{a10[1]:.3f}, {a10[2]:.3f}] predicted <=0.10: {a10[0] <= 0.10}")

print("\n=== B. Scaling variable x = L g^2, N=10 ===")
b = load("results_followup_B.json")
Ls = (40, 80, 160)
xs = [0.25, 0.5, 0.75, 1.0, 1.44, 2.0, 3.0, 4.0, 6.0, 9.0]
get = {(r["L"], r["x"]): r for r in b}
ratio = {x: [get[(L, x)]["var_gradient"] / get[(L, 0.0)]["var_gradient"] for L in Ls] for x in xs}
sb = {x: [get[(L, x)]["entropy_boundary_cut"] for L in Ls] for x in xs}
sm = {x: [get[(L, x)]["entropy_midblock_cut"] for L in Ls] for x in xs}
print("x      var ratio (L=40,80,160)    spread | boundary entropy         spread | mid-block entropy")
for x in xs:
    sp1, sp2 = max(ratio[x]) / min(ratio[x]), max(sb[x]) / min(sb[x])
    print(f"{x:<6}{np.round(ratio[x], 3)} {sp1:5.2f} | {np.round(sb[x], 3)} {sp2:5.2f} | {np.round(sm[x], 3)}")
print("largest spread, var ratio, x<=4:", round(max(max(ratio[x]) / min(ratio[x]) for x in xs if x <= 4), 3), "predicted <=1.5")
print("largest spread, boundary entropy, x<=4:", round(max(max(sb[x]) / min(sb[x]) for x in xs if x <= 4), 3), "predicted <=1.25")
for cut in ("entropy_midblock_cut", "entropy_boundary_cut"):
    for L in Ls:
        smax = max(get[(L, x)][cut] for x in xs + ["monolith"])
        v0 = get[(L, 0.0)]["var_gradient"]
        score = [get[(L, x)][cut] / smax * get[(L, x)]["var_gradient"] / v0 for x in xs]
        i = int(np.argmax(score))
        lx, s = np.log(xs), np.array(score)
        if 0 < i < len(xs) - 1:
            a, bb, _ = np.polyfit(lx[i - 1:i + 2], s[i - 1:i + 2], 2)
            star = float(np.exp(-bb / (2 * a)))
        else:
            star = xs[i]
        print(f"{cut} L={L}: x* grid {xs[i]}, parabola {star:.2f}, predicted 0.75-2.5: {0.75 <= star <= 2.5}, c=sqrt(x*)={np.sqrt(star):.2f}")

print("\n=== C. Trained parameters, 6 -> 12 qubits, L=40 ===")
c = load("results_followup_C.json")
gaps = [r["after"]["mean_abs_gap"] for r in c]
gaps0 = [r["before"]["mean_abs_gap"] for r in c]
full = [r["after"]["full_mean"] for r in c]
print(f"gap before training {np.mean(gaps0):.4f}, after {np.mean(gaps):.4f}, predicted <=0.10: {np.mean(gaps) <= 0.10}")
print(f"full <Z0> after training {np.mean(full):.3f} (worst {max(full):.3f}), predicted <=-0.7: {np.mean(full) <= -0.7}")

print("\n=== D. N=16 ===")
for r in load("results_followup_D.json"):
    t = r["truncated"]
    print(f"L={r['L']} K={r['K']}", {n: [round(x, 4) for x in v["nrmse_gradient"]] for n, v in t.items()},
          "n=6 <=0.05:", t["6"]["nrmse_gradient"][0] <= 0.05, "n=8 <=0.015:", t["8"]["nrmse_gradient"][0] <= 0.015)
