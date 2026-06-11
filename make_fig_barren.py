# -*- coding: utf-8 -*-
"""Regenere resultats_barren.png proprement (titre unique, accents, pas de texte superpose)
a partir de results_barren_v2.json. Pour le one-pager France Quantum."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "results_barren_v2.json")) as f:
    R = json.load(f)

Ns = R["Ns"]
mono = [R["monolithe"][str(N)] for N in Ns]
modu = [R["modulaire"][str(N)] for N in Ns]
ratio = modu[-1] / mono[-1]
sl_m = R["slopes"]["monolithe"]; sl_d = R["slopes"]["modulaire"]

fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.semilogy(Ns, mono, "o-", color="#d23b3b", lw=2.2, ms=8,
            label=f"Monolithe (tout intriqué) — pente {sl_m:+.2f}")
ax.semilogy(Ns, modu, "s-", color="#13b48a", lw=2.2, ms=8,
            label=f"Modulaire (blocs faiblement couplés) — pente {sl_d:+.2f}")
ax.set_xlabel("nombre de qubits N", fontsize=11)
ax.set_ylabel("variance du gradient (échelle log)", fontsize=11)
ax.set_title("La modularité préserve l'entraînabilité\nlà où le monolithe s'effondre (barren plateau)",
             fontsize=12.5, pad=10)
ax.set_xticks(Ns)
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95)
ax.annotate(f"×{ratio:.0f} à N=12",
            xy=(12, mono[-1]), xytext=(9.4, mono[-1]*4.5),
            color="#d23b3b", fontsize=10.5, ha="center",
            arrowprops=dict(arrowstyle="->", color="#d23b3b", lw=1.2))
fig.tight_layout()
fig.savefig(os.path.join(HERE, "resultats_barren.png"), dpi=140)
print(f"resultats_barren.png regeneree (ratio N=12 = x{ratio:.0f})")
