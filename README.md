# Weak inter-module coupling and the entanglement/trainability tradeoff in modular quantum circuits

> Independent research project. Reproducible numerical evidence that, in **modular** variational quantum circuits and quantum reservoirs, a **weak continuous inter-module coupling** `g` optimizes the tradeoff between entanglement (a quantum resource) and trainability (gradient variance / barren plateaus).

## One-sentence claim

In a block-structured circuit (strong intra-block entanglement, tunable inter-block ZZ coupling of continuous angle `g`), there is a **weak-coupling window** that keeps most of the gradient variance while roughly doubling inter-block entanglement — and this window is **set by circuit depth, not size**: `g*(L) ≈ 1.2/√L` in the deep regime (measured at N=10 for L=40–160, with a **pre-registered prediction at L=160 confirmed**: predicted 0.098, measured 0.094 ± 0.005), consistent with a lightcone-percolation picture where integrated inter-block leakage `L·g²` drives the transition. The **relative advantage of the window over the monolith grows with system size** (×1.1 at N=4 → ×2.6 at N=12), suggesting it becomes decisive precisely in the classically-hard regime.

<p align="center">
  <img src="resultats_barren.png" width="46%" alt="Barren plateau: monolith vs modular">
  <img src="resultats_depth_test.png" width="46%" alt="Depth law g*(L) with confirmed pre-registered prediction">
</p>

## What is here

| File | What it does |
|---|---|
| `barren_plateau_v2.py` | Monolith vs modular: gradient-variance collapse (slope −0.67 monolith vs +0.007 modular), ratio ×948 at N=12 |
| `sweet_spot.py` | At fixed N=10: entanglement and trainability vs `g` — the tradeoff curves cross |
| `scaling_fenetre.py` + `scaling_extend.py` | `g*(N)` for N=4–14, 5 seeds; the optimum stays at weak coupling at every size (no drift to the monolith) |
| `depth_test.py` + `depth_test_160.py` | **Main result**: `g*(L)` at fixed N=10, L=6–160 — crossover ~1/L → `1.2/√L`; pre-registered L=160 prediction confirmed |
| `scaling_fenetre_gpu.py` | GPU/CPU port (cupy/numpy) to push N=16–28 on free Kaggle/Colab; Rényi-2 entanglement for speed |
| `performance_sweetspot.py` | **Honest negative result**: full variational training at N=6 shows NO performance optimum at weak coupling — expected, since barren plateaus only bite at N≳10–12 (see limitations) |
| `make_fig_barren.py`, `make_fig_depth.py` | Regenerate the figures from the JSON results |
| `results_*.json`, `resultats_*.png` | Raw numbers and figures |

## Reproduce

```bash
pip install numpy matplotlib
python scaling_fenetre.py        # ~8 min CPU, produces resultats_scaling_fenetre.png
```

GPU (Kaggle T4, free): set accelerator to GPU, `pip install cupy-cuda12x`, run `scaling_fenetre_gpu.py` (reaches N≈28 in complex128).

## Method (honest scope)

- State-vector simulation, blocks of `m=2` qubits, intra-block ZZ angle `α=π/2` (strong), inter-block ZZ angle `g` (tunable, chain topology), `L=2N` layers of RY+RZ.
- Trainability = variance of the parameter-shift gradient of `⟨Z₀⟩` over random parameters.
- Entanglement = half-cut von Neumann entropy (`scaling_fenetre.py`) or Rényi-2 purity (GPU version, for speed).
- Sweet-spot score = (entanglement fraction) × (gradient-variance fraction); `g*` = parabolic argmax.

## Honest limitations (read before citing)

1. The principle "locality mitigates barren plateaus" is **established** (Cerezo et al. 2021). The contribution here is the **continuous coupling sweep**, the **depth law `g*(L)`**, and the unified VQC+reservoir framing — not the qualitative idea.
2. The performance advantage is **structurally beyond classical simulation**: barren plateaus only bite the monolith at `N ≳ 10–12`, so the decisive test needs **real hardware**.
3. The depth law is established at **N=10, one ansatz** (ZZ chain, m=2). Whether the constant (≈1.2) depends on N, block size m, or topology is open; an earlier apparent `g*(N)` trend conflated size and depth (our protocol used L=2N) and a 2-seed "plateau" claim was **retracted** after a rigorous 5-seed run — documented here deliberately rather than hidden.
4. **Open question (the #1 referee objection):** does the weak-coupling window escape the Cerezo et al. 2025 result "provable absence of barren plateaus ⇒ classical simulability"? This is the central thing to resolve.

## Closest prior work (assumed, cited up front)

- Cerezo et al., *Cost-function-dependent barren plateaus*, 2021 — locality ↔ trainability.
- Patti et al., *Entanglement Devised Barren Plateau Mitigation*, PRR 2021.
- Park et al., *Multi-Chip Ensembles*, arXiv:2505.08782 (2025) — inter-chip coupling, but **binary** (no continuous `g`).
- Askari, Kora, Simon, arXiv:2511.04900 (2025) — memory capacity peaks at weak coupling in spin-network QRC.
- Cerezo et al., *Absence of BP ⇒ classical simulability?*, Nat. Commun. 2025.

## Status

Preliminary but stress-tested: adversarial review, prior-art search over 17 papers, pre-registered predictions — **one confirmed** (L=160: predicted 0.098, measured 0.094 ± 0.005) — and one earlier claim **retracted** when a rigorous re-run falsified it. **Not a publishable breakthrough as-is** — a precise, coherent hypothesis with reproducible POCs, offered to start a conversation with researchers.

## Author

Nicolas Berger — independent researcher (France). nicobatberger@gmail.com, or open an Issue.
