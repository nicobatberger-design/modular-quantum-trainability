# Corrections, 31 October 2026

A re-check of this repository found errors in earlier versions of the README and of the research note. An independent reading by a second AI model added items 2, 8 and 9. This file stays, so that anyone who read the earlier versions can see what changed.

| # | Earlier statement | Correction |
|---|---|---|
| 1 | The decisive question is "structurally beyond classical simulation" and needs real hardware. | Contradicted for the observable studied here. At g = 1.2/√L, a 6-qubit truncated circuit reproduces the gradient of ⟨Z0⟩ of the N=12 and N=14 circuits with an NRMSE of 2 to 3 %, for L=40 to 160 (`truncation_test.py`). This is a numerical test on a local observable and random parameters. It is not a theorem. |
| 2 | "A modular network escapes the barren plateau": ratio ×1070 ± 167 at N=12. | The comparison is between an all-to-all CZ monolith and 2-qubit blocks with no link between them (`entangle_phases` in `hardening_suite.py`). Such blocks avoid the plateau by construction, because Z0 only sees a 2-qubit circuit. The ratio says nothing about coupled modular circuits. |
| 3 | Exponent −0.54 ± 0.07 "over 8 depths". | It comes from 5 depths, L=40 to 160. Nine depths exist. A single power law over 8 of them gives −0.81 and over all 9 gives −0.80, so the data show a crossover. |
| 4 | L=160: g* = 0.096 ± 0.004. | ±0.004 is the standard error of 10 seeds, while other ± values here are per-seed standard deviations (0.013 at this depth). Seeds 1-5 and 6-10 used different g grids. |
| 5 | "11 tests, 9 confirmed". | 7 met their written prediction. Rényi bias: 16 % measured for a limit of 15 %. Ring vs chain: −35 % for a predicted −10 %. Uniform block-size law: refuted. α=0.3: unresolved. |
| 6 | "Pre-registered". | Predictions were written in the scripts before the runs, but scripts and results entered the public repository in the same commits. No outside timestamp proves the order. Read it as "written before the run". |
| 7 | Park et al., "Multi-chip ensembles of variational quantum circuits". Askari et al., "Memory capacity of quantum reservoirs at weak coupling". | The titles are "Addressing the Current Challenges of Quantum Machine Learning through Multi-Chip Ensembles" (arXiv:2505.08782) and "Spin-Network Quantum Reservoir Computing with Distributed Inputs: The Role of Entanglement" (arXiv:2511.04900). |
| 8 | Cerezo et al. 2025 presented as showing that absence of barren plateaus implies classical simulability. | Their results are conditional and argued by circuit class, sometimes with a preliminary quantum data-acquisition step. They are not a universal theorem. |
| 9 | "Tripling inter-block entanglement". | The half-cut at N=10 crosses a block, so its entropy is not purely inter-block and is already non-zero at g=0. |
| 10 | "The relative advantage of the window grows with system size". | Withdrawn as a claim of advantage. It is a ratio of a score chosen by the author, and no performance gain was measured. |
| 11 | "AI-assisted for implementation and analysis". | Incomplete. AI tools also produced the theory sketch and the text. The author does not work in physics research. |

The empirical depth scaling on L=40 to 160 is not refuted by these corrections.
