# Weak inter-block coupling in modular quantum circuits

Exact state-vector study of chains of two-qubit blocks with a continuous inter-block ZZ angle `g`. It finds a depth-set coupling scale, and tests whether that regime needs quantum hardware. For the observable studied here, it does not.

**Read first:** [the research note (PDF)](note/weak-coupling-truncation-note.pdf) and [CORRECTIONS.md](CORRECTIONS.md). Earlier versions of this repository made claims that were wrong, and that file lists them.

This is not peer reviewed. AI tools wrote the code and the text, and the author does not work in physics research. Expert criticism is the purpose of publishing it. Open an issue, or write to nicobatberger@gmail.com.

## Results

1. **Depth scale.** A balance score between half-cut entropy and gradient variance peaks at `g*`. For `L` from 40 to 160 at `N=10`, the slope of `g*(L)` is −0.54 ± 0.07 (5 depths). A single power law over all 9 depths fails (−0.80). At fixed `L=20`, `g*` does not fall with `N` from 6 to 16, and its prefactor depends on where the entropy cut falls.
2. **Scaling variable.** Gradient variance and entropy collapse approximately onto functions of `x = L g²` for `x ≤ 4`. A Pauli-weight heuristic suggests this, and it accounts for `g* ∝ 1/√L`. It is not a derivation.
3. **Classically cheap in the tested range.** At `g = 1.2/√L`, a 6-qubit truncated circuit reproduces the gradient of `⟨Z0⟩` in 12-, 14- and 16-qubit circuits with an NRMSE of 2 to 3 % over random circuits, for depths up to `L=160` (`L=40` and `80` at `N=16`). A middle qubit needs a wider window: 18 to 20 % with 6 qubits and 3 to 4 % with 10, at `N=14`. The gradient component studied keeps an appreciable variance, but this is not a demonstration of general trainability.

## Open question

Moving to twice and four times this coupling at `L=80`, the gradient variance falls by factors of 3.6 and 60, while the 8-qubit truncation error rises from 0.5 % to 11 % and 123 %. Does any coupling keep a usable gradient while defeating the best classical estimators? Failure of this crude truncation proves nothing about hardness, because methods such as Pauli propagation can do far better.

## Limits

One circuit family, `N ≤ 16`, local observables, one gradient component, mostly random parameters, no noise, no computational task. No performance gain is shown. The score is a convention and not the optimum of a task. Full variational training at `N=6` shows no performance optimum at weak coupling (`performance_sweetspot.py`).

## Predictions

| Predictions | Timestamp | Outcome |
|---|---|---|
| [PREDICTIONS_2026-10-31.md](PREDICTIONS_2026-10-31.md): middle qubit, scaling variable, trained parameters, `N=16` | Pushed before the runs, commit `c5ee4f1` | All held. See below. |
| First truncation test, docstring of `truncation_test.py` | Written before the run, no outside timestamp | `g*` prediction held. Control failed at `n=10`: 61 % measured, at least 90 % predicted. |
| Depth-law batteries of June 2026 (`hardening_suite.py`, `overnight_suite.py`) | Written before the runs, no outside timestamp | 7 of 11 met their written prediction. See CORRECTIONS.md. |

Outcomes of the pushed predictions, from `python analyze_followup.py`:

| Test | Predicted | Measured |
|---|---|---|
| Middle qubit, 6-qubit window | 0.10 to 0.35 | 0.202 (L=40), 0.181 (L=160) |
| Middle qubit, 10-qubit window | at most 0.10 | 0.031 (L=40), 0.042 (L=160) |
| Variance fraction spread across depths, `x ≤ 4` | at most 1.5 | 1.27 |
| Boundary entropy spread across depths, `x ≤ 4` | at most 1.25 | 1.13 |
| Score maximum | `x` from 0.75 to 2.5 | 1.46 to 1.67 (mid-block cut), 1.92 to 2.19 (boundary cut) |
| Trained parameters, mean gap | at most 0.10 | 0.0012 |
| Trained parameters, full-circuit `⟨Z0⟩` | −0.7 or lower | −0.994 |
| `N=16`, 6 qubits kept | at most 0.05 | 0.021 (L=40), 0.031 (L=80) |
| `N=16`, 8 qubits kept | at most 0.015 | 0.0037 (L=40), 0.0057 (L=80) |

## Reproduce

```bash
pip install numpy matplotlib
python truncation_test.py results_truncation.json
python truncation_followup.py A results_followup_A.json   # also B, C, D
python analyze_followup.py
python note/make_note_figures.py
```

The runs take from minutes to about an hour on a laptop, depending on the test.

| Files | What they do |
|---|---|
| `truncation_test.py`, `truncation_followup.py`, `analyze_followup.py` | Truncation tests of October 2026 and their analysis |
| `note/` | The note, its source and its figures |
| `depth_test.py`, `depth_test_160.py`, `hardening_suite.py`, `overnight_suite.py`, `m_seeds_extension.py` | Depth-scale measurements of June 2026 |
| `scaling_fenetre.py`, `scaling_extend.py`, `scaling_fenetre_gpu.py`, `sweet_spot.py` | Earlier size scans. Their protocol ties depth to size with `L=2N`, which conflates the two. |
| `barren_plateau_v2.py`, `performance_sweetspot.py` | Monolith vs uncoupled blocks, and a small training test |
| `results_*.json`, `*_results.json`, `resultats_*.png` | Raw numbers and older figures |

## Author

Nicolas Berger, no research affiliation. MIT licence.
