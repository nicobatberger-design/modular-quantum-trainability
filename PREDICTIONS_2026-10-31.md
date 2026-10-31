# Predictions for `truncation_followup.py`

Written and pushed before any of these four tests was run. The commit date on GitHub is the timestamp. Outcomes will be added in a later commit, whatever they are.

Reasoning behind them, stated as a heuristic and not as a result. At small g, each layer moves Pauli weight across an inter-block link with a probability of order g². The number of crossings over L layers should then depend on x = L g² only. At x of order 1, a local observable should spread over a few blocks at most.

## A. Bulk observable (N=14, Z on qubit 6, L=40 and 160, g = 1.2/√L)

A middle qubit has neighbours on both sides, so truncation should be worse than at the edge.

- Window of 6 qubits (4 to 9): gradient NRMSE between 0.10 and 0.35.
- Window of 10 qubits (2 to 11): gradient NRMSE at most 0.10.

## B. Scaling variable (N=10, L=40, 80, 160, x from 0.25 to 9)

- For x ≤ 4, the gradient-variance ratio Var(g)/Var(0) differs across the three depths by a factor of at most 1.5.
- For x ≤ 4, the entropy at a block-boundary cut differs across the three depths by a factor of at most 1.25.
- The maximum of (entropy fraction) × (variance fraction) falls at x between 0.75 and 2.5 at all three depths.

## C. Trained parameters (train on 6 qubits, evaluate on N=12, L=40, g = 1.2/√L)

The other six qubits get random parameters, 20 draws per trained circuit, 8 training runs.

- After training, the mean absolute gap between truncated and full ⟨Z0⟩ is at most 0.10.
- After training, the full-circuit ⟨Z0⟩ averages −0.7 or lower.

## D. Third system size (N=16, L=40 and 80, g = 1.2/√L, Z on qubit 0)

- Truncation to 6 qubits: gradient NRMSE at most 0.05.
- Truncation to 8 qubits: gradient NRMSE at most 0.015.
