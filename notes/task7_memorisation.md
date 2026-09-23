# Week 4, Task 7: Investigating memorisation

## Question
Does the SVM start memorising training data instead of generalising as
the quantum feature map gets more expressive (more qubits, more layers)?

## Gap = train AUC minus test AUC (positive = overfitting)

### Qubit-count sweep (layers=2, chain_entangler fixed)
| n_qubits | test AUC | train AUC | gap |
|---|---|---|---|
| 2 | 0.834 | 0.821 | -0.013 |
| 3 | 0.776 | 0.874 | +0.098 |
| 4 | 0.824 | 0.936 | +0.112 |
| 6 | 0.678 | 0.973 | +0.295 |
| 8 | 0.691 | 0.987 | +0.296 |

### Depth sweep (n_qubits=2, chain_entangler fixed)
| layers | test AUC | train AUC | gap |
|---|---|---|---|
| 1 | 0.827 | 0.815 | -0.012 |
| 2 | 0.834 | 0.821 | -0.013 |
| 3 | 0.782 | 0.804 | +0.022 |
| 4 | 0.543 | 0.494 | -0.049 |

(Data: results/task5_qubit_count_sweep.csv, results/task5_depth_sweep.csv.
Plots: plots/task6_qubit_count_sweep.png, plots/task6_depth_sweep.png.)

## The qubit-count sweep shows clear memorisation
The gap grows steadily as qubit count goes up, from basically nothing at
2 qubits to +0.296 at 8. Train AUC climbs to ~0.99 while test AUC drops
from 0.834 to 0.691 over the same range - classic memorisation. I checked
this isn't just the task getting harder: the Bayes ceiling (Task 9) stays
flat-to-rising across this range, so the test AUC drop is a real property
of the model, not a moving target.

## Depth behaves differently - it breaks down rather than overfits
Up to layers=3 I see the same small growing gap as the qubit sweep. But
at layers=4, train AUC collapses too (0.804 -> 0.494) - both train and
test fall together. That's not memorisation (train would need to stay
high) - it looks more like the circuit has gotten complex enough that the
kernel loses coherent structure entirely, so the SVM can't even fit the
training set anymore. Worth treating as a separate failure mode, not just
"more overfitting."

## Why this happens: the off-diagonal collapse
Off-diagonal kernel mean shrinks as expressiveness goes up (0.444 ->
~0.19 across the qubit sweep, 0.671 -> 0.326 across layers 1-3). As
different training points stop looking similar to each other, each one
becomes its own island in kernel space - easy to fit perfectly (train AUC
up) but with nothing shared left to generalise to a new point (test AUC
down). At layers=4 this collapse has probably gone far enough that even
training points lose their mutual structure, which is why train AUC
drops too.

## Takeaway
More expressive circuits aren't better here - past a point they hurt
generalisation, matching what Ahmed's notes predicted. Qubit count shows
this as clean memorisation; depth shows a related but distinct
breakdown at layers=4.
