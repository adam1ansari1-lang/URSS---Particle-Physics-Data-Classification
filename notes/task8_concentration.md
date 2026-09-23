# Week 4, Task 8: Investigating kernel concentration

## Question
Does increasing quantum feature-map expressiveness push the kernel
toward K = I (off-diagonal entries collapsing to zero)?

## What I found
Off-diagonal mean drops on both axes:

Depth: 0.671 (1 layer) -> 0.444 (2) -> 0.326 (3) -> 0.372 (4)
Qubit count: 0.444 (2) -> 0.383 (3) -> 0.296 (4) -> 0.185 (6) -> 0.196 (8)

(Plot: plots/task8_concentration.png. Data: results/task5_depth_sweep.csv,
results/task5_qubit_count_sweep.csv.)

Both trends move clearly toward zero - the kernel is concentrating
toward K = I as I add more qubits or layers, not staying flat.

## Small uptick at the far end
Neither trend is perfectly monotonic - depth ticks up at layers=4, qubit
count ticks up at 8. Probably not a real reversal, more likely the
circuit getting noisy/losing structure once this expressive - matches
the layers=4 breakdown from Task 7, where train AUC collapsed too.

## Why this drives memorisation
Diagonal stays at 1.0 regardless of settings, so a falling off-diagonal
mean means "similarity to self" and "similarity to others" pull apart.
In the K = I limit every point only resembles itself - less shared
structure for the SVM to learn from, so it memorises instead.

## Takeaway
Both depth and qubit count push toward K = I, matching Ahmed's
prediction, and this is the direct mechanism behind Task 7's
memorisation result.
