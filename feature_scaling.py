#===========Task 8 ===========

n_qubits = 2

# ============ Part 1: Demonstrate the aliasing problem ============
print("=== Part 1: Angle periodicity / aliasing ===\n")

# Two events that differ by exactly 2*pi in each feature.
# Since RX(theta) and RX(theta + 2*pi) are the same rotation,
# these should produce IDENTICAL quantum states, even though
# the raw numbers are clearly very different.
x_a = [0.5, 1.2]
x_b = [0.5 + 2*np.pi, 1.2 + 2*np.pi]

kernel_self_a = exact_kernel_overlap(x_a, x_a, n_qubits)   # baseline: event vs itself
kernel_aliased = exact_kernel_overlap(x_a, x_b, n_qubits)  # event vs its "aliased twin"

print(f"x_a = {x_a}")
print(f"x_b = {[round(v, 4) for v in x_b]}  (x_a + 2*pi in each feature)")
print(f"K(x_a, x_a) = {kernel_self_a:.6f}   (event vs itself, should be 1.0)")
print(f"K(x_a, x_b) = {kernel_aliased:.6f}   (should ALSO be ~1.0, despite x_b looking very different!)")
print()

# Now show a genuinely different pair, for contrast
x_c = [2.0, 0.3]
kernel_different = exact_kernel_overlap(x_a, x_c, n_qubits)
print(f"x_c = {x_c}  (a genuinely different event)")
print(f"K(x_a, x_c) = {kernel_different:.6f}   (correctly much lower than 1.0)")
print()

print("Conclusion: x_a and x_b are numerically very different, but the kernel")
print("cannot tell them apart at all, because RX is 2*pi periodic. This is the")
print("aliasing failure mode: information is destroyed before the kernel sees it.\n")


# ============ Part 2: A sensible scaling range ============
print("=== Part 2: Choosing a safe angle range ===\n")

def scale_to_range(X, feature_min, feature_max, target_min=0, target_max=np.pi):
    """
    Linearly rescale features from [feature_min, feature_max] (found on
    the TRAINING set) into [target_min, target_max]. Using training-set
    min/max (not test-set) avoids the classic data-leakage mistake.
    """
    X = np.array(X, dtype=float)
    scaled = (X - feature_min) / (feature_max - feature_min)   # -> [0, 1]
    scaled = scaled * (target_max - target_min) + target_min   # -> [target_min, target_max]
    return scaled


# Toy "training set" to determine the scaling range
X_train = np.array([
    [0.2, 1.0],
    [1.5, 2.2],
    [0.8, 0.4],
    [2.0, 1.8],
])

train_min = X_train.min(axis=0)
train_max = X_train.max(axis=0)
print(f"Training set feature min: {train_min}, max: {train_max}")

X_train_scaled = scale_to_range(X_train, train_min, train_max, target_min=0, target_max=np.pi)
print("Training set scaled into [0, pi]:")
print(np.round(X_train_scaled, 4))
print()

# A well-behaved test point (within training range)
x_test_ok = [1.0, 1.5]
x_test_ok_scaled = scale_to_range(x_test_ok, train_min, train_max, target_min=0, target_max=np.pi)
print(f"In-range test point {x_test_ok} scales to {np.round(x_test_ok_scaled, 4)} -> inside [0, pi], fine.")

# A test point OUTSIDE the training range (Task 8's second question)
x_test_extreme = [5.0, -3.0]   # far beyond anything seen in training
x_test_extreme_scaled = scale_to_range(x_test_extreme, train_min, train_max, target_min=0, target_max=np.pi)
print(f"Out-of-range test point {x_test_extreme} scales to {np.round(x_test_extreme_scaled, 4)} "
      f"-> falls OUTSIDE [0, pi]!")
print()
print("Decision: clip scaled values to [0, pi] using np.clip before feeding them")
print("to the feature map, rather than letting them wrap around via periodicity.")
print("This matches the same principle used in Week 1/2: the scaler is fit on the")
print("training split only, and any transform derived from it is applied consistently")
print("to test data, including deciding how to handle out-of-distribution values.")