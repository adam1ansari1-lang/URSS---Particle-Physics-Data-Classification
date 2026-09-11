import numpy as np
from quantum_kernels import exact_kernel_overlap


def scale_to_range(X, feature_min, feature_max, target_min=0, target_max=np.pi):
    """
    Linearly rescale features from [feature_min, feature_max] (found on the
    TRAINING set) into [target_min, target_max], then CLIP so no value can
    ever fall outside that range -- this is what actually protects against
    aliasing for out-of-distribution test points.
    """
    X = np.array(X, dtype=float)
    scaled = (X - feature_min) / (feature_max - feature_min)
    scaled = scaled * (target_max - target_min) + target_min
    return np.clip(scaled, target_min, target_max)


# Everything below here only runs when THIS file is executed directly
# (python feature_scaling.py) -- NOT when another file does
# "from feature_scaling import scale_to_range". This stops the demo
# prints from flooding the output of every other script that imports
# this file.
if __name__ == "__main__":
    n_qubits = 2

    print("=== Part 1: Angle periodicity / aliasing ===\n")
    x_a = [0.5, 1.2]
    x_b = [0.5 + 2*np.pi, 1.2 + 2*np.pi]
    kernel_self_a = exact_kernel_overlap(x_a, x_a, n_qubits)
    kernel_aliased = exact_kernel_overlap(x_a, x_b, n_qubits)
    print(f"K(x_a, x_a) = {kernel_self_a:.6f}")
    print(f"K(x_a, x_b) = {kernel_aliased:.6f}  (should also be close to 1.0)")

    print("\n=== Part 2: Choosing a safe angle range ===\n")
    X_train = np.array([[0.2, 1.0], [1.5, 2.2], [0.8, 0.4], [2.0, 1.8]])
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)

    x_test_extreme = [5.0, -3.0]
    x_test_extreme_scaled = scale_to_range(x_test_extreme, train_min, train_max)
    print(f"Out-of-range test point {x_test_extreme} scales to "
          f"{np.round(x_test_extreme_scaled, 4)} -> clipped into [0, pi].")
