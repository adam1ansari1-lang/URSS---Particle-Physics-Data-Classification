"""
run_task9_bayes_ceiling.py
Task 9 (bonus): compute the Bayes-optimal AUC at each n_dim used in the
qubit-count sweep, to confirm the sweep isn't secretly changing task
difficulty instead of just the kernel's qubit count. Code from Ahmed's
Week 4 notes; separation_arg below reproduces the EXACT shell_radius
values run_task5_sweep.py actually generates at each n_dim, so this
measures the real ceiling for that dataset.
"""

import numpy as np
from scipy.stats import chi, norm
from sklearn.metrics import roc_auc_score


def bayes_auc(
    n_features: int = 2,
    separation: float = 1.35,
    shell_width: float = 0.15,
    bkg_scale: float = 1.2,
    n_mc: int = 200_000,
    seed: int = 0,
) -> float:
    """Best AUC any classifier could achieve on this toy configuration."""
    rng = np.random.default_rng(seed)
    typical_r = bkg_scale * np.sqrt(max(n_features - 1, 1))
    r_mean = separation * typical_r
    r_std = shell_width * r_mean

    r_b = chi.rvs(df=n_features, scale=bkg_scale, size=n_mc, random_state=seed)
    r_s = rng.normal(r_mean, r_std, size=n_mc)
    r = np.concatenate([r_b, r_s])
    y = np.concatenate([np.zeros(n_mc), np.ones(n_mc)])

    p_s = norm.pdf(r, r_mean, r_std)
    p_b = chi.pdf(r, df=n_features, scale=bkg_scale)
    return float(roc_auc_score(y, p_s / (p_b + 1e-300)))


# Must match run_task5_sweep.py's shell_radius_for exactly, or this checks
# the wrong dataset.
BG_SIGMA = 1.0
SEPARATION = 2.0 / (BG_SIGMA * np.sqrt(2))


def shell_radius_for(n_dim):
    return SEPARATION * BG_SIGMA * np.sqrt(n_dim)


if __name__ == "__main__":
    print(f"{'n_dim':6s} {'shell_radius':13s} {'bayes_ceiling':14s}")
    for n_dim in [2, 3, 4, 6, 8]:
        sr = shell_radius_for(n_dim)
        separation_arg = sr / (BG_SIGMA * np.sqrt(max(n_dim - 1, 1)))
        ceiling = bayes_auc(n_features=n_dim, separation=separation_arg,
                             shell_width=0.15, bkg_scale=BG_SIGMA)
        print(f"{n_dim:6d} {sr:13.3f} {ceiling:14.4f}")
