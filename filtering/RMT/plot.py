import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from filtering.RMT.rmt import mp_pdf


def plot_mp_fit(
    eigvals: np.ndarray,
    q: float,
    sigma2: float,
    lambdaP: float,
    save_path: str = "filtering/RMT/mp_fit.png",
) -> None:
    grid = np.linspace(max(eigvals.min(), 1e-6), eigvals.max() * 1.05, 500)
    theoretical_density = mp_pdf(grid, q, sigma2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(eigvals, bins=40, density=True, alpha=0.6, label="Empirical eigenvalues")
    ax.plot(grid, theoretical_density, color="red", label="Fitted Marchenko-Pastur pdf")
    ax.axvline(lambdaP, color="black", linestyle="--", label=f"lambdaP = {lambdaP:.3f}")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")
    ax.set_title(f"Eigenvalue spectrum vs. fitted MP distribution (q={q:.3f}, sigma2={sigma2:.3f})")
    ax.legend()

    fig.savefig(save_path)
    plt.close(fig)