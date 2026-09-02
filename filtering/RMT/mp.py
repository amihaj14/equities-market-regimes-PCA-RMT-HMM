import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity

from dataloader.dataloader import load_returns, TICKERS, START_DATE, END_DATE, CACHE_PATH


def build_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def mp_pdf(lam: np.ndarray, q: float, sigma2: float) -> np.ndarray:
    lambda_minus = sigma2 * (1 - np.sqrt(q)) ** 2
    lambda_plus = sigma2 * (1 + np.sqrt(q)) ** 2
    inside = np.clip((lambda_plus - lam) * (lam - lambda_minus), 0, None)
    pdf = np.sqrt(inside) / (2 * np.pi * q * sigma2 * lam)
    return np.where((lam >= lambda_minus) & (lam <= lambda_plus), pdf, 0.0)


def fit_once(bulk: np.ndarray, q0: float) -> tuple[float, float, float]:
    grid = np.linspace(bulk.min() * 0.5, bulk.max() * 1.5, 200)
    kde = KernelDensity(kernel="gaussian", bandwidth="silverman").fit(bulk.reshape(-1, 1))
    empirical_density = np.exp(kde.score_samples(grid.reshape(-1, 1)))

    def objective(params):
        q, sigma2 = params
        theoretical_density = mp_pdf(grid, q, sigma2)
        return np.sum((empirical_density - theoretical_density) ** 2)

    result = minimize(
        objective,
        x0=[q0, 1.0],
        bounds=[(1e-3, 0.999), (1e-3, None)],
    )
    q, sigma2 = result.x
    lambdaP = sigma2 * (1 + np.sqrt(q)) ** 2
    return q, sigma2, lambdaP


def fit_marchenko_pastur(
    corr_mat: pd.DataFrame,
    n_obs: int | None = None,
    n_exclude: int = 1,
    max_iter: int = 10,
    min_bulk_size: int = 5,
):
    eigvals = np.linalg.eigvalsh(corr_mat.values)
    n = len(eigvals)
    q0 = n / n_obs if n_obs else 0.5

    n_exclude = max(n_exclude, 1)
    for _ in range(max_iter):
        n_bulk = n - n_exclude
        if n_bulk < min_bulk_size:
            break

        bulk = eigvals[:n_bulk]
        q, sigma2, lambdaP = fit_once(bulk, q0)

        n_above = int(np.sum(bulk > lambdaP))
        if n_above == 0:
            break
        n_exclude += n_above

    return q, sigma2, lambdaP


if __name__ == "__main__":
    from filtering.RMT.plot import plot_mp_fit

    returns, _ = load_returns(TICKERS, START_DATE, END_DATE, CACHE_PATH)
    corr_mat = build_correlation_matrix(returns)
    q, sigma2, lambdaP = fit_marchenko_pastur(corr_mat, n_obs=returns.shape[0])
    print(f"q = {q:.4f}, sigma2 = {sigma2:.4f}, lambdaP = {lambdaP:.4f}")
    eigvals = np.linalg.eigvalsh(corr_mat.values)
    n_signal = int(np.sum(eigvals > lambdaP))
    print(f"Eigenvalues flagged as signal (> lambdaP): {n_signal} of {len(eigvals)}")
    print("Top 5 eigenvalues:", eigvals[-5:])
    plot_mp_fit(eigvals, q, sigma2, lambdaP)
