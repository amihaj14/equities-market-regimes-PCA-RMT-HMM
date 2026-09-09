import numpy as np
import pandas as pd
from scipy.stats import kruskal

from dataloader.dataloader import load_returns, load_vix, TICKERS, START_DATE, END_DATE, CACHE_PATH
from HMM.hmm import labeled_transmat
from HMM.plot import plot_regimes
from pipelines.denoised_pipeline import run_denoised_pipeline, run_denoised_pca
from pipelines.raw_pipeline import run_raw_pipeline, run_raw_pca


def median_run_length(regimes):
    run_id = (regimes != regimes.shift()).cumsum()
    return regimes.groupby(run_id).size().median()


def transitions_per_year(regimes):
    transitions = int((regimes.values[1:] != regimes.values[:-1]).sum())
    n_years = (regimes.index[-1] - regimes.index[0]).days / 365.25
    return transitions / n_years


def vix_regime_separation(vix_z, regimes):
    aligned = pd.concat([vix_z.rename("vix_z"), regimes.rename("regime")], axis=1).dropna()
    groups = [group["vix_z"].values for _, group in aligned.groupby("regime")]
    medians = aligned.groupby("regime")["vix_z"].median()
    stat, pvalue = kruskal(*groups)
    return medians, stat, pvalue


def loading_cosine_similarities(loadings_a, loadings_b):
    # abs() because PCA eigenvector signs are arbitrary; doesn't correct for
    # PC2/PC3 order swaps between the two halves, only sign flips.
    sims = {}
    for col in loadings_a.columns:
        va, vb = loadings_a[col].values, loadings_b[col].values
        sims[col] = abs(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
    return sims


def split_sample_stability(returns, pca_fn, k=3):
    midpoint = len(returns) // 2
    _, loadings_first = pca_fn(returns.iloc[:midpoint], k=k)
    _, loadings_second = pca_fn(returns.iloc[midpoint:], k=k)
    return loading_cosine_similarities(loadings_first, loadings_second)


def summarize(name, factors, model, regimes, vix_z):
    print(f"\n=== {name} ===")
    print("Factor correlation (should be ~diagonal):")
    print(factors.corr())
    print("\nRegime counts:")
    print(regimes.value_counts())
    print(f"\nMedian regime run length: {median_run_length(regimes):.1f} trading days")
    print(f"Regime transitions per year: {transitions_per_year(regimes):.2f}")
    print("\nTransition matrix (Bull/Neutral/Bear order):")
    print(labeled_transmat(model).round(3))

    medians, stat, pvalue = vix_regime_separation(vix_z, regimes)
    print("\nVIX z-score median by regime:")
    print(medians)
    print(f"Kruskal-Wallis across regimes: H = {stat:.2f}, p = {pvalue:.3e}")


if __name__ == "__main__":
    returns, prices = load_returns(TICKERS, START_DATE, END_DATE, CACHE_PATH)
    vix = load_vix(START_DATE, END_DATE)
    vix_z = (vix - vix.mean()) / vix.std()

    raw_factors, raw_loadings, raw_model, raw_regimes = run_raw_pipeline(returns)
    denoised_factors, denoised_loadings, denoised_model, denoised_regimes = run_denoised_pipeline(returns)

    summarize("Raw (PCA on unfiltered correlation matrix)", raw_factors, raw_model, raw_regimes, vix_z)
    summarize("Denoised (RMT-filtered correlation matrix, then PCA)", denoised_factors, denoised_model, denoised_regimes, vix_z)

    print("\n=== Split-sample loading stability (cosine similarity, first half vs. second half) ===")
    print("Raw:", split_sample_stability(returns, run_raw_pca))
    print("Denoised:", split_sample_stability(returns, run_denoised_pca))

    plot_regimes(prices, raw_regimes, save_path="pipelines/regimes_raw.png")
    plot_regimes(prices, denoised_regimes, save_path="pipelines/regimes_denoised.png")
