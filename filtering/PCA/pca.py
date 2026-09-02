import pandas as pd
import numpy as np


def run_pca(returns, denoised_corr, k = 3):
    eigenvals, eigenvecs = np.linalg.eigh(denoised_corr.values)

    order = np.argsort(eigenvals)[::-1][:k]
    topk_eigenvecs = eigenvecs[:, order]

    pc_cols = [f"PC{i+1}" for i in range(k)]
    loadings = pd.DataFrame(topk_eigenvecs, index=denoised_corr.index, columns=pc_cols)

    standardized_rets = (returns - returns.mean())/returns.std()
    standardized_rets = standardized_rets[denoised_corr.columns]

    factor_values = standardized_rets.values @ topk_eigenvecs
    factors = pd.DataFrame(factor_values, index=returns.index, columns=pc_cols)

    return factors, loadings


if __name__ == "__main__":
    from dataloader.dataloader import load_returns, TICKERS, START_DATE, END_DATE, CACHE_PATH
    from filtering.RMT.mp import build_correlation_matrix, fit_marchenko_pastur
    from filtering.RMT.denoise_corr import denoise_corr

    returns, _ = load_returns(TICKERS, START_DATE, END_DATE, CACHE_PATH)
    corr_mat = build_correlation_matrix(returns)
    _, _, lambdaP = fit_marchenko_pastur(corr_mat, n_obs=returns.shape[0])
    denoised_mat = denoise_corr(corr_mat, lambdaP)

    factors, loadings = run_pca(returns, denoised_mat)
    print(factors.head())
    print(factors.corr())