import numpy as np
import pandas as pd



def denoise_corr(corr_mat: pd.DataFrame, lambdaP: float) -> pd.DataFrame:
    eigenvals, eigenvecs = np.linalg.eigh(corr_mat.values)

    noise_mask = eigenvals <= lambdaP
    avg_noise_bulk = eigenvals[noise_mask].mean()
    eigenvals_clipped = np.where(noise_mask, avg_noise_bulk, eigenvals)

    denoised = eigenvecs @ np.diag(eigenvals_clipped) @ eigenvecs.T

    d = np.sqrt(np.diag(denoised))
    denoised = denoised / np.outer(d, d)

    return pd.DataFrame(denoised, index=corr_mat.index, columns=corr_mat.columns)

if __name__ == "__main__":
    from dataloader.dataloader import load_returns, TICKERS, START_DATE, END_DATE, CACHE_PATH
    from filtering.RMT.mp import build_correlation_matrix, fit_marchenko_pastur
    returns, _ = load_returns(TICKERS, START_DATE, END_DATE, CACHE_PATH)
    corr_mat = build_correlation_matrix(returns)
    _, _, lambdaP = fit_marchenko_pastur(corr_mat, n_obs=returns.shape[0])
    denoised_mat = denoise_corr(corr_mat, lambdaP)
    print(denoised_mat)