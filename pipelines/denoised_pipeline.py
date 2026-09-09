from filtering.RMT.mp import build_correlation_matrix, fit_marchenko_pastur
from filtering.RMT.denoise_corr import denoise_corr
from filtering.PCA.pca import run_pca
from HMM.hmm import fit_hmm, decode_regimes


def run_denoised_pca(returns, k=3):
    corr_mat = build_correlation_matrix(returns)
    _, _, lambdaP = fit_marchenko_pastur(corr_mat, n_obs=returns.shape[0])
    denoised_mat = denoise_corr(corr_mat, lambdaP)
    return run_pca(returns, denoised_mat, k=k)


def run_denoised_pipeline(returns, n_states=3, k=3):
    factors, loadings = run_denoised_pca(returns, k=k)
    model = fit_hmm(factors, n_states=n_states)
    regimes = decode_regimes(model, factors)

    return factors, loadings, model, regimes


if __name__ == "__main__":
    from dataloader.dataloader import load_returns, TICKERS, START_DATE, END_DATE, CACHE_PATH

    returns, _ = load_returns(TICKERS, START_DATE, END_DATE, CACHE_PATH)
    factors, loadings, model, regimes = run_denoised_pipeline(returns)
    print(regimes.value_counts())
