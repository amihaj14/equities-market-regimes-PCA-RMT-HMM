import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt

REGIME_COLORS = {"Bull": "#0ca30c", "Bear": "#d03b3b"}


def plot_regimes(
    prices: pd.DataFrame,
    regimes: pd.Series,
    save_path: str = "HMM/regimes.png",
) -> None:
    aligned_prices = prices.loc[regimes.index]
    index_level = (aligned_prices / aligned_prices.iloc[0]).mean(axis=1) * 100

    fig, ax = plt.subplots(figsize=(18, 5))

    run_id = (regimes != regimes.shift()).cumsum()
    seen_labels = set()
    for _, run in regimes.groupby(run_id):
        regime = run.iloc[0]
        if regime not in REGIME_COLORS:
            continue
        label = regime if regime not in seen_labels else None
        seen_labels.add(regime)
        ax.axvspan(
            run.index[0], run.index[-1], color=REGIME_COLORS[regime], alpha=0.3,
            label=label,
        )

    ax.plot(index_level.index, index_level.values, color="#0b0b0b", linewidth=1.2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Equal-weighted index (start = 100)")
    ax.set_title("Decoded HMM regimes vs. equal-weighted price index (unshaded = Neutral)")
    ax.legend(loc="upper left")

    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from dataloader.dataloader import load_returns, TICKERS, START_DATE, END_DATE, CACHE_PATH
    from filtering.RMT.mp import build_correlation_matrix, fit_marchenko_pastur
    from filtering.RMT.denoise_corr import denoise_corr
    from filtering.PCA.pca import run_pca
    from HMM.hmm import fit_hmm, decode_regimes

    returns, prices = load_returns(TICKERS, START_DATE, END_DATE, CACHE_PATH)
    corr_mat = build_correlation_matrix(returns)
    _, _, lambdaP = fit_marchenko_pastur(corr_mat, n_obs=returns.shape[0])
    denoised_mat = denoise_corr(corr_mat, lambdaP)

    factors, loadings = run_pca(returns, denoised_mat)
    model = fit_hmm(factors, n_states=3)
    regimes = decode_regimes(model, factors)

    plot_regimes(prices, regimes)
