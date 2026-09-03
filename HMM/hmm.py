import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

def fit_hmm(factors, n_states, covariance_type="full", n_itr=1000, tol=1e-4, random_state=42):
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_itr,
        tol=tol,
        random_state=random_state,
    )

    model.fit(factors.values)
    return model

def compute_bic(model, factors, n_states):
    n_obs,  n_features = factors.shape
    log_likelihood = model.score(factors.values)

    n_transmat_params = n_states*(n_states-1)
    n_startprob_params = n_states - 1
    n_mean_params = n_states*n_features
    n_cov_params = n_states*n_features*(n_features+1)//2

    k = n_transmat_params + n_startprob_params + n_mean_params + n_cov_params
    return -2.0*log_likelihood+k * np.log(n_obs)

def select_n_states(factors, candidates=(2,3,4)):
    models = {}
    bic_scores = {}

    for n_states in candidates:
        model = fit_hmm(factors,n_states)
        models[n_states] = model
        bic_scores[n_states] = compute_bic(model, factors, n_states)

    best_n_states = min(bic_scores, key=bic_scores.get)
    return models[best_n_states], best_n_states, bic_scores

def decode_regimes(model, factors):
    state_sequence = model.predict(factors.values)
    n_states = model.n_components

    volatilities = np.array([np.trace(model.covars_[state])for state in range(n_states)])
    vol_rank = np.argsort(volatilities)
    state_to_rank = {state: rank for rank, state in enumerate(vol_rank)}

    if n_states == 3:
        rank_to_label = {0: "Bull", 1: "Neutral", 2: "Bear"}
    else:
        rank_to_label = {rank: f"State_{rank}_of_{n_states}" for rank in range(n_states)}

    labels = [rank_to_label[state_to_rank[state]]for state in state_sequence]
    return pd.Series(labels, index=factors.index, name="regime")



if __name__ == "__main__":
    from dataloader.dataloader import load_returns, TICKERS, START_DATE, END_DATE, CACHE_PATH
    from filtering.RMT.mp import build_correlation_matrix, fit_marchenko_pastur
    from filtering.RMT.denoise_corr import denoise_corr
    from filtering.PCA.pca import run_pca

    returns, _ = load_returns(TICKERS, START_DATE, END_DATE, CACHE_PATH)
    corr_mat = build_correlation_matrix(returns)
    _, _, lambdaP = fit_marchenko_pastur(corr_mat, n_obs=returns.shape[0])
    denoised_mat = denoise_corr(corr_mat, lambdaP)

    factors, loadings = run_pca(returns, denoised_mat)

    _, _, bic_scores = select_n_states(factors)
    print("BIC scores (diagnostic only, does not select the model):", bic_scores)

    model = fit_hmm(factors, n_states=3)
    regimes = decode_regimes(model, factors)
    print(regimes.value_counts())

    assert np.allclose(model.transmat_.sum(axis=1), 1.0)

    log_likelihoods = np.array(model.monitor_.history)
    assert np.all(np.diff(log_likelihoods) >= -1e-6)
    print("Validation checks passed.")