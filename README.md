# Equities Market Regimes: PCA + RMT + HMM

Can you tell the difference between real market structure and pure statistical noise, and does removing the noise actually help you detect market regimes better? This
project is an empirical test of that question, not just a regime-detection tool.

## The idea

If you build a correlation matrix out of daily returns for a large basket of stocks,
most of what you get back is noise. With ~98 stocks and a few thousand days of
history, there just isn't enough data for every pairwise correlation to be
estimated precisely, a lot of the matrix is statistical static rather than real
co-movement. Random Matrix Theory (RMT) gives you a principled way to tell the two
apart: the Marchenko-Pastur distribution predicts exactly what the eigenvalues of a
*pure noise* correlation matrix should look like, so any eigenvalues that fall
outside that predicted noise band are probably picking up genuine structure,
sectors, macro factors, systematic risk, rather than randomness.

The hypothesis this repo tests: if you filter out that noise with RMT *before*
running PCA, and then feed the resulting factors into a Hidden Markov Model, do you
get more accurate and stable market-regime detection (bull / bear / neutral) than if
you just run PCA on the raw, unfiltered correlation matrix?

The actual deliverable isn't a regime detector, it's the head-to-head comparison
between the "raw" and "denoised" pipelines, run under identical conditions, to get an
honest answer to that question.

## How it works

```
Raw daily returns (98 stocks, 2010-2026)
      │
      ▼
Build correlation matrix
      │
      ▼
Fit Marchenko-Pastur distribution ─── (find the noise/signal boundary)
      │
      ▼
Denoise the correlation matrix ────── (clip noise eigenvalues, rebuild)
      │
      ▼
PCA (top 3 components)
      │
      ▼
Hidden Markov Model (3 states) ────── (Bull / Bear / Neutral)
      │
      ▼
Decoded regime timeline
```

The raw comparison branch skips the Marchenko-Pastur/denoising steps and runs PCA
straight on the unfiltered correlation matrix instead. Everything downstream (PCA,
HMM, validation) is identical between the two branches so the comparison is fair.

## Why these specific choices

- **98 individual stocks, not 5 sector ETFs.** RMT needs a reasonably large basket to
  have a meaningful noise bulk to filter out in the first place, with only 5 assets
  there's essentially nothing for it to do.
- **Both the noise-fit parameters are estimated from the data, not assumed.** Real
  returns aren't perfectly IID, so rather than fixing the theoretical noise ratio at
  its textbook value, both free parameters of the Marchenko-Pastur fit are optimized
  against the actual eigenvalue histogram, refit iteratively until it stabilizes.
- **The HMM sees all 3 PCA factors directly**, as a multivariate Gaussian, rather than
  being collapsed into a single composite signal, matching the approach used in the
  research literature this project is based on.
- **VIX is never fed into the model.** It's only used afterward, as a sanity check. 
  Do the periods the HMM labels as high-volatility/bear actually line up with spikes
  in the VIX?

## Data

Daily prices for 98 stocks across 5 sectors (tech, consumer discretionary,
industrials, financials, energy) from 2010 to 2026, pulled via `yfinance` and cached
locally. See `dataloader/dataloader.py` for the exact universe and loading logic.

## Status

This is a research project, actively in progress. Built so far:
- Data loading/caching (`dataloader/`)
- Marchenko-Pastur fitting and the noise/signal eigenvalue split (`filtering/RMT/`)
- Correlation matrix denoising (`filtering/RMT/denoise_corr.py`)
- PCA on the denoised matrix (`filtering/PCA/pca.py`)

Still to come: the HMM itself, the raw (non-denoised) comparison branch, a
rolling-window version of the full pipeline, and the VIX-based sanity check.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate      # or `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
python -m filtering.RMT.mp       # fit Marchenko-Pastur, plot eigenvalue spectrum
python -m filtering.PCA.pca      # run the full pipeline through PCA
```

Scripts are run as modules from the repo root (`python -m ...`), not as bare files —
that's what puts the repo root on the Python path so the local packages resolve.
