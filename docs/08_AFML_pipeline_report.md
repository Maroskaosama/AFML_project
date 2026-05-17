# AFML 10-Stock + 101 Formulaic Alphas: Pipeline Final Report

## 1. Universe & Data
- **Tickers**: AAPL, AMZN, NVDA, GOOGL, JNJ, JPM, MSFT, XOM, META, TSLA
- **Date range**: 2012-05-18 to 2025-04-30
- **Panel OHLCV**: 32560 rows x 6 cols, 0 NaN

## 2. Alpha Feature Engineering
- **Total alphas computed**: 101 (WorldQuant 101 Formulaic Alphas)
- **After exclusion** (NaN>40%, constant, inf): surviving set
- **After redundancy pruning** (|corr|>0.85): reduced set
- **Final alpha budget**: 33 alphas
- **Top alphas by MDI**: alpha041, alpha028, alpha012, alpha009

## 3. AFML Labels & Sample Weights
- **Total pooled events**: 881
- **Label distribution**: +1 = 478, -1 = 403
- **Sample weights**: uniqueness-scaled, range (0.0222, 4.5059)
- **Feature set**: 17 TS features + 33 alpha features = 50 total

## 4. Cross-Validation (MultiAssetPurgedKFold)
- **Scheme**: 5-fold time-block CV, embargo=1%, cross-sectional safe
- **Majority baseline**: 0.5426
- **CV accuracy (TS only, 17 feat)**: 0.5504 +/- 0.0491
- **CV accuracy (full 50 feat)**: 0.5438 +/- 0.0230
- **Best hyperparams**: max_depth=7, min_samples_leaf=30, max_features=sqrt
- **CV accuracy (tuned)**: 0.5551

## 5. Feature Importance
| Feature | MDI | MDA | SFI |
|---------|-----|-----|-----|
| alpha041 | 0.0646 | 0.0056 | 0.5563 |
| alpha012 | 0.0567 | -0.0132 | 0.5677 |
| alpha028 | 0.0565 | 0.0008 | 0.5962 |
| alpha009 | 0.0414 | -0.0042 | 0.5448 |
| ret_10d | 0.0353 | -0.0033 | 0.5066 |
| ret_20d | 0.0342 | -0.0025 | 0.5490 |
| amihud_illiquidity | 0.0337 | -0.0021 | 0.5286 |
| bekker_parkinson_vol | 0.0326 | -0.0013 | 0.5150 |
| alpha022 | 0.0300 | -0.0075 | 0.4826 |
| shannon_entropy | 0.0295 | -0.0066 | 0.5307 |
| ret_5d | 0.0293 | -0.0041 | 0.5104 |
| alpha011 | 0.0279 | -0.0070 | 0.5101 |
| roll_spread | 0.0279 | -0.0123 | 0.5319 |
| vol_20d | 0.0267 | -0.0060 | 0.5185 |
| vol_50d | 0.0243 | -0.0052 | 0.5224 |

## 6. Meta-Labeling & Bet Sizing
- **Primary model**: Random Forest on 17 TS features only (direction prediction)
- **Meta-label**: 1 if primary prediction correct, 0 if wrong
- **Secondary model**: Random Forest on all 50 features (predict meta-label probability)
- **Bet size**: primary_direction x meta_probability (signed position)
- **Primary CV accuracy**: 0.5540 +/- 0.0436
- **Meta CV accuracy**: 0.5357 +/- 0.0402
- **Avg |bet size|**: 0.5677

## 7. Backtesting Results (OOF test folds)

| Ticker | N | TotalRet | Sharpe | HitRate | MaxDD |
|--------|---|----------|--------|---------|-------|
| AAPL | 50 | 0.3686 | 1.6640 | 0.620 | -0.1431 |
| AMZN | 35 | -0.2236 | -1.1285 | 0.457 | -0.4570 |
| NVDA | 166 | 1.2262 | 2.5438 | 0.596 | -0.3933 |
| GOOGL | 74 | 0.2623 | 1.3933 | 0.554 | -0.1540 |
| JNJ | 81 | 0.0969 | 0.5504 | 0.593 | -0.1927 |
| JPM | 50 | 0.4031 | 1.4613 | 0.640 | -0.1648 |
| MSFT | 90 | 0.1867 | 0.7093 | 0.511 | -0.2498 |
| XOM | 137 | 0.3248 | 1.0399 | 0.555 | -0.2651 |
| META | 107 | -0.3871 | -1.0359 | 0.449 | -0.6455 |
| TSLA | 91 | 0.0374 | 0.0943 | 0.462 | -0.5342 |
| **PORTFOLIO** | 881 | 2.2953 | 2.3774 | 0.544 | -0.8131 |

## 8. Validation Audit
- **Total checks**: 32
- **PASS**: 31, **WARN**: 1, **FAIL**: 0

| Section | Check | Status | Detail |
|---------|-------|--------|--------|
| A: Source | fracdiff w_0==1.0 (last in oldest-first array) | PASS | w[-1]=1.000000 |
| A: Source | fracdiff weights monotone |w_k| from oldest to newest | PASS | min_mag=1.00e-05, max_mag=1.00e+00 |
| A: Source | fracdiff returns empty series when width > len(series) | PASS | len=0 |
| A: Source | MultiAssetPurgedKFold importable | PASS | n_splits=5, pct_embargo=0.01 |
| A: Source | rank_cs is cross-sectional (not time-series) | PASS | row0 ranks: A=0.50, B=1.00 |
| A: Source | adv(d) = sma(close*volume, d) (dollar volume) | PASS | adv3[A]=100.0 (expect 100), adv3[B]=200.0 (expect 200) |
| B: Data | Panel OHLCV has all 10 tickers | PASS | 10/10 tickers present |
| B: Data | Panel OHLCV zero NaN | PASS | NaN count: 0 |
| B: Data | Panel OHLCV has 6 columns | PASS | shape=(32560, 6) |
| B: Data | Pruned alpha panel has 33 features | PASS | shape=(32560, 33) |
| B: Data | No pruned alpha has >40% NaN | PASS | max NaN%=9.2% |
| B: Data | All labels have t1 > t0 | PASS | 0 violations |
| B: Data | All non-NaN bins are -1 or +1 (NaN=open events at data-end, expected) | PASS | 0 invalid bins |
| B: Data | Pooled dataset zero NaN (excl. ticker/t1) | PASS | NaN count: 0 |
| B: Data | Pooled label balance < 25% imbalance | PASS | +1=478, -1=403, imbalance=8.51% |
| B: Data | No t1 < t0 in pooled dataset | PASS | 0 rows with t1 < t0 |
| B: Data | cv_baseline_multistock.parquet exists | PASS |  |
| B: Data | meta_labeled_predictions.parquet exists | PASS |  |
| C: AFML | Sample weights sum > 0 | PASS | sum=1045.88, mean=1.1872 |
| C: AFML | Sample weights in (0, 1] | WARN | min=0.0222, max=4.5059 |
| C: AFML | fracdiff feature present for all tickers | PASS | 10/10 tickers have fracdiff |
| C: AFML | No train/test date overlap in any CV fold | PASS | 5-fold PurgedKFold checked |
| C: AFML | Purging removes all train samples whose t1 reaches test period | PASS | Checked all 5 folds |
| C: AFML | Baseline CV (TS-only) > majority class | PASS | CV_TS=0.5504 vs majority=0.5426 |
| C: AFML | Meta predictions have required columns | PASS | cols present: True |
| C: AFML | Meta probabilities in [0, 1] | PASS | min=0.4081, max=0.7179 |
| D: Alpha | alpha001 (NVDA) in [-0.5, 0.5] (rank - 0.5) | PASS | min=-0.400, max=0.500, n=3233 |
| D: Alpha | alpha012 vs manual formula corr > 0.99 | PASS | corr=1.0000, n=3255 |
| D: Alpha | alpha028 (NVDA) has values | PASS | n=3233, mean=-0.0024 |
| D: Alpha | alpha041 vs (sqrt(H*L) - close) corr > 0.85 | PASS | corr=0.9980, n=3256 |
| D: Alpha | alpha002 (NVDA) has values and is bounded | PASS | n=3249, |max|=0.9920 |
| D: Alpha | alpha056 is all-NaN (no market cap data, expected) | PASS | all NaN: True |

## 9. Artifacts
| File | Description |
|------|-------------|
| data/processed/panel_ohlcv.parquet | Raw 10-stock OHLCV panel |
| data/processed/panel_alpha_features.parquet | All 101 alpha features |
| data/processed/panel_alpha_features_pruned.parquet | 33 selected alphas |
| data/processed/pooled_modelling.parquet | 881 pooled events, 50 features |
| data/processed/cv_baseline_multistock.parquet | Baseline CV results |
| data/processed/meta_labeled_predictions.parquet | Meta-labeled OOF signals |
| data/processed/backtest_stats.parquet | Per-ticker and portfolio P&L stats |
| data/processed/mdi_importance.parquet | MDI feature importances |
| data/processed/mda_importance.parquet | MDA feature importances |
| data/processed/sfi_importance.parquet | SFI feature importances |
| reports/figures/P6_mdi_importance.png | MDI bar chart |
| reports/figures/P6_mda_importance.png | MDA bar chart |
| reports/figures/P6_backtest_results.png | Cumulative P&L + bet distribution |
| reports/figures/P6_importance_comparison.png | MDI/MDA/SFI comparison |

---
_Generated by AFML 10-Stock + 101 Alphas Pipeline_