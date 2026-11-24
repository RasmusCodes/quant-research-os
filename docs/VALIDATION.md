# Statistical validation

Every experiment, whatever produced it, is validated by the same code. This document gives each
statistic, its formula and its source, then the scorecard and the verdict rule built from them. All
thresholds are defaults of `qros.validation.report.ValidationConfig` and can be changed.

Notation: `r_t` are simple per-period net returns, `T` the number of observations, `q` the periods per
year (inferred from the dates and snapped to 1/4/12/52/252/365), `SR` the **per-period** Sharpe ratio
`mean / sd` (sample sd, ddof 1), `g3` the skewness and `g4` the non-excess kurtosis (population
moments). Annualised Sharpe = `SR * sqrt(q)`.

## Windows

Each experiment records an in-sample (train) window, a test window and optionally a holdout. Metrics
are computed separately for `is`, `oos` (test), `holdout` and `full` (all non-holdout data). If only a
test window is given, everything before it is in-sample.

## Performance metrics (`validation.metrics.performance_summary`)

Total return and CAGR (compounded), annualised mean and volatility, Sharpe, Sortino (downside
deviation `sqrt(mean(min(r,0)^2))`), maximum drawdown (from the running peak of the equity curve,
starting at 1) and its longest duration, Calmar, hit rate (share of non-zero periods that are
positive), skewness, kurtosis, the iid t-statistic, the Newey-West t-statistic (Bartlett kernel, lag
`floor(4 (T/100)^(2/9))`, a common rule of thumb; Newey and West 1987) and its one-sided p-value
(normal approximation), best/worst period, 95% VaR and CVaR, average and annualised turnover and cost
drag, average exposures, gross Sharpe, and beta, alpha, correlation and information ratio against a
benchmark when one is given. The test suite checks the Newey-West statistic against statsmodels.

## Sharpe-ratio inference (`validation.sharpe`)

- **Standard error** (Lo 2002; Mertens 2002, non-normal returns):
  `Var[SR] = (1 - g3 SR + (g4 - 1)/4 SR^2) / (T - 1)`.
- **Probabilistic Sharpe ratio** (Bailey and López de Prado 2012):
  `PSR(SR*) = Phi((SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2))`, the probability that the
  true Sharpe ratio exceeds `SR*`.
- **Deflated Sharpe ratio** (Bailey and López de Prado 2014): `DSR = PSR(SR0)` with the hurdle
  `SR0 = sqrt(V) ((1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e)))`, the expected maximum Sharpe
  ratio of `N` unskilled trials whose estimated Sharpe ratios have cross-sectional variance `V`
  (`gamma` = Euler-Mascheroni constant). In qros, `V` is the sample variance of the per-period
  in-sample Sharpe ratios of the family's trials, and `N` is the effective number of trials below. The
  test suite reproduces the paper's worked example (SR0 = 0.1132 per period, DSR = 0.9004).
- **Effective number of trials**: correlated trials are fewer independent tests. qros reports
  `N_corr = rho + (1 - rho) N` (rho = average pairwise correlation of the trials' returns; 1 when all
  are identical, N when uncorrelated) and the participation ratio of the correlation matrix's
  eigenvalues `(sum l)^2 / sum l^2`. The DSR uses `N_corr` plus the externally declared trials.
- **Minimum track record length** (Bailey and López de Prado 2012):
  `MinTRL = 1 + (1 - g3 SR + (g4 - 1)/4 SR^2) (Phi^-1(1 - a) / (SR - SR*))^2`, reported in years for
  `a = 0.05`, `SR* = 0`.
- **Sharpe haircut** (after Harvey and Liu 2015): the in-sample Sharpe ratio's two-sided p-value from
  `t = SR sqrt(T)` is Holm-adjusted together with the other trials' p-values and converted back into a
  Sharpe ratio; the haircut is the share removed.

## Multiple testing (`validation.multiple_testing`)

Bonferroni, Sidak, Holm (step-down), Hochberg (step-up), Benjamini-Hochberg and Benjamini-Yekutieli,
checked against statsmodels. Missing p-values are ignored. Every trial in a family that was evaluated
on its test window counts as a test of that window, so an experiment's out-of-sample p-value
(`1 - PSR(0)` on its test window) is Holm-adjusted across the family. This is how qros handles
repeated testing on the same out-of-sample period; the sealed holdout handles the final test.

## Resampling (`validation.bootstrap`)

Stationary bootstrap (Politis and Romano 1994): blocks start at random positions with geometric
lengths of mean `T^(1/3)` (a rule of thumb; pass your own for persistent series). 1,000 resamples with
a fixed seed give percentile 95% intervals for the Sharpe ratio of each window and the bootstrap
probability that it is not positive. `paired_sharpe_difference` bootstraps the difference of two
strategies' Sharpe ratios on common dates (used by `qros exp compare`).

## Probability of backtest overfitting (`validation.pbo`)

Combinatorially symmetric cross-validation (Bailey, Borwein, López de Prado and Zhu 2017). The
family's non-holdout returns are aligned on common dates (a T x N matrix) and cut into S = 16
contiguous blocks. For each of the C(16, 8) = 12,870 choices of eight blocks as in-sample, the
configuration with the best in-sample Sharpe ratio is found and its out-of-sample rank w = rank/(N+1)
recorded as the logit `ln(w/(1-w))`. PBO is the share of logits at or below zero: how often the
in-sample winner lands at or below the out-of-sample median. Also reported: the median logit,
in-sample and out-of-sample Sharpe of the winner, the probability of an out-of-sample loss and the
slope of out-of-sample on in-sample performance. Needs at least 4 configurations and 100 common
observations; block sums are precomputed, so each split costs one matrix product.

## Out-of-sample schemes (`validation.splits`)

- **Walk-forward**: `walk_forward_splits` (fixed train/test sizes, rolling or expanding, optional gap)
  and `walk_forward_by_years` (test on calendar years, train on all or the last k prior years).
  `walk_forward_optimize` selects the best configuration on each train window and stitches the test
  windows into an out-of-sample record of the *selection procedure*, with the switch rate, the modal
  selection and its share (parameter stability). For causal strategies, evaluating each configuration
  once on the full sample and slicing by window equals re-running per fold.
- **Purged k-fold with embargo** (López de Prado 2018, ch. 7): each observation has a label interval
  `[t0, t1]`; training observations whose interval overlaps the test fold's span are purged, and the
  embargo share of observations right after the test fold is dropped from training.

## Sensitivity (`validation.sensitivity`)

- **Parameter plateau**: neighbours of a configuration differ in exactly one numeric parameter by one
  step on that parameter's grid of tried values. `plateau ratio = mean(neighbour IS Sharpe) / own IS
  Sharpe`. Also reported: rank in the family, family median, share of configurations with a positive
  Sharpe ratio, and two-parameter heatmaps.
- **Costs**: the Sharpe ratio of `gross - m * costs` for m = 0, 1, 2, 3, and the break-even multiplier.
- **Outliers**: the Sharpe ratio after trimming the best *and* worst 1% of periods (trimming only the
  best days would penalise every strategy).
- **Variants**: any alternative series a strategy returns (for example a one-day execution delay).

## Regimes (`validation.regimes`)

Labels use only past information. Volatility regimes compare 63-day realised volatility of the
benchmark with the terciles of its own history up to the previous day (at least a year of history);
trend regimes use the sign of the trailing 252-day benchmark return up to the previous day; calendar
years need no benchmark. For each regime: observations, annualised return, Sharpe ratio, t-statistic
and share of total P&L, plus the share of regimes with a positive Sharpe ratio.

## The scorecard

| Check | Passes when | Core |
|---|---|---|
| Hypothesis pre-registered | the linked hypothesis was registered before the run started (for imported runs: before the declared start time) | yes |
| Out-of-sample significance | Holm-adjusted test-window p <= 0.05 (needs at least a year of test data) | yes, unless the holdout was evaluated |
| Deflated Sharpe ratio | DSR >= 0.95 on the in-sample window | yes |
| PBO | family PBO <= 0.25 | yes (when computable) |
| Out-of-sample / in-sample Sharpe | ratio >= 0.5 (when the in-sample Sharpe is positive) | yes |
| Parameter plateau | ratio >= 0.7 (when neighbours were tried) | yes |
| Costs | Sharpe ratio with doubled costs > 0 | yes (when costs are known) |
| Regimes | positive in >= 2/3 of volatility regimes (or >= 1/2 of calendar years) | yes |
| Outliers | trimmed Sharpe ratio > 0 | yes |
| Sealed holdout test | `p * k <= 0.05` where p = 1 - PSR(0) on the holdout and k is this experiment's look number, and k <= the holdout's limit | yes (when evaluated) |
| Reproduced | the latest re-run matched exactly or numerically | yes (when re-run) |
| No critical hygiene warnings | none on the experiment, its family or (for amended versions) its hypothesis | yes |

When an experiment evaluates the sealed holdout, the holdout test is its confirmatory test and the
test-window significance becomes informational: every trial in the family has seen the test window,
the holdout has been seen by no one before (or by a counted number of earlier looks).

## The verdict

Evaluated in order:

1. **invalid**: a critical integrity finding: look-ahead (`PIT_VIOLATION`), undeclared or excessive
   holdout use, an amended hypothesis tested on data already seen, a failed reproduction, corrupted
   data, a survivorship-biased universe, or returns computed from unadjusted prices.
2. **insufficient**: no test window of at least a year and no evaluated holdout (or **no-edge** if the
   in-sample Sharpe ratio is also below 0.5).
3. **robust**: every applicable core check passes, including the DSR and the confirmatory test.
4. **fragile**: attractive in-sample (Sharpe >= 0.5 or PSR >= 0.95) but fails at least one core check.
5. **no-edge**: otherwise.

Experiments with scalar results only (no return series) get **insufficient** (or invalid on an
integrity failure): trial accounting, provenance and reproduction still apply.

## Limitations

- The DSR treats a family's trials as draws from one population; mixing unrelated strategies in a
  family distorts the variance `V`. Families should hold the configurations of one idea.
- `N_corr` is a heuristic; the eigenvalue estimate is reported beside it.
- PBO uses contiguous blocks, so strong non-stationarity across blocks affects it; it describes the
  selection procedure over a family, not one configuration.
- The bootstrap block length is a rule of thumb, and the normal approximations behind the p-values
  assume enough observations.
- The thresholds are conventions, not laws; they are recorded in every validation report.

## References

- Bailey, D. H., and M. López de Prado (2012). The Sharpe ratio efficient frontier. *Journal of Risk* 15(2).
- Bailey, D. H., and M. López de Prado (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management* 40(5).
- Bailey, D. H., J. M. Borwein, M. López de Prado and Q. J. Zhu (2017). The probability of backtest overfitting. *Journal of Computational Finance* 20(4).
- Benjamini, Y., and Y. Hochberg (1995). Controlling the false discovery rate. *JRSS B* 57(1).
- Benjamini, Y., and D. Yekutieli (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics* 29(4).
- Harvey, C. R., and Y. Liu (2015). Backtesting. *Journal of Portfolio Management* 42(1).
- Harvey, C. R., Y. Liu and H. Zhu (2016). ... and the cross-section of expected returns. *Review of Financial Studies* 29(1).
- Hochberg, Y. (1988). A sharper Bonferroni procedure for multiple tests of significance. *Biometrika* 75(4).
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6(2).
- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal* 58(4).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Mertens, E. (2002). Comments on variance of the IID estimator in Lo (2002). Working paper.
- Newey, W. K., and K. D. West (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica* 55(3).
- Politis, D. N., and J. P. Romano (1994). The stationary bootstrap. *JASA* 89(428).
