# Experiment protocol and sources

## Decide before fitting

The question is whether nonlinear regression and a calibrated interval improve a fictional promise-setting decision. The seed, 36-day mechanism, time windows, 11 feature names, XGBoost parameters, alpha 0.1 and late/slack weights 4/1 are fixed inputs. There is no parameter search, early stopping, refit using calibration labels, or test-selected seed. OLS and XGBoost see identical training rows and feature columns.

The data schema exposes outcomes and timing for audit, but the matrix constructor reads only its explicit feature allowlist. Route/courier IDs, event IDs, absolute time, shift and completion/label fields cannot enter the model through column selection. Split validation checks every partition pair for identity overlap, feature availability, chronological boundaries and label maturity.

## The interval calculation

Fit the point estimator on training data only and clip its prediction to nonnegative duration support. On the separate calibration period, compute absolute residuals. With `n` calibration observations and miscoverage `alpha`, choose `k = ceil((n+1)*(1-alpha))`; the radius is the kth smallest residual, without interpolation. Use `[max(0, prediction-radius), prediction+radius]` for a new dispatch. Clipping the lower endpoint preserves membership for durations whose support is nonnegative.

If `k > n`, a finite sample radius is unavailable at that nominal level. The implementation returns an unbounded upper endpoint, reports interval width as N/A, and marks upper-endpoint promise metrics unavailable. It does not clamp the rank to a flattering finite maximum. Empty slices use null rates/means rather than zero coverage or invented loss.

The familiar finite-sample marginal coverage result requires exchangeability of calibration and future examples after the training procedure is fixed. The paper presents an IID setting; exchangeability is the relevant rank-symmetry condition. Neither a chronological partition nor disjoint identities guarantees it. This lab deliberately introduces a change after calibration and reports the resulting failure. It makes no group-conditional coverage guarantee and implements no covariate-shift correction, rolling recalibration or adaptive conformal method.

## Promise and audit definitions

For a proposed promise `p` and realized duration `y`, late minutes are `max(y-p,0)` and slack is `max(p-y,0)`. A missed promise means `y > p`. Both averages include all deliveries, including zeros. Illustrative loss is `4*late + 1*slack`; it is not currency. The two fixed promise policies use the point prediction or its calibrated interval's upper endpoint.

The report separately measures MAE, RMSE, signed error, underestimation strictly greater than ten minutes, interval membership, each tail miss and mean interval width. Slice labels overlap, so their counts must not be summed into a larger test set. Period, rush/off-peak and service-zone partitions each reconcile independently. The loss ignores changed customer/dispatcher behavior and cannot identify causal ROI.

## Validate the method rather than a favorable score

Tests cover data availability, all identity partition pairs, excluded leakage columns, exact conformal rank/ties, tiny calibration samples, nonnegative support, empty groups, policy denominators, tail/membership reconciliation, actual XGBoost execution, source/dataset hashes, raw prediction reconstruction and report links outside the repository. No test requires XGBoost to beat the baseline or attain nominal coverage on this drifted stream.

The saved model/interval results are empirical for one constructed dataset and training configuration. Repeating many seeds after inspecting the test and publishing only the best would change the protocol. Future windows/seeds and any recalibration method should be declared before evaluation.

## Primary sources and conceptual credit

- [User-supplied LinkedIn post by Andres Vourakis](https://www.linkedin.com/posts/andresvourakis_xgboost-and-similar-tree-based-models-are-share-7523776256539029505-auas/): topic inspiration supplied by the user. Its full text was not accessible to the browsing tool; no quotation or unverified company-specific claim is drawn from it.
- [Uber Engineering, Productionizing Distributed XGBoost to Train Deep Tree Models with Large Data Sets at Uber, December 10, 2019](https://www.uber.com/us/en/blog/productionizing-distributed-xgboost/): historical evidence of tree-based ETA work. It motivates the engineering question, not this project's data, scale, design or performance claims. No distributed-training claim is made here.
- [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor) and [parameters](https://xgboost.readthedocs.io/en/stable/parameter.html): public APIs used for squared-error CPU histogram regression. Tested package version is pinned to 3.1.2; documentation may track a newer release.
- [XGBoost installation guide](https://xgboost.readthedocs.io/en/stable/install.html#minimal-installation-cpu-only) and [official xgboost-cpu 3.1.2 package](https://pypi.org/project/xgboost-cpu/3.1.2/): CPU-only Linux x86_64 installation path used by the workflow, avoiding unnecessary GPU dependencies.
- [Lei, G'Sell, Rinaldo, Tibshirani and Wasserman, Distribution-Free Predictive Inference for Regression](https://arxiv.org/abs/1604.04173), [author-hosted paper](https://stat.cmu.edu/~ryantibs/papers/conformal-jasa.pdf): split-conformal residual calibration, separation of fitting/ranking, and its assumptions. The implementation is original and does not copy their code or experimental numbers.

All project figures, synthetic records, fitted models, metrics and narrative are original. Source references establish method provenance; they do not imply affiliation with Uber, Instacart, DoorDash or any other company.
