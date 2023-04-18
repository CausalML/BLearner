# B-Learner: Quasi-Oracle Bounds on Heterogeneous Causal Effects Under Hidden Confounding

Metalearner for estimating lower and upper bounds of conditional average treatment effects (CATEs) in the presence of unobserved confounding. These bounds are sharp, valid and have quasi-oracle efficiency.

Replication code for [B-Learner: Quasi-Oracle Bounds on Heterogeneous Causal Effects Under Hidden Confounding](https://arxiv.org). 

## Requirements

* [pyreadr](https://pypi.org/project/pyreadr/)
* [econml](https://github.com/microsoft/EconML)
* [doubleml](https://github.com/DoubleML/doubleml-for-py)
* [sklearn-quantile](https://pypi.org/project/sklearn-quantile/)
* [pytorch](https://pytorch.org/)
* [ray](https://pypi.org/project/ray/)
* [pytorch-lightning](https://www.pytorchlightning.ai/)

## Example Usage

We train the `BLearner` on an observational dataset $Z=(X, A, Y)$ and predict the bounds on a test set $X_{test}$. 

```Python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn_quantile import RandomForestQuantileRegressor
from models.blearner.nuisance import RFKernel, KernelSuperquantileRegressor
from models.blearner import BLearner

gamma = np.e # corresponds to Lambda in the paper 
tau = gamma / (1+gamma)

# Propensity model
propensity_model = RandomForestClassifier()
# Outcome model
mu_model = RandomForestRegressor()
# Quantiles
quantile_model_upper = RandomForestQuantileRegressor(q=tau)
quantile_model_lower = RandomForestQuantileRegressor(q=1-tau)
# CVaR model
cvar_model_upper = KernelSuperquantileRegressor(
        kernel=RFKernel(clone(mu_model, safe=False)),
        tau=tau,
        tail="right")
cvar_model_lower = KernelSuperquantileRegressor(
        kernel=RFKernel(clone(mu_model, safe=False)),
        tau=1-tau,
        tail="left")
# Bounds model
cate_bounds_model = RandomForestRegressor()
# BLearner estimator
BLearner_est = BLearner(
    propensity_model = propensity_model, 
    quantile_plus_model = quantile_model_upper, 
    quantile_minus_model = quantile_model_lower,
    mu_model = mu_model, 
    cvar_plus_model = cvar_model_upper, 
    cvar_minus_model = cvar_model_lower, 
    cate_bounds_model = cate_bounds_model, 
    use_rho=True,
    gamma=gamma)
BLearner_est.fit(X, A, Y)
lower_bound, upper_bound = BLearner_est.effect(X_test)
```

## Replication Code for Paper

The following commands will replicate the figures from the [B-Learner](https://arxiv.org/abs/2304.10577) paper.

* For Figure 1, run `python rates.py`
* For Figure 2, run `python compute_intervals.py`
* For Figure 3, run `python 401k.py`