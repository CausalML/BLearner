#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from joblib import Parallel, delayed
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 50
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
rc = {
    "figure.constrained_layout.use": True,
    "axes.titlesize": 20,
}
sns.set_theme(style="darkgrid", palette="colorblind", rc=None)

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn_quantile import RandomForestQuantileRegressor
from econml.dr import DRLearner
from models.blearner.nuisance import RFKernel, KernelSuperquantileRegressor

# This method
from models.blearner import BLearner
# The DR Learner
from econml.dr import DRLearner
from doubleml.datasets import fetch_401K

def ggplot_style_grid(figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    # Give plot a gray background like ggplot.
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 16
    ax.set_facecolor('#EBEBEB')
    # Remove border around plot.
    [ax.spines[side].set_visible(False) for side in ax.spines]
    # Style the grid.
    ax.grid(which='major', color='white', linewidth=1.2)
    ax.grid(which='minor', color='white', linewidth=0.6)
    # Show the minor ticks and grid.
    ax.minorticks_on()
    # Now hide the minor ticks (but leave the gridlines).
    ax.tick_params(which='minor', bottom=False, left=False)
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(True)
        axis.set_major_formatter(formatter)
    return ax

if __name__ == "__main__":
    # 401K data
    df = fetch_401K(return_type='DataFrame')
    # X1: age (int)
    # X2: inc -> income (int)
    # X3: educ -> education, in #years completed (int)
    # X4: fsize -> family size (int)
    # X5: marr -> marrital status (binary)
    # X6: two_earn -> two earners (binary)
    # X7: db -> defined benefit pension status (binary)
    # X8: pira -> IRA participation
    # X9: hown -> home ownership
    # A: e401 -> 401 (k) eligibility (binary)
    # Y: net_tfa -> net financial assets (float)
    feat_names = ['age', 'inc', 'educ', 'fsize', 'marr', 'twoearn', 'db', 'pira', 'hown']
    X = df[feat_names].values
    A = df["e401"].values
    Y = df["net_tfa"].values

    random_state = 12345
    n_estimators = 100
    max_depth = 7
    max_features = 3
    min_samples_leaf = 10
    log_gamma = 1
    gamma = np.e**log_gamma
    log_gammas = np.arange(0, 1.1, 0.1)


    pct_lb_negative = []
    lower_bounds = []
    upper_bounds = []
    for i, log_gamma in enumerate(log_gammas[1:]):
        print(f"Running with log_gamma={log_gamma}...")
        gamma = np.e**log_gamma
        #Propensity model
        tau = gamma / (1+gamma)
        propensity_model = RandomForestClassifier(
                                n_estimators=n_estimators, 
                                max_depth=max_depth, 
                                max_features=max_features, 
                                min_samples_leaf=min_samples_leaf, 
                                n_jobs=-2)
        # Outcome model
        mu_model = RandomForestRegressor(
                                n_estimators=n_estimators, 
                                max_depth=max_depth, 
                                max_features=max_features, 
                                min_samples_leaf=min_samples_leaf, 
                                n_jobs=-2)
        # Quantiles
        quantile_model_upper = RandomForestQuantileRegressor(n_estimators=n_estimators, 
                                max_depth=max_depth, 
                                max_features=max_features, 
                                min_samples_leaf=min_samples_leaf, 
                                n_jobs=-2,
                                q=tau)
        quantile_model_lower = RandomForestQuantileRegressor(n_estimators=n_estimators, 
                                max_depth=max_depth, 
                                max_features=max_features, 
                                min_samples_leaf=min_samples_leaf, 
                                n_jobs=-2,
                                q=1-tau)
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
        cate_bounds_model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        n_jobs=-2)
        use_rho = True
        CATE_bounds_est = BLearner(propensity_model = propensity_model, 
                                    quantile_plus_model = quantile_model_upper, 
                                    quantile_minus_model = quantile_model_lower,
                                    mu_model = mu_model, 
                                    cvar_plus_model = cvar_model_upper, 
                                    cvar_minus_model = cvar_model_lower, 
                                    cate_bounds_model = cate_bounds_model, 
                                    use_rho=use_rho,
                                    gamma=gamma,
                                    cv=1)
        CATE_bounds_est.fit(X, A, Y)
        lower_bound, upper_bound = CATE_bounds_est.effect(X)
        pct_lb_negative.append(np.mean(lower_bound<=0))
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    CATE_est = DRLearner(
        model_propensity=propensity_model, 
        model_regression=mu_model, 
        model_final=cate_bounds_model, 
        cv=1)
    CATE_est.fit(Y, A, X=X)
    cates = CATE_est.effect(X)
    pct_lb_negative = [np.mean(cates<=0)] + pct_lb_negative

    # Plot negative fraction vs Lambda
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    # Give plot a gray background like ggplot.
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 16
    ax.set_facecolor('#EBEBEB')
    plt.plot(log_gammas, np.array(pct_lb_negative)*100)
    plt.ylabel("% Negative CATE Lower Bounds")
    plt.xlabel("$\log(\Lambda)$")
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.savefig(f"results/401k/401k_negative_lb_vs_log_lambda.pdf", bbox_inches="tight", dpi=100)
    plt.show()


    for i in [1, 4]:
        ax = ggplot_style_grid(figsize=(4, 4))
        effect_bins = np.arange(-15000, 50000, 2500)
        #lower_bound, upper_bound = CATE_bounds_est.effect(X)
        plt.hist(lower_bounds[i], bins=effect_bins, histtype="step", label="Lower bound", zorder=5, color="C0", 
                density=True, lw=1.2)
        plt.hist(CATE_est.effect(X), bins=effect_bins, histtype="step", label="CATE", zorder=6, color="black", ls='--',
                density=True, lw=1.2)
        plt.hist(upper_bounds[i], bins=effect_bins, histtype="step", label="Upper bound", zorder=7, color="C3",
                density=True, lw=1.2)
        log_gamma = (i+1)*0.1
        plt.xlabel(f"Effect ($\log(\Lambda)={log_gamma:0.1f}$)")
        plt.ylabel("Density")
        plt.legend(prop={'size': 12})
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.savefig(f"results/401k/401k_hist_log_lambda_{log_gamma:0.1f}.pdf", bbox_inches="tight", dpi=100)
        plt.show()




