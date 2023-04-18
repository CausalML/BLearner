#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from joblib import Parallel, delayed
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 50
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
rc = {
    "figure.constrained_layout.use": True,
    "axes.titlesize": 20,
}
sns.set_theme(style="darkgrid", palette="colorblind", rc=None)
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from econml.grf import RegressionForest
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm 

from datasets.synthetic import Synthetic
from models.blearner.nuisance import (
    RFKernel, KernelSuperquantileRegressor,
    KernelQuantileRegressor, RBFKernel
)

# This method
from models.blearner import BLearner, NNBLearner
from models.quince.quince import Quince
from models.kernel.kernel import KernelRegressor

class RBFKernelWeights:
    def __init__(self, scale=1):
        self.kernel = RBF(length_scale=scale)
        
    def fit(self, X, Y):
        self.X_train = X
        return self
        
    def predict(self, X):
        weights = self.kernel(X, self.X_train)
        # Normalize weights
        norm_weights = weights/weights.sum(axis=1).reshape(-1, 1)
        return norm_weights

def gen_rf_nuisances(n_estimators=100, max_depth=6, min_samples_leaf=0.05, gamma=1, grf=False):
    tau = gamma / (1+gamma)
    #propensity_model = LogisticRegression(C=10)
    propensity_model = LogisticRegression(solver="saga", penalty="elasticnet", l1_ratio=0.5)
    if grf:
        core_model = RegressionForest(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        n_jobs=-2)
    else:
        core_model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        n_jobs=-2)
        
    # Mu model
    mu_model = clone(core_model, safe=False)
    ### Quantile and CVaR models
    # Models for tau quantile
    quantile_model_upper = KernelQuantileRegressor( 
        kernel=RFKernel(clone(core_model, safe=False)),
        tau=tau)
    cvar_model_upper = KernelSuperquantileRegressor(
        kernel=RFKernel(clone(core_model, safe=False)),
        tau=tau,
        tail="right")
    # Models for 1-tau quantile
    quantile_model_lower = KernelQuantileRegressor( 
        kernel=RFKernel(clone(core_model, safe=False)),
        tau=1-tau)
    cvar_model_lower = KernelSuperquantileRegressor(
        kernel=RFKernel(clone(core_model, safe=False)),
        tau=1-tau,
        tail="left")
    # Second stage model
    cate_bounds_model = clone(core_model, safe=False)
    return {
        "propensity_model": propensity_model,
        "quantile_plus_model": quantile_model_upper,
        "quantile_minus_model": quantile_model_lower,
        "mu_model": mu_model,
        "cvar_plus_model": cvar_model_upper, 
        "cvar_minus_model": cvar_model_lower, 
        "cate_bounds_model": cate_bounds_model

    }

def gen_rbf_nuisances(scale=0.1, gamma=1):
    tau = gamma / (gamma+1)
    #propensity_model = LogisticRegression(C=10)
    propensity_model = LogisticRegression(solver="saga", penalty="elasticnet", l1_ratio=0.5)
    mu_model = RBFKernel(scale=scale)
    core_model = RBFKernelWeights(scale=scale)
    ### Quantile and CVaR models
    # Models for tau quantile
    quantile_model_upper = KernelQuantileRegressor( 
        kernel=clone(core_model, safe=False),
        tau=tau)
    cvar_model_upper = KernelSuperquantileRegressor(
        kernel=clone(core_model, safe=False),
        tau=tau,
        tail="right")
    # Models for 1-tau quantile
    quantile_model_lower = KernelQuantileRegressor( 
        kernel=clone(core_model, safe=False),
        tau=1-tau)
    cvar_model_lower = KernelSuperquantileRegressor(
        kernel=clone(core_model, safe=False),
        tau=1-tau,
        tail="left")
    # Second stage model
    cate_bounds_model = RBFKernel(scale=scale)
    return {
        "propensity_model": propensity_model,
        "quantile_plus_model": quantile_model_upper,
        "quantile_minus_model": quantile_model_lower,
        "mu_model": mu_model,
        "cvar_plus_model": cvar_model_upper, 
        "cvar_minus_model": cvar_model_lower, 
        "cate_bounds_model": cate_bounds_model

    }

def prop_func(X):
    return 1/(1+np.exp(-0.75*X[:, 0]-0.5))

def outcome_func(X, A):
    return (2*A-1)*(X[:, 0]+1)-2*np.sin((4*A-2)*X[:, 0])

class HSKStats:
    def __init__(self, tau, noise_scale=1, alpha=3, n_samples=20000):
        self.tau = tau
        self.noise_scale = noise_scale
        self.alpha = alpha
        self.n_samples = n_samples
    
    def fit(self):
        self.X_sigma = np.arange(0, 4*self.noise_scale, 0.001)
        ys = np.array([self.alpha * x * (2*np.random.binomial(1, 0.5, size=self.n_samples)-1) \
            + np.random.normal(scale=self.noise_scale, size=self.n_samples) for x in self.X_sigma])
        self.q_upper = np.array([np.quantile(ys[i], q=self.tau) for i in range(len(ys))])
        self.q_lower = np.array([np.quantile(ys[i], q=1-self.tau) for i in range(len(ys))])
        self.cvar_upper = np.array([np.mean(ys[i][ys[i] >= self.q_upper[i]]) for i in range(len(ys))])
        self.cvar_lower = np.array([np.mean(ys[i][ys[i] <= self.q_lower[i]]) for i in range(len(ys))])
        return self
    
    def predict(self, X_sigma):
        return (
            np.interp(np.abs(X_sigma), self.X_sigma, self.q_upper),
            np.interp(np.abs(X_sigma), self.X_sigma, self.q_lower),
            np.interp(np.abs(X_sigma), self.X_sigma, self.cvar_upper),
            np.interp(np.abs(X_sigma), self.X_sigma, self.cvar_lower)
        )

class LogNormalDGP:
    def __init__(self, seed=1, gamma=1, noise_scale=0.2, heterosk=True):
        self.seed = seed
        self.noise_scale = noise_scale
        self.prop_func = prop_func
        self.outcome_func = outcome_func
        self.heterosk = heterosk
        self.gamma = gamma
        self.tau = gamma / (1+gamma)
    
    def get_data(self, n, p=1):
        np.random.seed(self.seed)
        X = np.random.uniform(-1, 1, size=(n, p))
        A = np.random.binomial(1, self.prop_func(X))
        Y = np.random.lognormal(self.outcome_func(X, A), self.noise_scale) 
        return X, A, Y
        
    def get_cate_bounds(self, X):
        nuis = self.get_oracle_nuisances(X, use_rho=True)
        e = nuis[:, 0]
        mu_0 = nuis[:, 5]
        mu_1 = nuis[:, 6]
        rho_plus_0 = nuis[:, 7]
        rho_plus_1 = nuis[:, 8]
        rho_minus_0 = nuis[:, 9]
        rho_minus_1 = nuis[:, 10]
        psi_plus_1 = e * mu_1 + (1-e)*rho_plus_1
        psi_minus_1 = e * mu_1 + (1-e)*rho_minus_1
        psi_plus_0 = (1-e) * mu_0 + e*rho_plus_0
        psi_minus_0 = (1-e) * mu_0 + e*rho_minus_0
        upper_cate_true, lower_cate_true = psi_plus_1 - psi_minus_0, psi_minus_1 - psi_plus_0
        return lower_cate_true, upper_cate_true
    
    def get_oracle_nuisances(self, X, use_rho=True):
        from scipy.special import expit, erf, erfinv
        e = self.prop_func(X)
        log_mu_0 = self.outcome_func(X, 0)
        log_mu_1 = self.outcome_func(X, 1)
        q0_plus = np.exp(log_mu_0+self.noise_scale*np.sqrt(2)*erfinv(2*self.tau-1))
        q0_minus = np.exp(log_mu_0+self.noise_scale*np.sqrt(2)*erfinv(2*(1-self.tau)-1))
        q1_plus = np.exp(log_mu_1+self.noise_scale*np.sqrt(2)*erfinv(2*self.tau-1))
        q1_minus = np.exp(log_mu_1+self.noise_scale*np.sqrt(2)*erfinv(2*(1-self.tau)-1))
        cvar0_plus = 0.5*np.exp(log_mu_0+self.noise_scale**2/2)*(
            1+erf(self.noise_scale/np.sqrt(2)-erfinv(2*self.tau-1)))/(1-self.tau)
        cvar1_plus = 0.5*np.exp(log_mu_1+self.noise_scale**2/2)*(
            1+erf(self.noise_scale/np.sqrt(2)-erfinv(2*self.tau-1)))/(1-self.tau)
        cvar0_minus = 0.5*np.exp(log_mu_0+self.noise_scale**2/2)*(
            1-erf(self.noise_scale/np.sqrt(2)-erfinv(2*(1-self.tau)-1)))/(1-self.tau)
        cvar1_minus = 0.5*np.exp(log_mu_1+self.noise_scale**2/2)*(
            1-erf(self.noise_scale/np.sqrt(2)-erfinv(2*(1-self.tau)-1)))/(1-self.tau)
        mu_0 = np.exp(self.outcome_func(X, 0)+self.noise_scale**2/2)
        mu_1 = np.exp(self.outcome_func(X, 1)+self.noise_scale**2/2)
        rho_plus_1 = (1/self.gamma)*mu_1 + (1-1/self.gamma)*cvar1_plus
        rho_minus_1 = (1/self.gamma)*mu_1 + (1-1/self.gamma)*cvar1_minus  
        rho_plus_0 = (1/self.gamma)*mu_0 + (1-1/self.gamma)*cvar0_plus
        rho_minus_0 = (1/self.gamma)*mu_0 + (1-1/self.gamma)*cvar0_minus
        psi_plus_1 = e * mu_1 + (1-e)*rho_plus_1
        psi_minus_1 = e * mu_1 + (1-e)*rho_minus_1
        psi_plus_0 = (1-e) * mu_0 + e*rho_plus_0
        psi_minus_0 = (1-e) * mu_0 + e*rho_minus_0
        upper_cate_true, lower_cate_true = psi_plus_1 - psi_minus_0, psi_minus_1 - psi_plus_0
        if use_rho:
            return np.hstack((
                e.reshape(-1, 1),
                q0_plus.reshape(-1, 1),
                q1_plus.reshape(-1, 1),
                q0_minus.reshape(-1, 1),
                q1_minus.reshape(-1, 1),
                mu_0.reshape(-1, 1),
                mu_1.reshape(-1, 1),
                rho_plus_0.reshape(-1, 1),
                rho_plus_1.reshape(-1, 1),
                rho_minus_0.reshape(-1, 1),
                rho_minus_1.reshape(-1, 1)
            ))
        else:
            return np.hstack((
                e.reshape(-1, 1),
                q0_plus.reshape(-1, 1),
                q1_plus.reshape(-1, 1),
                q0_minus.reshape(-1, 1),
                q1_minus.reshape(-1, 1),
                mu_0.reshape(-1, 1),
                mu_1.reshape(-1, 1),
                cvar0_plus.reshape(-1, 1),
                cvar1_plus.reshape(-1, 1),
                cvar0_minus.reshape(-1, 1),
                cvar1_minus.reshape(-1, 1)
            ))
        
    def get_test_set(self, p, dx=0.01):
        np.random.seed(self.seed)
        x_test = np.arange(-1, 1+dx, dx).reshape(-1, 1)
        if p > 1:    
            x_test = np.hstack((x_test, np.random.uniform(-1, 1, size=(x_test.shape[0], p-1))))
        return x_test

class SyntheticDGP:
    def __init__(self, seed=1, gamma=1, noise_scale=1, heterosk=False, alpha=3):
        self.seed = seed
        self.noise_scale = noise_scale
        self.prop_func = prop_func
        self.outcome_func = outcome_func
        self.heterosk = heterosk
        self.alpha = alpha
        self.gamma = gamma
        self.tau = gamma / (1+gamma)
        if self.heterosk:
            self.hsk_stats = HSKStats(
                tau=self.tau, 
                noise_scale=self.noise_scale, 
                alpha=self.alpha).fit()
    
    def get_data(self, n, p=1):
        np.random.seed(self.seed)
        if self.heterosk:
            assert p >=2
            X = np.hstack((
                np.random.uniform(-2, 2, size=(n, p-1)),
                np.random.normal(size=(n, 1))
            ))
            A = np.random.binomial(1, self.prop_func(X[:, :-1]))
            Y = self.outcome_func(X, A) + np.random.normal(scale=self.noise_scale, size=n) + \
                self.alpha * np.abs(X[:, -1]) * (2*np.random.binomial(1, 0.5, size=n)-1)
        else:
            X = np.random.uniform(-2, 2, size=(n, p))
            A = np.random.binomial(1, self.prop_func(X))
            Y = self.outcome_func(X, A) + np.random.normal(scale=self.noise_scale, size=n)
        return X, A, Y
        
    def get_cate_bounds(self, X):
        nuis = self.get_oracle_nuisances(X, use_rho=True)
        e = nuis[:, 0]
        mu_0 = nuis[:, 5]
        mu_1 = nuis[:, 6]
        rho_plus_0 = nuis[:, 7]
        rho_plus_1 = nuis[:, 8]
        rho_minus_0 = nuis[:, 9]
        rho_minus_1 = nuis[:, 10]
        psi_plus_1 = e * mu_1 + (1-e)*rho_plus_1
        psi_minus_1 = e * mu_1 + (1-e)*rho_minus_1
        psi_plus_0 = (1-e) * mu_0 + e*rho_plus_0
        psi_minus_0 = (1-e) * mu_0 + e*rho_minus_0
        upper_cate_true, lower_cate_true = psi_plus_1 - psi_minus_0, psi_minus_1 - psi_plus_0
        return lower_cate_true, upper_cate_true
    
    def get_oracle_nuisances(self, X, use_rho=True):
        e = self.prop_func(X)
        mu_0 = self.outcome_func(X, 0)
        mu_1 = self.outcome_func(X, 1)
        if self.heterosk:
            q_upper, q_lower, cvar_upper, cvar_lower = self.hsk_stats.predict(X[:, -1])
            q0_plus = mu_0 + q_upper
            q0_minus = mu_0 + q_lower
            q1_plus = mu_1 + q_upper
            q1_minus = mu_1 + q_lower
            cvar0_plus = mu_0 + cvar_upper
            cvar0_minus = mu_0 + cvar_lower
            cvar1_plus = mu_1 + cvar_upper
            cvar1_minus = mu_1 + cvar_lower
        else:
            rv = norm()
            q0_plus = mu_0 + rv.ppf(self.tau)*self.noise_scale
            q0_minus = mu_0 - rv.ppf(self.tau)*self.noise_scale
            q1_plus = mu_1 + rv.ppf(self.tau)*self.noise_scale
            q1_minus = mu_1 - rv.ppf(self.tau)*self.noise_scale
            cvar0_plus = mu_0 + 1/(1-self.tau)*rv.pdf(rv.ppf(self.tau))*self.noise_scale
            cvar0_minus = mu_0 - 1/(1-self.tau)*rv.pdf(rv.ppf(1-self.tau))*self.noise_scale
            cvar1_plus = mu_1 + 1/(1-self.tau)*rv.pdf(rv.ppf(self.tau))*self.noise_scale
            cvar1_minus = mu_1 - 1/(1-self.tau)*rv.pdf(rv.ppf(self.tau))*self.noise_scale
        rho_plus_1 = (1/self.gamma)*mu_1 + (1-1/self.gamma)*cvar1_plus
        rho_minus_1 = (1/self.gamma)*mu_1 + (1-1/self.gamma)*cvar1_minus  
        rho_plus_0 = (1/self.gamma)*mu_0 + (1-1/self.gamma)*cvar0_plus
        rho_minus_0 = (1/self.gamma)*mu_0 + (1-1/self.gamma)*cvar0_minus
        psi_plus_1 = e * mu_1 + (1-e)*rho_plus_1
        psi_minus_1 = e * mu_1 + (1-e)*rho_minus_1
        psi_plus_0 = (1-e) * mu_0 + e*rho_plus_0
        psi_minus_0 = (1-e) * mu_0 + e*rho_minus_0
        upper_cate_true, lower_cate_true = psi_plus_1 - psi_minus_0, psi_minus_1 - psi_plus_0
        if use_rho:
            return np.hstack((
                e.reshape(-1, 1),
                q0_plus.reshape(-1, 1),
                q1_plus.reshape(-1, 1),
                q0_minus.reshape(-1, 1),
                q1_minus.reshape(-1, 1),
                mu_0.reshape(-1, 1),
                mu_1.reshape(-1, 1),
                rho_plus_0.reshape(-1, 1),
                rho_plus_1.reshape(-1, 1),
                rho_minus_0.reshape(-1, 1),
                rho_minus_1.reshape(-1, 1)
            ))
        else:
            return np.hstack((
                e.reshape(-1, 1),
                q0_plus.reshape(-1, 1),
                q1_plus.reshape(-1, 1),
                q0_minus.reshape(-1, 1),
                q1_minus.reshape(-1, 1),
                mu_0.reshape(-1, 1),
                mu_1.reshape(-1, 1),
                cvar0_plus.reshape(-1, 1),
                cvar1_plus.reshape(-1, 1),
                cvar0_minus.reshape(-1, 1),
                cvar1_minus.reshape(-1, 1)
            ))
        
    def get_test_set(self, p, dx=0.01):
        np.random.seed(self.seed)
        x_test = np.arange(-2, 2+dx, dx).reshape(-1, 1)
        if p > 1:    
            x_test = np.hstack((x_test, np.random.uniform(-2, 2, size=(x_test.shape[0], p-1))))
            if self.heterosk:
                assert p>=2
                x_test[:, -1] = np.random.normal(scale=self.noise_scale, size=x_test.shape[0])
        return x_test

def run_simulation(model, seed, dgp_class, n, p, gamma, sharp_cate, use_rho):
    dgp_class.seed = seed
    X, A, Y = dgp_class.get_data(n=n, p=p)
    model_iter = clone(model, safe=False)
    model_iter.random_state = seed
    model_iter.fit(X, A, Y)
    if sharp_cate:
        nuis_oracle = dgp.get_oracle_nuisances(X, use_rho=use_rho)
        model_iter.fit_oracle(X, A, Y, nuis_oracle=nuis_oracle)
    return model_iter

FNAME_TEMPLATE = "results/synthetic/rates/{model_config}_n_iter_{n_iter}_n_{n}_p_{p}_gamma_{gamma:0.2f}{hsk}.csv"
def save_results_to_file(trained_models, n, p, heterosk, x_test, true_bounds, gamma, model_config):
    model_name, nuis_name, final_name = model_config
    true_lower, true_upper = true_bounds
    n_iter = len(trained_models)
    colnames = ["model"] + [f"pred{i}" for i in range(x_test.shape[0])] + ["MSE"]
    if model_name == "BLearner":
        model_names = np.hstack((
            np.repeat("BLearner", n_iter), 
            np.repeat("oracle", n_iter), 
            np.repeat("plugin", n_iter),
            ["true_upper"]))
        preds = np.vstack(
                    (
                        np.array([trained_models[i].effect(x_test)[1] for i in range(n_iter)]),
                        np.array([trained_models[i].effect_oracle(x_test)[1] for i in range(n_iter)]),
                        np.array([trained_models[i].effect_plugin(x_test)[1] for i in range(n_iter)]),
                        true_upper,
                    ))
    elif model_name == "Quince":
        def get_quince_bounds(model):
            mu_0_lo, mu_0_hi = model.predict_mu_bounds(x=x_test, a=0, _lambda=gamma, num_samples=1000)
            mu_1_lo, mu_1_hi = model.predict_mu_bounds(x=x_test, a=1, _lambda=gamma, num_samples=1000)
            upper_bound = mu_1_hi - mu_0_lo
            lower_bound = mu_1_lo - mu_0_hi
            return lower_bound, upper_bound 
        model_names = np.hstack((
            np.repeat("Quince", n_iter), 
            ["true_upper"]))
        preds = np.vstack(
                    (np.array([get_quince_bounds(trained_models[i])[1] for i in range(n_iter)]),
                    true_upper
                    ))
    elif model_name == "Kernel":
        model_names = np.hstack((
            np.repeat("Kernel", n_iter), 
            ["true_upper"]))
        preds = np.vstack(
                    (np.array([trained_models[i].predict(x_test, gamma=gamma)[2].flatten() for i in range(n_iter)]),
                    true_upper,
                    ))
    mse = np.array([mean_squared_error(true_upper, pred) for pred in preds]).reshape(-1, 1)
    preds = np.hstack((
        model_names.reshape(-1, 1),
        preds,
        mse
    ))
    preds_fname = FNAME_TEMPLATE.format(
        model_config=model_name if nuis_name is None else f"{model_name}_nuis_{nuis_name}_final_{final_name}",
        n_iter=n_iter,
        n=n,
        p=p,
        gamma=gamma,
        hsk="_hsk" if heterosk else ""
    )
    pd.DataFrame(preds, columns=colnames).to_csv(preds_fname, index=False)

def load_results_from_file(n_iter, n, p, heterosk, gamma, model_config):
    model_name, nuis_name, final_name = model_config
    preds_fname = FNAME_TEMPLATE.format(
        model_config=model_name if nuis_name is None else f"{model_name}_nuis_{nuis_name}_final_{final_name}",
        n_iter=n_iter,
        n=n,
        p=p,
        gamma=gamma,
        hsk="_hsk" if heterosk else ""
    )
    preds = pd.read_csv(preds_fname)
    d = {
        f"{model_name}_MSE": preds[preds.model==model_name]["MSE"].values
    }
    if nuis_name is not None:
        d.update(
            {
            "oracle_MSE": preds[preds.model=="oracle"]["MSE"].values,
            "plugin_MSE": preds[preds.model=="plugin"]["MSE"].values,
            }
        )
    return d

def ggplot_log_style(figsize, log_y=False, loc_maj_large=True):
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
    if log_y:
        ax.loglog()
        if loc_maj_large:
            locmaj_y = LogLocator(base=10.0, subs=(1, 3), numticks=12)
        else:
            locmaj_y = LogLocator(base=10.0, subs=(10**(-0.5),1), numticks=12)
        ax.yaxis.set_major_locator(locmaj_y)
        locmin_y = LogLocator(base=10.0, subs=(10**(-0.25), 10**0.25), numticks=12)
        ax.yaxis.set_minor_locator(locmin_y)
    else:
        ax.set_xscale('log')
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))        
    locmin_x = LogLocator(base=10.0, subs=(10**0.5,),numticks=12)
    ax.xaxis.set_minor_locator(locmin_x)
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
        #axis.set_minor_formatter(formatter)
    return ax

def aggregate_results(n_iter, ns, p, heterosk, gamma, model_config):
    model_name, nuis_name, final_name = model_config
    results = {n: load_results_from_file(n_iter, n, p, heterosk, gamma, model_config) for n in ns}
    if model_name == "BLearner":
        model_means = np.array([results[n][f"BLearner_MSE"] for n in ns]).mean(axis=1)
        oracle_means = np.array([results[n]["oracle_MSE"] for n in ns]).mean(axis=1)
        plugin_means = np.array([results[n]["plugin_MSE"] for n in ns]).mean(axis=1)
        model_sd = np.array([results[n][f"BLearner_MSE"] for n in ns]).std(axis=1) / np.sqrt(n_iter)
        oracle_sd = np.array([results[n]["oracle_MSE"] for n in ns]).std(axis=1) / np.sqrt(n_iter)
        plugin_sd = np.array([results[n]["plugin_MSE"] for n in ns]).std(axis=1) / np.sqrt(n_iter)
        return {
            f"{model_name}_{nuis_name}_{final_name}": model_means,
            f"Oracle_{final_name}": oracle_means,
            f"Plugin_{nuis_name}": plugin_means,
            f"{model_name}_{nuis_name}_{final_name}_SD": model_sd,
            f"Oracle_{final_name}_SD": oracle_sd,
            f"Plugin_{nuis_name}_SD": plugin_sd
        }
    else:
        model_means = np.array([results[n][f"{model_name}_MSE"] for n in ns]).mean(axis=1)
        model_sd = np.array([results[n][f"{model_name}_MSE"] for n in ns]).std(axis=1) / np.sqrt(n_iter)
        return {
            model_name: model_means,
            f"{model_name}_SD": model_sd
        }

if __name__ == "__main__":

    gamma = np.e
    n_iter = 50
    ns = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
    p = 5
    heterosk = False
    dgp = SyntheticDGP(seed=1, gamma=gamma, noise_scale=1, heterosk=heterosk)
    x_test = dgp.get_test_set(p=p, dx=0.01)
    true_bounds = dgp.get_cate_bounds(x_test)

    model_names = ["BLearner", "Quince", "Kernel"]
    nuis_names = ["RF", "RBF", "NN"]
    final_stages = ["RF", "RBF", "NN"]
    model_configs = [
        ("BLearner", "RF", "RF"), 
        ("BLearner", "RBF", "RBF"), 
        ("BLearner", "RBF", "RF"),
        ("Kernel", None, None),
        ("BLearner", "NN", "NN"),
        ("BLearner", "NN", "RF"),
        ("Quince", None, None)
    ]

    # Make results folders if they don't exist
    path =  "./results/synthetic/rates"
    if not os.path.exists(path):
        os.makedirs(path)
    path =  "./results/synthetic/plots"
    if not os.path.exists(path):
        os.makedirs(path)
    
    for model_config in model_configs:
        for n in tqdm(ns):
            print(f"Running {model_config} for n={n}...")
            # Hyperparams
            kernel_scale = 0.9*n**(-1/(4+p))
            nuisances_rf = gen_rf_nuisances(
                n_estimators=200, 
                max_depth=6, 
                min_samples_leaf=0.02, 
                gamma=gamma, 
                grf=True)
            nuisances_rbf = gen_rbf_nuisances(
                scale=kernel_scale,
                gamma=gamma)
            nn_args = {
                "dim_hidden": 100,
                "depth": 4,
                "negative_slope": 0.3,
                "layer_norm": False,
                "dropout_rate": 0.2,
                "learning_rate": 5e-4,
                "batch_size": 50,
                "max_epochs": 500,
                "verbose": False,
                "num_components": 2,
                "patience":50
            }
            use_rho = True
            propensity_model = LogisticRegression(solver="saga", penalty="elasticnet", l1_ratio=0.5)
            # Configs
            model_name, nuis_name, final_name = model_config
            use_nn = True if model_name=="Quince" or nuis_name=="NN" else False
            sharp_cate = (model_name == "BLearner")
            if use_nn:
                if model_name == "BLearner":
                    if final_name == "NN":
                        model = NNBLearner(nn_args=nn_args,
                                            use_rho=use_rho,
                                            gamma=gamma,
                                            cv=1)
                    else:
                        model = NNBLearner(nn_args=nn_args,
                                            cate_bounds_model=nuisances_rf["cate_bounds_model"],
                                            use_rho=use_rho,
                                            gamma=gamma,
                                            cv=1)
                elif model_name == "Quince":
                    model = Quince(**nn_args)
                trained_models = [run_simulation(model=model,
                                                seed=seed,
                                                dgp_class=dgp,
                                                n=n,
                                                p=p,
                                                gamma=gamma,
                                                sharp_cate=sharp_cate,
                                                use_rho=use_rho) for seed in np.arange(n_iter)]
            else:
                if model_name == "BLearner":
                    if nuis_name == "RF":
                        nuisances = nuisances_rf.copy()
                    if nuis_name == "RBF":
                        nuisances = nuisances_rbf.copy()
                        if final_name == "RF":
                            nuisances["cate_bounds_model"] = nuisances_rf["cate_bounds_model"]
                    model = BLearner(
                            **nuisances,
                            use_rho=True,
                            gamma=gamma,
                            cv=1)
                elif model_name == "Kernel":
                    model = KernelRegressor(
                        initial_length_scale=kernel_scale,
                        propensity_model=propensity_model
                        )
                trained_models = Parallel(n_jobs=-2, backend='loky', verbose=1)(
                                        delayed(run_simulation)(model=model,
                                                                seed=seed,
                                                                dgp_class=dgp,
                                                                n=n,
                                                                p=p,
                                                                gamma=gamma,
                                                                sharp_cate=sharp_cate,
                                                                use_rho=use_rho
                                                                ) for seed in np.arange(n_iter))
            save_results_to_file(trained_models, 
                                n=n, p=p, 
                                x_test=x_test,
                                heterosk=heterosk,
                                true_bounds=true_bounds, 
                                gamma=gamma, 
                                model_config=model_config)

    name_to_label = {
        "Oracle_RF": r"$\widehat{\tau}^+$(Oracle, RF)",
        "Oracle_RBF": r"$\widehat{\tau}^+$(Oracle, GK)",
        "Oracle_NN": r"$\widehat{\tau}^+$(Oracle, NN)",
        "Plugin_RF": r"$\widehat{\tau}^+$(RF, Plugin)",
        "Plugin_RBF": r"$\widehat{\tau}^+$(GK, Plugin)",
        "Plugin_NN": r"$\widehat{\tau}^+$(NN, Plugin)",
        "BLearner_RF_RF":r"$\widehat{\tau}^+$(RF, RF)",
        "BLearner_RBF_RBF":r"$\widehat{\tau}^+$(GK, GK)",
        "BLearner_NN_NN":r"$\widehat{\tau}^+$(NN, NN)",
        "BLearner_RBF_RF":r"$\widehat{\tau}^+$(GK, RF)",
        "BLearner_NN_RF":r"$\widehat{\tau}^+$(NN, RF)",
        "Kernel": "Sensitivity Kernel",
        "Quince": "Quince"
    }

    # Example for double-robustness
    model_config = model_configs[0]
    lss = ['-', '--', ':']
    agg_results_rf_rf = aggregate_results(n_iter, ns, p, heterosk, gamma, model_config)
    ax = ggplot_log_style(figsize=(7, 4), log_y=True)
    for i, (k, v) in enumerate(agg_results_rf_rf.items()):
        if "SD" not in k:
            plt.plot(ns, v, label=name_to_label[k], ls=lss[i], color="C0")
            model_sd = agg_results_rf_rf[f"{k}_SD"]
            plt.fill_between(ns, v - model_sd, v + model_sd, 
                            color="C0",alpha=0.2)
            plt.plot(ns, v, ls=lss[i], color="C0")
    ax.set_xlabel("n")
    ax.set_ylabel("Mean Squared Error")
    plt.legend(fontsize=12)
    plt.savefig(f"results/synthetic/plots/MSE_RF_Rates_n_iter_{n_iter}_p_{p}_gamma_{gamma:0.2f}.pdf", dpi=200)

    # All methods + ours with different second stages
    model_config = model_configs[3]
    agg_results_kernel = aggregate_results(n_iter, ns, p, heterosk, gamma, model_config)
    model_config = model_configs[2]
    agg_results_rbf_rf = aggregate_results(n_iter, ns, p, heterosk, gamma, model_config)
    model_config = model_configs[1]
    agg_results_rbf_rbf = aggregate_results(n_iter, ns, p, heterosk, gamma, model_config)
    model_config = model_configs[0]
    agg_results_rf_rf = aggregate_results(n_iter, ns, p, heterosk, gamma, model_config)

    model_config = model_configs[4]
    agg_results_nn_nn = aggregate_results(n_iter, ns, p, heterosk, gamma, model_config)
    model_config = model_configs[5]
    agg_results_nn_rf = aggregate_results(n_iter, ns, p, heterosk, gamma, model_config)
    model_config = model_configs[6]
    agg_results_quince = aggregate_results(n_iter, ns, p, heterosk, gamma, model_config)
    lss = ['-', '--', '-.', ':']
    ax = ggplot_log_style(figsize=(7, 4), log_y=True, loc_maj_large=False)

    # Gaussian Kernels
    v = agg_results_kernel["Kernel"]
    model_sd = agg_results_kernel["Kernel_SD"]
    plt.plot(ns, v, label=name_to_label["Kernel"], color="C2", ls=lss[1])
    plt.fill_between(ns, v - model_sd, v + model_sd, color="C2", alpha=0.15)

    v = agg_results_rbf_rbf["BLearner_RBF_RBF"]
    model_sd = agg_results_rbf_rbf["BLearner_RBF_RBF_SD"]
    plt.plot(ns, v, label=name_to_label["BLearner_RBF_RBF"], color="C2", ls=lss[0])
    plt.fill_between(ns, v - model_sd, v + model_sd, color="C2", alpha=0.15)

    # Neural networks 
    v = agg_results_quince["Quince"]
    model_sd = agg_results_quince["Quince_SD"]
    plt.plot(ns, v, label=name_to_label["Quince"], color="C4", ls=lss[2])
    plt.fill_between(ns, v - model_sd, v + model_sd, color="C4", alpha=0.15)

    v = agg_results_nn_nn["BLearner_NN_NN"]
    model_sd = agg_results_nn_nn["BLearner_NN_NN_SD"]
    plt.plot(ns, v, label=name_to_label["BLearner_NN_NN"], color="C4", ls=lss[0])
    plt.fill_between(ns, v - model_sd, v + model_sd, color="C4", alpha=0.15)

    # Random Forests
    v = agg_results_rbf_rf["BLearner_RBF_RF"]
    model_sd = agg_results_rbf_rf["BLearner_RBF_RF_SD"]
    plt.plot(ns, v, label=name_to_label["BLearner_RBF_RF"], color="C0", ls=lss[1])
    plt.fill_between(ns, v - model_sd, v + model_sd, color="C0", alpha=0.15)

    v = agg_results_nn_rf["BLearner_NN_RF"]
    model_sd = agg_results_nn_rf["BLearner_NN_RF_SD"]
    plt.plot(ns, v, label=name_to_label["BLearner_NN_RF"], color="C0", ls=lss[2])
    plt.fill_between(ns, v - model_sd, v + model_sd, color="C0", alpha=0.15)

    v = agg_results_rf_rf["BLearner_RF_RF"]
    model_sd = agg_results_rf_rf["BLearner_RF_RF_SD"]
    plt.plot(ns, v, label=name_to_label["BLearner_RF_RF"], color="C0", ls=lss[0])
    plt.fill_between(ns, v - model_sd, v + model_sd, color="C0", alpha=0.15)

    ax.set_xlabel("n")
    ax.set_ylabel("Mean Squared Error")
    plt.legend(fontsize=12, loc=(1.02, 0.42))
    plt.savefig(f"results/synthetic/plots/MSE_all_Rates_n_iter_{n_iter}_p_{p}_gamma_{gamma:0.2f}.pdf", dpi=200, bbox_inches="tight")
