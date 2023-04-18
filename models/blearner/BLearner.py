from ast import Lambda
import numpy as np
from sklearn import clone
from sklearn.model_selection import KFold

from .utils import (CATE_Nuisance_Model, NN_Nuisance_Model, Quince_Nuisance_Model, _crossfit)
from .nuisance import MuNet


#######################
# Main BLearner class #
#######################

class _BaseBLearner:
    """Base class for BLearner estimators."""

    def __init__(self,
                 nuisance_model,
                 cate_bounds_model,
                 use_rho=False,
                 gamma=1,
                 proj_idx=None,
                 cv=5,
                 random_state=None):
        self.gamma = gamma
        self.tau = self.gamma / (1 + self.gamma)
        self.use_rho = use_rho
        self.cate_upper_model = clone(cate_bounds_model, safe=False)
        self.cate_lower_model = clone(cate_bounds_model, safe=False)
        self.cate_upper_model_oracle = clone(cate_bounds_model, safe=False)
        self.cate_lower_model_oracle = clone(cate_bounds_model, safe=False)
        self.nuisance_model = nuisance_model
        self.proj_idx = proj_idx
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, A, Y, X_val=None, A_val=None, Y_val=None):
        if self.cv > 1:
            folds = list(KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state).split(X))
            nuisances, self.nuisance_models = _crossfit(self.nuisance_model,
                                    folds, X, A, Y, X_val, A_val, Y_val)
        else:
            self.nuisance_model.fit(X, A, Y, X_val, A_val, Y_val)
            nuisances = self.nuisance_model.predict(X)
        self._fit_with_nuisances(X, A, Y, nuisances, oracle=False)
        return self

    def fit_oracle(self, X, A, Y, nuis_oracle):
        self._fit_with_nuisances(X, A, Y, nuis_oracle, oracle=True)
        return self

    def effect(self, X):
        return (
            self.cate_lower_model.predict(X).flatten(), 
            self.cate_upper_model.predict(X).flatten()
            )

    def effect_oracle(self, X):
        return (
            self.cate_lower_model_oracle.predict(X).flatten(), 
            self.cate_upper_model_oracle.predict(X).flatten()
            )

    def effect_plugin(self, X):
        nuisances = self.predict_nuisances(X)
        e = nuisances[:, 0]
        mu_0 = nuisances[:, 5]
        mu_1 = nuisances[:, 6]
        if self.use_rho:
            rho_plus_1 = nuisances[:, 8]
            rho_minus_1 = nuisances[:, 10]  
            rho_plus_0 = nuisances[:, 7]
            rho_minus_0 = nuisances[:, 9]
        else:
            rho_plus_1 = (1/self.gamma)*mu_1 + (1-1/self.gamma)*nuisances[:, 8]
            rho_minus_1 = (1/self.gamma)*mu_1 + (1-1/self.gamma)*nuisances[:, 10]  
            rho_plus_0 = (1/self.gamma)*mu_0 + (1-1/self.gamma)*nuisances[:, 7]
            rho_minus_0 = (1/self.gamma)*mu_0 + (1-1/self.gamma)*nuisances[:, 9]
        psi_plus_1 = e * mu_1 + (1-e)*rho_plus_1
        psi_minus_1 = e * mu_1 + (1-e)*rho_minus_1
        psi_plus_0 = (1-e) * mu_0 + e*rho_plus_0
        psi_minus_0 = (1-e) * mu_0 + e*rho_minus_0
        return (psi_minus_1 - psi_plus_0, psi_plus_1 - psi_minus_0)

    def mu0(self, x):
        if self.cv == 1:
            return self.nuisance_model.predict(x)[:, 5]
        else:
            return np.mean([nuis_model.predict(x)[:, 5] for nuis_model in self.nuisance_models], axis=0)

    def mu1(self, x):
        if self.cv == 1:
            return self.nuisance_model.predict(x)[:, 6]
        else:
            return np.mean([nuis_model.predict(x)[:, 6] for nuis_model in self.nuisance_models], axis=0)

    def predict_nuisances(self, X):
        if self.cv == 1:
            return self.nuisance_model.predict(X)
        else:
            return np.mean([nuis_model.predict(X) for nuis_model in self.nuisance_models], axis=0)

    def _fit_with_nuisances(self, X, A, Y, nuisances, oracle=False):
        phi_plus_1 = self._get_pseudo_outcome_plus_1(X, A, Y, nuisances)
        phi_minus_0 = self._get_pseudo_outcome_minus_0(X, A, Y, nuisances)
        phi_plus = phi_plus_1 - phi_minus_0

        phi_minus_1 = self._get_pseudo_outcome_minus_1(X, A, Y, nuisances)
        phi_plus_0 = self._get_pseudo_outcome_plus_0(X, A, Y, nuisances)
        phi_minus = phi_minus_1 - phi_plus_0

        # Fit final regression models
        if oracle:
            self.cate_upper_model_oracle.fit(X[:, self.proj_idx] if self.proj_idx is not None else X, phi_plus)
            self.cate_lower_model_oracle.fit(X[:, self.proj_idx] if self.proj_idx is not None else X, phi_minus)
        else:
            self.cate_upper_model.fit(X[:, self.proj_idx] if self.proj_idx is not None else X, phi_plus)
            self.cate_lower_model.fit(X[:, self.proj_idx] if self.proj_idx is not None else X, phi_minus)

    def _get_pseudo_outcome_plus_1(self, X, A, Y, nuisances):
        e = nuisances[:, 0]
        q_tau_1 = nuisances[:, 2]
        if self.use_rho:
            rho_plus_1 = nuisances[:, 8]
        else:
            mu_1 = nuisances[:, 6]
            cvar_tau_1 = nuisances[:, 8]
            rho_plus_1 = (1 / self.gamma) * mu_1 + (1 - (1 / self.gamma)) * cvar_tau_1
        R_plus_1 = (1 / self.gamma) * Y + (1 - (1 / self.gamma)) * (
                q_tau_1 + (1 / (1 - self.tau)) * np.maximum(Y - q_tau_1, 0))
        phi = rho_plus_1 + (A / e) * (R_plus_1 - rho_plus_1) + A * Y - A * R_plus_1
        return phi

    def _get_pseudo_outcome_minus_1(self, X, A, Y, nuisances):
        e = nuisances[:, 0]
        q_tau_1 = nuisances[:, 4]
        if self.use_rho:
            rho_minus_1 = nuisances[:, 10]
        else:
            mu_1 = nuisances[:, 6]
            cvar_tau_1 = nuisances[:, 10]
            rho_minus_1 = (1 / self.gamma) * mu_1 + (1 - (1 / self.gamma)) * cvar_tau_1
        R_minus_1 = (1 / self.gamma) * Y + (1 - (1 / self.gamma)) * (
                q_tau_1 + (1 / (1 - self.tau)) * np.minimum(Y - q_tau_1, 0))
        phi = rho_minus_1 + (A / e) * (R_minus_1 - rho_minus_1) + A * Y - A * R_minus_1

        return phi

    def _get_pseudo_outcome_plus_0(self, X, A, Y, nuisances):
        e = nuisances[:, 0]
        q_tau_0 = nuisances[:, 1]
        if self.use_rho:
            rho_plus_0 = nuisances[:, 7]
        else:
            mu_0 = nuisances[:, 5]
            cvar_tau_0 = nuisances[:, 7]
            rho_plus_0 = (1 / self.gamma) * mu_0 + (1 - (1 / self.gamma)) * cvar_tau_0
        R_plus_0 = (1 / self.gamma) * Y + (1 - (1 / self.gamma)) * (
                q_tau_0 + (1 / (1 - self.tau)) * np.maximum(Y - q_tau_0, 0))

        phi = rho_plus_0 + (1 - A) / (1 - e) * (R_plus_0 - rho_plus_0) + (1 - A) * Y - (1 - A) * R_plus_0
        return phi

    def _get_pseudo_outcome_minus_0(self, X, A, Y, nuisances):
        e = nuisances[:, 0]
        q_tau_0 = nuisances[:, 3]
        if self.use_rho:
            rho_minus_0 = nuisances[:, 9]
        else:
            mu_0 = nuisances[:, 5]
            cvar_tau_0 = nuisances[:, 9]
            rho_minus_0 = (1 / self.gamma) * mu_0 + (1 - (1 / self.gamma)) * cvar_tau_0
        R_minus_0 = (1 / self.gamma) * Y + (1 - (1 / self.gamma)) * (
                q_tau_0 + (1 / (1 - self.tau)) * np.minimum(Y - q_tau_0, 0))
        phi = rho_minus_0 + (1 - A) / (1 - e) * (R_minus_0 - rho_minus_0) + (1 - A) * Y - (1 - A) * R_minus_0
        return phi

class BLearner(_BaseBLearner):
    """Estimator for CATE sharp bounds that uses doubly-robust correction techniques.

    Parameters
    ----------
    propensity_model : classification model (scikit-learn or other)
        Estimator for Pr[A=1 | X=x].  Must implement `fit` and `predict_proba` methods.
    quantile_plus_model : quantile regression model (e.g. RandomForestQuantileRegressor)
        Estimator for the 1-tau conditional quantile. Must implement `fit` and `predict` methods.
    quantile_minus_model : quantile regression model (e.g. RandomForestQuantileRegressor)
        Estimator for the tau conditional quantile. Must implement `fit` and `predict` methods.
    mu_model : regression model (scikit-learn or other)
        Estimator for the conditional outcome E[Y | X=x, A=a] when `use_rho=False` or for the modified
        conditional outcome rho_+(x, a)=E[Gamma^{-1}Y+(1-Gamma^{-1}){q + 1/1(1-tau)*max{Y-q, 0}} | X=x, A=a]
        Must implement `fit` and `predict` methods.
    cvar_plus_model : superquantile model (default=None)
        Estimator for the conditional right tau tail CVaR when `use_rho=False`. Must implement `fit` and `predict` methods.
        Only used when `use_rho=False`.
     cvar_minus_model : superquantile model (default=None)
        Estimator for the conditional left tau tail CVaR when `use_rho=False`. Must implement `fit` and `predict` methods.
        Only used when `use_rho=False`.
    use_rho :  bool (default=False)
        Whether to construct rho using a direct regression with plug-in quantiles (`use_rho=True`) or to estimate rho by
        estimating the conditional outcome and conditional CVaR models separately (`use_rho=False`).
    gamma : float, >=1
        Sensitivity model parameter. Must be greater than 1.
    proj_idx : array of int (default=None)
        Feature indices on which to project the cate bounds. Default to using all indices.
    cv : int, (default=5)
        The number of folds to use for K-fold cross-validation.
    random_state : int (default=None)
        Controls the randomness of the estimator.
    """

    def __init__(self,
                 propensity_model,
                 quantile_plus_model,
                 quantile_minus_model,
                 mu_model,
                 cate_bounds_model,
                 cvar_plus_model=None,
                 cvar_minus_model=None,
                 use_rho=False,
                 gamma=1,
                 proj_idx=None,
                 cv=5,
                 random_state=None):
        if not use_rho and (cvar_plus_model is None or cvar_minus_model is None):
            raise ValueError("'cvar_model' parameter cannot be None when use_rho=False.")
        nuisance_model = CATE_Nuisance_Model(propensity_model,
                                             quantile_plus_model, quantile_minus_model,
                                             mu_model,
                                             cvar_plus_model, cvar_minus_model,
                                             use_rho=use_rho,
                                             gamma=gamma)
        super().__init__(
                 nuisance_model=nuisance_model,
                 cate_bounds_model=cate_bounds_model,
                 use_rho=use_rho,
                 gamma=gamma,
                 proj_idx=proj_idx,
                 cv=cv,
                 random_state=random_state)

class NNBLearner(_BaseBLearner):
    def __init__(
        self,
        nn_args,
        final_nn_args=None,
        cate_bounds_model=None,
        use_rho=False,
        gamma=1,
        proj_idx=None,
        cv=5,
        random_state=None):
        nuisance_model = Quince_Nuisance_Model(
            nn_args,
            use_rho=use_rho,
            gamma=gamma)
        if cate_bounds_model is None:
            if final_nn_args is None:
                final_nn_args = nn_args
            cate_bounds_model = MuNet(**final_nn_args) # TODO: maybe this is user inputed?
        super().__init__(
                 nuisance_model=nuisance_model,
                 cate_bounds_model=cate_bounds_model,
                 use_rho=use_rho,
                 gamma=gamma,
                 proj_idx=proj_idx,
                 cv=cv,
                 random_state=random_state)
