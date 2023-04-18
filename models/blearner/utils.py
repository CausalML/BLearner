
import numpy as np
import torch
from sklearn import clone
from .nuisance import PropensityNet, MuNet
from ..quince.quince import Quince


##################
# Wrapper models #
##################

def _crossfit(model, folds, X, A, Y, X_val=None, A_val=None, Y_val=None):
    model_list = []
    fitted_inds = []

    for idx, (train_idxs, test_idxs) in enumerate(folds):
        model_list.append(clone(model, safe=False))
        fitted_inds = np.concatenate((fitted_inds, test_idxs))
        model_list[idx].fit(X[train_idxs], A[train_idxs], Y[train_idxs], X_val, A_val, Y_val)
        nuisance = model_list[idx].predict(X[test_idxs])
        if idx == 0:
            nuisances = np.full((X.shape[0], nuisance.shape[1]), np.nan)
        nuisances[test_idxs] = nuisance
    return nuisances, model_list


class CATE_Nuisance_Model:
    def __init__(self,
                 propensity_model,
                 quantile_plus_model,
                 quantile_minus_model,
                 mu_model,
                 cvar_plus_model,
                 cvar_minus_model,
                 gamma=1,
                 use_rho=False):
        self.use_rho = use_rho
        self.gamma = gamma
        self.tau = self.gamma / (1 + self.gamma)
        self.propensity_model = clone(propensity_model, safe=False)
        self.quantile_plus_models = [clone(quantile_plus_model, safe=False), clone(quantile_plus_model, safe=False)]
        self.quantile_minus_models = [clone(quantile_minus_model, safe=False), clone(quantile_minus_model, safe=False)]
        if self.use_rho:
            self.mu_models = [clone(mu_model, safe=False), clone(mu_model, safe=False)]
            self.rho_plus_models = [clone(mu_model, safe=False), clone(mu_model, safe=False)]
            self.rho_minus_models = [clone(mu_model, safe=False), clone(mu_model, safe=False)]
        else:
            self.mu_models = [clone(mu_model, safe=False), clone(mu_model, safe=False)]
            self.cvar_minus_models = [clone(cvar_minus_model, safe=False), clone(cvar_minus_model, safe=False)]
            self.cvar_plus_models = [clone(cvar_plus_model, safe=False), clone(cvar_plus_model, safe=False)]

    def fit(self, X, A, Y, X_val=None, A_val=None, Y_val=None):
        self.propensity_model.fit(X, A)
        self.quantile_plus_models[0].fit(X[A == 0], Y[A == 0])
        self.quantile_plus_models[1].fit(X[A == 1], Y[A == 1])
        self.quantile_minus_models[0].fit(X[A == 0], Y[A == 0])
        self.quantile_minus_models[1].fit(X[A == 1], Y[A == 1])
        self.mu_models[0].fit(X[A == 0], Y[A == 0])
        self.mu_models[1].fit(X[A == 1], Y[A == 1])
        if self.use_rho:
            # rho_plus_1
            q_tau_1 = self.quantile_plus_models[1].predict(X[A == 1])
            R_plus_1 = (1 / self.gamma) * Y[A == 1] + (1 - (1 / self.gamma)) * (
                    q_tau_1 + (1 / (1 - self.tau)) * np.maximum(Y[A == 1] - q_tau_1, 0))
            self.rho_plus_models[1].fit(X[A == 1], R_plus_1)
            # rho_minus_1
            q_tau_1 = self.quantile_minus_models[1].predict(X[A == 1])
            R_minus_1 = (1 / self.gamma) * Y[A == 1] + (1 - (1 / self.gamma)) * (
                    q_tau_1 + (1 / (1 - self.tau)) * np.minimum(Y[A == 1] - q_tau_1, 0))
            self.rho_minus_models[1].fit(X[A == 1], R_minus_1)
            # rho_plus_0
            q_tau_0 = self.quantile_plus_models[0].predict(X[A == 0])
            R_plus_0 = (1 / self.gamma) * Y[A == 0] + (1 - (1 / self.gamma)) * (
                    q_tau_0 + (1 / (1 - self.tau)) * np.maximum(Y[A == 0] - q_tau_0, 0))
            self.rho_plus_models[0].fit(X[A == 0], R_plus_0)
            # rho_minus_0
            q_tau_0 = self.quantile_minus_models[0].predict(X[A == 0])
            R_minus_0 = (1 / self.gamma) * Y[A == 0] + (1 - (1 / self.gamma)) * (
                    q_tau_0 + (1 / (1 - self.tau)) * np.minimum(Y[A == 0] - q_tau_0, 0))
            self.rho_minus_models[0].fit(X[A == 0], R_minus_0)

        else:
            self.mu_models[0].fit(X[A == 0], Y[A == 0])
            self.mu_models[1].fit(X[A == 1], Y[A == 1])
            self.cvar_plus_models[0].fit(X[A == 0], Y[A == 0])
            self.cvar_plus_models[1].fit(X[A == 1], Y[A == 1])
            self.cvar_minus_models[0].fit(X[A == 0], Y[A == 0])
            self.cvar_minus_models[1].fit(X[A == 1], Y[A == 1])

    def predict(self, X):
        if self.use_rho:
            predictions = np.hstack((
                self.propensity_model.predict_proba(X)[:, [1]],
                self.quantile_plus_models[0].predict(X).reshape(-1, 1),
                self.quantile_plus_models[1].predict(X).reshape(-1, 1),
                self.quantile_minus_models[0].predict(X).reshape(-1, 1),
                self.quantile_minus_models[1].predict(X).reshape(-1, 1),
                self.mu_models[0].predict(X).reshape(-1, 1),
                self.mu_models[1].predict(X).reshape(-1, 1),
                self.rho_plus_models[0].predict(X).reshape(-1, 1),
                self.rho_plus_models[1].predict(X).reshape(-1, 1),
                self.rho_minus_models[0].predict(X).reshape(-1, 1),
                self.rho_minus_models[1].predict(X).reshape(-1, 1)
            ))
        else:
            predictions = np.hstack((
                self.propensity_model.predict_proba(X)[:, [1]],
                self.quantile_plus_models[0].predict(X).reshape(-1, 1),
                self.quantile_plus_models[1].predict(X).reshape(-1, 1),
                self.quantile_minus_models[0].predict(X).reshape(-1, 1),
                self.quantile_minus_models[1].predict(X).reshape(-1, 1),
                self.mu_models[0].predict(X).reshape(-1, 1),
                self.mu_models[1].predict(X).reshape(-1, 1),
                self.cvar_plus_models[0].predict(X).reshape(-1, 1),
                self.cvar_plus_models[1].predict(X).reshape(-1, 1),
                self.cvar_minus_models[0].predict(X).reshape(-1, 1),
                self.cvar_minus_models[1].predict(X).reshape(-1, 1),
            ))
        return predictions

class NN_Nuisance_Model:

    def __init__(self,
        nn_args,
        use_rho=False,
        gamma=1
    ):
        self.gamma = gamma
        self.tau = self.gamma / (1 + self.gamma)
        self.use_rho = use_rho
        self.propensity_model = PropensityNet(
            tau=self.tau,
            **nn_args
        )
        self.outcome0_model = MuNet(
            tau=self.tau,
            **nn_args
        )
        self.outcome1_model = MuNet(
            tau=self.tau,
            **nn_args
        )

    def fit(self, X, A, Y, X_val=None, A_val=None, Y_val=None):
        # Fit propensity model
        self.propensity_model.fit(X, A)
        # Fit outcome models
        self.outcome0_model.fit(X[A==0], Y[A==0])
        self.outcome1_model.fit(X[A==1], Y[A==1])

    def _predict_outcome(self, outcome_model, X):
        # Density based predictions
        outcome_model.preprocess_for_prediction()
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(X).float()),
            batch_size=2 * outcome_model.batch_size,
            shuffle=False,
        )
        with torch.no_grad():
            # Predict quantiles
            quantile_plus = []
            quantile_minus = []
            mu = []
            cvar_plus = []
            cvar_minus = []
            rho_plus = []
            rho_minus = []
            for batch in dl:
                batch = batch[0].to("cuda" if torch.cuda.is_available() else "cpu")
                y_samples = outcome_model.model(batch).sample(torch.Size([1000]))
                quantile_plus_batch = torch.quantile(y_samples, self.tau, dim=0)
                quantile_minus_batch = torch.quantile(y_samples, 1-self.tau, dim=0)
                quantile_plus.append(quantile_plus_batch)
                quantile_minus.append(quantile_minus_batch)
                mu.append(torch.mean(y_samples, dim=0))
                if self.use_rho:
                    R_minus = (1 / self.gamma) * y_samples + (1 - (1 / self.gamma)) * (
                    quantile_minus_batch + (1 / (1 - self.tau)) * (torch.minimum(y_samples, quantile_minus_batch) - quantile_minus_batch)
                    )
                    rho_minus.append(torch.mean(R_minus, dim=0))
                    R_plus = (1 / self.gamma) * y_samples + (1 - (1 / self.gamma)) * (
                    quantile_plus_batch + (1 / (1 - self.tau)) * (torch.maximum(y_samples, quantile_plus_batch) - quantile_plus_batch)
                    )
                    rho_plus.append(torch.mean(R_plus, dim=0))
                else:
                    mask = (y_samples >= quantile_plus_batch).float()
                    cvar_plus.append(
                        (y_samples * mask).sum(0) / (mask.sum(0) + 1e-7)
                    )
                    mask = (y_samples <= quantile_minus_batch).float()
                    cvar_minus.append(
                        (y_samples * mask).sum(0) / (mask.sum(0) + 1e-7)
                    )
            if self.use_rho:
                return (
                        outcome_model.process_prediction(torch.cat(quantile_plus, dim=0)),
                        outcome_model.process_prediction(torch.cat(quantile_minus, dim=0)),
                        outcome_model.process_prediction(torch.cat(mu, dim=0)),
                        outcome_model.process_prediction(torch.cat(rho_plus, dim=0)),
                        outcome_model.process_prediction(torch.cat(rho_minus, dim=0))
                        )
            else:
                return (
                        outcome_model.process_prediction(torch.cat(quantile_plus, dim=0)),
                        outcome_model.process_prediction(torch.cat(quantile_minus, dim=0)),
                        outcome_model.process_prediction(torch.cat(mu, dim=0)),
                        outcome_model.process_prediction(torch.cat(cvar_plus, dim=0)),
                        outcome_model.process_prediction(torch.cat(cvar_minus, dim=0))
                        )

    def predict(self, X):
        preds0 = self._predict_outcome(self.outcome0_model, X)
        preds1 = self._predict_outcome(self.outcome1_model, X)
        if self.use_rho:
            predictions = np.hstack((
                self.propensity_model.predict_proba(X)[:, [1]],
                preds0[0].reshape(-1, 1),
                preds1[0].reshape(-1, 1),
                preds0[1].reshape(-1, 1),
                preds1[1].reshape(-1, 1),
                preds0[2].reshape(-1, 1),
                preds1[2].reshape(-1, 1),
                preds0[3].reshape(-1, 1),
                preds1[3].reshape(-1, 1),
                preds0[4].reshape(-1, 1),
                preds1[4].reshape(-1, 1)
            ))
        else:
            predictions = np.hstack((
                self.propensity_model.predict_proba(X)[:, [1]],
                preds0[0].reshape(-1, 1),
                preds1[0].reshape(-1, 1),
                preds0[1].reshape(-1, 1),
                preds1[1].reshape(-1, 1),
                preds0[2].reshape(-1, 1),
                preds1[2].reshape(-1, 1),
                preds0[3].reshape(-1, 1),
                preds1[3].reshape(-1, 1),
                preds0[4].reshape(-1, 1),
                preds1[4].reshape(-1, 1),
            ))
        return predictions


class Quince_Nuisance_Model:

    def __init__(self,
        nn_args,
        use_rho=False,
        gamma=1
    ):
        self.gamma = gamma
        self.tau = self.gamma / (1 + self.gamma)
        self.use_rho = use_rho
        self.quince_model = Quince(**nn_args)

    def fit(self, X, A, Y, X_val=None, A_val=None, Y_val=None):
        # Fit quince model
        self.quince_model.fit(x=X, a=A, y=Y, x_val=X_val, a_val=A_val, y_val=Y_val)

    def predict(self, X):
        if self.use_rho:
            rho_minus_0, rho_plus_0 = self.quince_model.predict_rho(x=X, a=0, q=self.tau)
            rho_minus_1, rho_plus_1 = self.quince_model.predict_rho(x=X, a=1, q=self.tau)
            predictions = np.hstack((
                self.quince_model.predict_pi(x=X)[:, [1]],
                self.quince_model.predict_quantile(x=X, a=0, q=self.tau).reshape(-1, 1),
                self.quince_model.predict_quantile(x=X, a=1, q=self.tau).reshape(-1, 1),
                self.quince_model.predict_quantile(x=X, a=0, q=1-self.tau).reshape(-1, 1),
                self.quince_model.predict_quantile(x=X, a=1, q=1-self.tau).reshape(-1, 1),
                self.quince_model.predict_mu(x=X, a=0).reshape(-1, 1),
                self.quince_model.predict_mu(x=X, a=1).reshape(-1, 1),
                rho_plus_0.reshape(-1, 1),
                rho_plus_1.reshape(-1, 1),
                rho_minus_0.reshape(-1, 1),
                rho_minus_1.reshape(-1, 1)
            ))
        else:
            cvar_minus_0, cvar_plus_0 = self.quince_model.predict_cvar(x=X, a=0, q=self.tau)
            cvar_minus_1, cvar_plus_1 = self.quince_model.predict_cvar(x=X, a=1, q=self.tau)
            predictions = np.hstack((
                self.quince_model.predict_pi(x=X)[:, [1]],
                self.quince_model.predict_quantile(x=X, a=0, q=self.tau).reshape(-1, 1),
                self.quince_model.predict_quantile(x=X, a=1, q=self.tau).reshape(-1, 1),
                self.quince_model.predict_quantile(x=X, a=0, q=1-self.tau).reshape(-1, 1),
                self.quince_model.predict_quantile(x=X, a=1, q=1-self.tau).reshape(-1, 1),
                self.quince_model.predict_mu(x=X, a=0).reshape(-1, 1),
                self.quince_model.predict_mu(x=X, a=1).reshape(-1, 1),
                cvar_plus_0.reshape(-1, 1),
                cvar_plus_1.reshape(-1, 1),
                cvar_minus_0.reshape(-1, 1),
                cvar_minus_1.reshape(-1, 1)
            ))
        return predictions
