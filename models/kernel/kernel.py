import torch
import numpy as np

from sklearn.gaussian_process import kernels
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_kernels

def _alpha_func(pi, lambda_):
    return 1 / (lambda_ * pi) + 1 - 1 / (lambda_)


def _beta_func(pi, lambda_):
    return lambda_ / (pi) + 1 - lambda_


class KernelRegressor:
    def __init__(
        self,
        initial_length_scale=1.0,
        propensity_model=None,
        verbose=False,
    ):
        self.kernel = kernels.RBF(length_scale=initial_length_scale)
        if propensity_model is None:
            self.propensity_model = LogisticRegression()
        else:
            self.propensity_model = propensity_model

        self._gamma = None
        self.alpha_0 = None
        self.alpha_1 = None
        self.beta_0 = None
        self.beta_1 = None
        self.verbose = verbose

    def fit(self, X, A, Y):
        idx = np.argsort(Y.ravel())
        self.x = X[idx]
        self.t = A[idx].reshape(-1, 1)
        self.y = Y[idx].reshape(-1, 1)
        self.s = self.y.std()
        self.m = self.y.mean()
        self.propensity_model.fit(self.x, self.t.ravel())
        self.e = self.propensity_model.predict_proba(self.x)[:, -1:]

    def predict(self, X, gamma):
        self._gamma = gamma
        self.alpha_0 = _alpha_func(pi=1 - self.e, lambda_=gamma)
        self.alpha_1 = _alpha_func(pi=self.e, lambda_=gamma)

        self.beta_0 = _beta_func(pi=1 - self.e, lambda_=gamma)
        self.beta_1 = _beta_func(pi=self.e, lambda_=gamma)

        k = self.k(X)

        lambda_top_1 = []
        lambda_top_0 = []
        lambda_bottom_1 = []
        lambda_bottom_0 = []
        for i in range(k.shape[0]):
            lambda_top_1.append(self.lambda_top_1(i, k).reshape(1, -1))
            lambda_top_0.append(self.lambda_top_0(i, k).reshape(1, -1))
            lambda_bottom_1.append(self.lambda_bottom_1(i, k).reshape(1, -1))
            lambda_bottom_0.append(self.lambda_bottom_0(i, k).reshape(1, -1))
        lambda_top_1 = np.vstack(lambda_top_1)
        lambda_top_0 = np.vstack(lambda_top_0)
        lambda_bottom_1 = np.vstack(lambda_bottom_1)
        lambda_bottom_0 = np.vstack(lambda_bottom_0)

        tau_top = []
        tau_bottom = []
        for i in range(k.shape[0]):
            tau_top.append(
                lambda_top_1[:, i : i + 1].max(axis=0)
                - lambda_bottom_0[:, i : i + 1].min(axis=0)
            )
            tau_bottom.append(
                lambda_bottom_1[:, i : i + 1].min(axis=0)
                - lambda_top_0[:, i : i + 1].max(axis=0)
            )
        tau_top = np.stack(tau_top)
        tau_bottom = np.stack(tau_bottom)
        tau_mean = self.tau(k=k)
        return tau_mean, tau_bottom, tau_top

    def k(self, x):
        return pairwise_kernels(
            self.embed(x), self.embed(self.x), metric=self.kernel, filter_params=True
        )

    def mu0_w(self, w, k):
        return np.matmul(k, (1 - self.t) * self.y * w) / (
            np.matmul(k, (1 - self.t) * w) + 1e-7
        )

    def mu1_w(self, w, k):
        return np.matmul(k, self.t * self.y * w) / (np.matmul(k, self.t * w) + 1e-7)

    def lambda_top_0(self, u, k):
        t = 1 - self.t
        alpha = np.matmul(k[:, :u], t[:u] * self.alpha_0[:u])
        beta = np.matmul(k[:, u:], t[u:] * self.beta_0[u:])
        alpha_y = np.matmul(k[:, :u], t[:u] * self.alpha_0[:u] * self.y[:u])
        beta_y = np.matmul(k[:, u:], t[u:] * self.beta_0[u:] * self.y[u:])
        return (alpha_y + beta_y) / (alpha + beta)

    def lambda_top_1(self, u, k):
        t = self.t
        alpha = np.matmul(k[:, :u], t[:u] * self.alpha_1[:u])
        beta = np.matmul(k[:, u:], t[u:] * self.beta_1[u:])
        alpha_y = np.matmul(k[:, :u], t[:u] * self.alpha_1[:u] * self.y[:u])
        beta_y = np.matmul(k[:, u:], t[u:] * self.beta_1[u:] * self.y[u:])
        return (alpha_y + beta_y) / (alpha + beta)

    def lambda_bottom_0(self, u, k):
        t = 1 - self.t
        alpha = np.matmul(k[:, u:], t[u:] * self.alpha_0[u:])
        beta = np.matmul(k[:, :u], t[:u] * self.beta_0[:u])
        alpha_y = np.matmul(k[:, u:], t[u:] * self.alpha_0[u:] * self.y[u:])
        beta_y = np.matmul(k[:, :u], t[:u] * self.beta_0[:u] * self.y[:u])
        return (alpha_y + beta_y) / (alpha + beta)

    def lambda_bottom_1(self, u, k):
        t = self.t
        alpha = np.matmul(k[:, u:], t[u:] * self.alpha_1[u:])
        beta = np.matmul(k[:, :u], t[:u] * self.beta_1[:u])
        alpha_y = np.matmul(k[:, u:], t[u:] * self.alpha_1[u:] * self.y[u:])
        beta_y = np.matmul(k[:, :u], t[:u] * self.beta_1[:u] * self.y[:u])
        return (alpha_y + beta_y) / (alpha + beta)

    def mu0(self, k):
        return self.mu0_w(w=(1 - self.e) ** -1, k=k)

    def mu1(self, k):
        return self.mu1_w(w=self.e ** -1, k=k)

    def tau(self, k):
        return self.mu1(k) - self.mu0(k)

    def fit_length_scale(self, dataset, grid):
        best_err = np.inf
        best_h = None
        count = 0
        for h in grid:
            kernel = kernels.RBF(length_scale=h)
            k = pairwise_kernels(
                self.embed(dataset.x),
                self.embed(self.x),
                metric=kernel,
                filter_params=False,
            )
            mu0 = self.mu0(k)
            mu1 = self.mu1(k)
            y = dataset.y.reshape(-1, 1)
            t = dataset.t.reshape(-1, 1)
            err0 = mean_squared_error(y[t == 0], mu0[t == 0])
            err1 = mean_squared_error(y[t == 1], mu1[t == 1])
            err = err0 + err1
            if err < best_err:
                best_err = err
                best_h = h
                count = 0
            elif count < 20:
                count += 1
            else:
                break
            if self.verbose:
                print(f"h-{h:.03f}_err-{err:.03f}")
        self.kernel.length_scale = best_h

    def embed(self, x):
        return x