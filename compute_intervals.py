import json
import math
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import ray
from scipy import stats
import torch
from tqdm import tqdm
from datasets.ihdp import IHDP

from sklearn.linear_model import LogisticRegression
from econml.grf import RegressionForest
from sklearn.ensemble import RandomForestRegressor
from sklearn_quantile import RandomForestQuantileRegressor
from models.blearner.nuisance import (
    RFKernel, KernelSuperquantileRegressor,
    KernelQuantileRegressor
)
from models.kernel.kernel import KernelRegressor

from models.blearner import NNBLearner, BLearner
from models.quince.quince import Quince

from utils.plotting import fill_between
from sklearn import clone

project_path = Path(os.getcwd()) / "ihdp"
ray.init(num_gpus=4)

GAMMAS = {
    "0.0": math.exp(0.0),
    "0.1": math.exp(0.1),
    "0.2": math.exp(0.2),
    "0.5": math.exp(0.5),
    "0.7": math.exp(0.7),
    "1.0": math.exp(1.0),
    "1.2": math.exp(1.2),
    "1.5": math.exp(1.5),
    "2.0": math.exp(2.0),
    "2.5": math.exp(2.5),
    "3.0": math.exp(3.0),
    "3.5": math.exp(3.5),
    "4.0": math.exp(4.0),
    "4.5": math.exp(4.5),
    "5.0": math.exp(5.0),
    "5.5": math.exp(5.5),
    "6.0": math.exp(6.0),
    "6.5": math.exp(6.5),
    "7.0": math.exp(7.0),
    "7.5": math.exp(7.5),
    "8.0": math.exp(8.0),
    "8.5": math.exp(8.5),
    "9.0": math.exp(9.0),
    "9.5": math.exp(9.5),
    "10.": math.exp(10.0),
}


def errorbar(
        x,
        y,
        y_err,
        x_label,
        y_label,
        title,
        marker_label=None,
        x_pad=-20,
        y_pad=-45,
        legend_loc="upper left",
        file_path=None,
):
    _ = plt.figure(figsize=(682 / 72, 512 / 72), dpi=72)
    plt.errorbar(
        x,
        y,
        yerr=y_err,
        linestyle="None",
        marker="o",
        elinewidth=1.0,
        capsize=2.0,
        label=marker_label,
    )
    lim = max(np.abs(x.min()), np.abs(x.max())) * 1.1
    r = np.arange(-lim, lim, 0.1)
    _ = plt.plot(r, r, label="Ground Truth")
    _ = plt.tick_params(axis="x", direction="in", pad=x_pad)
    _ = plt.tick_params(axis="y", direction="in", pad=y_pad)
    _ = plt.xlabel(x_label)
    _ = plt.ylabel(y_label)
    _ = plt.ylim([-lim, lim])
    _ = plt.legend(loc=legend_loc)
    _ = plt.title(title)
    plt.grid()
    plt.show()


def plot_errorbars(tau_true, tau_hat, log_gamma, title=""):
    tau_mean = tau_hat["mean"].mean(0)
    tau_top = tau_hat["top"].mean(0)
    tau_bottom = tau_hat["bottom"].mean(0)
    tau_top = torch.abs(
        tau_top - tau_mean
    )
    tau_bottom = torch.abs(
        tau_bottom - tau_mean
    )

    errorbar(
        x=tau_true,
        y=tau_mean,
        y_err=torch.cat([tau_top.unsqueeze(0), tau_bottom.unsqueeze(0)]),
        x_label=r"$\tau(\mathbf{x})$",
        y_label=r"$\widehat{\tau}(\mathbf{x})$",
        marker_label=f"$\log\Gamma = $ {log_gamma}",
        x_pad=-20,
        y_pad=-45,
        title=title,
    )


def plot_errorbars_quince(tau_true, tau_hat):
    tau_mean = tau_hat["mean"].mean(0)
    tau_top = torch.abs(
        tau_hat["top"].mean(0) + 2 * tau_hat["mean"].std(0) - tau_mean
    )
    tau_bottom = torch.abs(
        tau_hat["bottom"].mean(0) - 2 * tau_hat["mean"].std(0) - tau_mean
    )
    errorbar(
        x=tau_true,
        y=tau_mean,
        y_err=torch.cat([tau_top.unsqueeze(0), tau_bottom.unsqueeze(0)]),
        x_label=r"$\tau(\mathbf{x})$",
        y_label=r"$\widehat{\tau}(\mathbf{x})$",
        marker_label=f"$\log\Gamma = $ {0}",
        x_pad=-20,
        y_pad=-45,
    )


def compute_intervals_NNBLearner(model_args, final_nn_args, ds_train, ds_valid, ds_test, gamma):
    X_train = ds_train.x
    Y_train = ds_train.y
    A_train = ds_train.t

    X_val = ds_valid.x
    Y_val = ds_valid.y
    A_val = ds_valid.t

    X_test = ds_test.x

    cate_bounds_est = NNBLearner(nn_args=model_args,
                                   final_nn_args=final_nn_args, gamma=gamma, cv=1)
    cate_bounds_est.fit(X=X_train, A=A_train, Y=Y_train, X_val=X_val,
                        A_val=A_val, Y_val=Y_val)

    tau_bottom, tau_top = cate_bounds_est.effect(X_test)
    tau_bottom = tau_bottom.reshape(-1, 1)
    tau_top = tau_top.reshape(-1, 1)

    tau_mean = (cate_bounds_est.mu1(X_test) - cate_bounds_est.mu0(X_test)).reshape(-1, 1)

    tau_hat = {
        "bottom": torch.Tensor(tau_bottom.transpose()),
        "top": torch.Tensor(tau_top.transpose()),
        "mean": torch.Tensor(tau_mean.transpose().tolist()),
    }

    return tau_hat


@ray.remote(num_gpus=0.2)
def compute_all_intervals_NNBLearner(trial, model_args, final_nn_args, ds_train, ds_valid, ds_test):
    intervals_nn = {}
    output_dir = project_path / f"trial-{trial:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "intervals_nn.json"
    if file_path.exists():
        with file_path.open(mode="r") as fp:
            intervals_nn = load_intervals(file_path=file_path)
    else:
        for k, v in GAMMAS.items():
            log_gamma = k
            gamma = v
            tau_hat_nn = compute_intervals_NNBLearner(model_args=model_args, final_nn_args=final_nn_args,
                                                        ds_train=ds_train,
                                                        ds_valid=ds_valid,
                                                        ds_test=ds_test, gamma=gamma)
            # # plot_errorbars(tau_true=tau_true_ihdp, tau_hat=tau_hat_nn, log_gamma=log_gamma, title="NNBLearner")
            intervals_nn.update({k: tau_hat_nn})

        save_intervals(intervals=intervals_nn, file_path=file_path)

    return intervals_nn




def compute_intervals_BLearner(ds_train, ds_valid, ds_test, gamma):
    X_train = ds_train.x
    Y_train = ds_train.y
    A_train = ds_train.t

    X_val = ds_valid.x
    Y_val = ds_valid.y
    A_val = ds_valid.t

    X_test = ds_test.x

    n_estimators = 200
    max_depth = 6
    # max_features = 24
    min_samples_leaf = 0.01
    rbf_scale = 0.15
    final_stage = "RF"
    use_rho = True

    # Train model
    tau = gamma / (1 + gamma)
    propensity_model = LogisticRegression(C=1, penalty='elasticnet', solver='saga', l1_ratio=0.7, max_iter=10000)
    # Mu model
    mu_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        # max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-2)

    # Quantile and CVaR models
    # Models for the tau quantile and cvar
    quantile_model_upper = RandomForestQuantileRegressor(n_estimators=n_estimators,
                                                         max_depth=max_depth,
                                                         # max_features=max_features,
                                                         min_samples_leaf=min_samples_leaf,
                                                         n_jobs=-2,
                                                         q=tau)
    cvar_model_upper = KernelSuperquantileRegressor(
        kernel=RFKernel(
            RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                # max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                n_jobs=-2)
        ),
        tau=tau,
        tail="right")

    # Models for the 1-tau quantile and cvar
    quantile_model_lower = RandomForestQuantileRegressor(n_estimators=n_estimators,
                                                         max_depth=max_depth,
                                                         # max_features=max_features,
                                                         min_samples_leaf=min_samples_leaf,
                                                         n_jobs=-2,
                                                         q=1 - tau)
    cvar_model_lower = KernelSuperquantileRegressor(
        kernel=RFKernel(
            RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                # max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                n_jobs=-2)
        ),
        tau=1 - tau,
        tail="left")
    cate_bounds_model = RegressionForest(
                     n_estimators=n_estimators,
                     max_depth=max_depth,
                     min_samples_leaf=min_samples_leaf,
                     n_jobs=-2)
    # CATE bound model
    cate_bounds_est = BLearner(propensity_model=propensity_model,
                                 quantile_plus_model=quantile_model_upper,
                                 quantile_minus_model=quantile_model_lower,
                                 mu_model=mu_model,
                                 cvar_plus_model=cvar_model_upper,
                                 cvar_minus_model=cvar_model_lower,
                                 cate_bounds_model=cate_bounds_model,
                                 use_rho=use_rho,
                                 gamma=gamma)


    cate_bounds_est.fit(X=X_train, A=A_train, Y=Y_train, X_val=X_val,
                        A_val=A_val, Y_val=Y_val)

    tau_bottom, tau_top = cate_bounds_est.effect(X_test)
    tau_bottom = tau_bottom.reshape(-1, 1)
    tau_top = tau_top.reshape(-1, 1)

    tau_mean = (cate_bounds_est.mu1(X_test) - cate_bounds_est.mu0(X_test)).reshape(-1, 1)

    tau_hat = {
        "bottom": torch.Tensor(tau_bottom.transpose()),
        "top": torch.Tensor(tau_top.transpose()),
        "mean": torch.Tensor(tau_mean.transpose().tolist()),
    }

    return tau_hat


def compute_intervals_NNBLearner_ensemble(model_args, ds_train, ds_valid, ds_test, gamma, n_ensemble):
    tau_means = []
    tau_uppers = []
    tau_lowers = []

    for n in tqdm(range(n_ensemble)):
        intervals_n = compute_intervals_NNBLearner(model_args=model_args, ds_train=ds_train,
                                                     ds_valid=ds_valid, ds_test=ds_test, gamma=gamma)
        tau_means.append(intervals_n["mean"])
        tau_lowers.append(intervals_n["bottom"])
        tau_uppers.append(intervals_n["top"])

    tau_hat = {
        "bottom": (torch.cat(tau_lowers).mean(0)).unsqueeze(0),
        "top": (torch.cat(tau_uppers).mean(0)).unsqueeze(0),
        "mean": (torch.cat(tau_means).mean(0)).unsqueeze(0),
    }

    return tau_hat


def gen_rf_nuisances(n_estimators=100, max_depth=6, min_samples_leaf=0.05, gamma=1, grf=False):
    tau = gamma / (1 + gamma)
    # propensity_model = LogisticRegression(C=10)
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
        tau=1 - tau)
    cvar_model_lower = KernelSuperquantileRegressor(
        kernel=RFKernel(clone(core_model, safe=False)),
        tau=1 - tau,
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
@ray.remote#(num_gpus=0.2)
def compute_all_intervals_BLearner(trial, ds_train, ds_valid, ds_test):
    intervals_rf = {}
    output_dir = project_path / f"trial-{trial:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "intervals_rf.json"
    if file_path.exists():
        with file_path.open(mode="r") as fp:
            intervals_rf = load_intervals(file_path=file_path)
    else:
        for k, v in GAMMAS.items():
            print(k,v)
            log_gamma = k
            gamma = v
            tau_hat_rf = compute_intervals_BLearner(ds_train=ds_train,
                                                      ds_valid=ds_valid,
                                                      ds_test=ds_test, gamma=gamma)
            intervals_rf.update({k: tau_hat_rf})
        save_intervals(intervals=intervals_rf, file_path=file_path)

    return intervals_rf


def compute_intervals_Quince(model_args, ds_train, ds_valid, ds_test, gamma):
    X_train = ds_train.x
    Y_train = ds_train.y
    A_train = ds_train.t

    X_val = ds_valid.x
    Y_val = ds_valid.y
    A_val = ds_valid.t

    X_test = ds_test.x
    num_examples = X_test.shape[0]

    model = Quince(**model_args)
    model.fit(x=X_train, a=A_train, y=Y_train, x_val=X_val, a_val=A_val, y_val=Y_val)

    mu_0 = model.predict_mu(x=X_test, a=0)
    mu_1 = model.predict_mu(x=X_test, a=1)

    tau_mean = (mu_1 - mu_0).reshape(-1, 1)

    mu_0_lo, mu_0_hi = model.predict_mu_bounds(x=X_test, a=0, _lambda=gamma, num_samples=num_examples)
    mu_1_lo, mu_1_hi = model.predict_mu_bounds(x=X_test, a=1, _lambda=gamma, num_samples=num_examples)
    tau_top = (mu_1_hi - mu_0_lo).reshape(-1, 1)
    tau_bottom = (mu_1_lo - mu_0_hi).reshape(-1, 1)

    tau_hat = {
        "bottom": torch.Tensor(tau_bottom.transpose()),
        "top": torch.Tensor(tau_top.transpose()),
        "mean": torch.Tensor(tau_mean.transpose().tolist()),
    }

    return tau_hat

def compute_intervals_Kernel(ds_train, ds_valid, ds_test, gamma):
    X_train = ds_train.x
    Y_train = ds_train.y
    A_train = ds_train.t

    X_val = ds_valid.x
    Y_val = ds_valid.y
    A_val = ds_valid.t

    X_test = ds_test.x
    num_examples = X_test.shape[0]

    #hyper parameters
    p = 10
    kernel_scale = 0.5*num_examples**(-1/(4+p))
    propensity_model = LogisticRegression(C=1, penalty='elasticnet', solver='saga', l1_ratio=0.7, max_iter=10000)

    model = KernelRegressor(
                    initial_length_scale=kernel_scale,
                    propensity_model=propensity_model
                    )
    model.fit(X=X_train, A=A_train, Y=Y_train)

    tau_mean, tau_bottom, tau_top = model.predict(X=X_test, gamma=gamma)
    tau_mean = tau_mean.reshape(-1, 1)
    tau_bottom = tau_bottom.reshape(-1, 1)
    tau_top = tau_top.reshape(-1, 1)


    tau_hat = {
        "bottom": torch.Tensor(tau_bottom.transpose()),
        "top": torch.Tensor(tau_top.transpose()),
        "mean": torch.Tensor(tau_mean.transpose().tolist()),
    }

    return tau_hat

@ray.remote(num_gpus=0.2)
def compute_all_intervals_Kernel(trial, ds_train, ds_valid, ds_test):
    intervals_kernel = {}
    output_dir = project_path / f"trial-{trial:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "intervals_kernel.json"
    if file_path.exists():
        with file_path.open(mode="r") as fp:
            intervals_kernel = load_intervals(file_path=file_path)
    else:
        for k, v in GAMMAS.items():
            log_gamma = k
            gamma = v
            tau_hat_kernel = compute_intervals_Kernel(ds_train=ds_train,
                                                      ds_valid=ds_valid,
                                                      ds_test=ds_test, gamma=gamma)
            intervals_kernel.update({k: tau_hat_kernel})
        save_intervals(intervals=intervals_kernel, file_path=file_path)

    return intervals_kernel

@ray.remote(num_gpus=0.2)
def compute_all_intervals_Quince(trial, model_args, ds_train, ds_valid, ds_test):
    intervals_quince = {}
    output_dir = project_path / f"trial-{trial:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "intervals_q.json"
    if file_path.exists():
        with file_path.open(mode="r") as fp:
            intervals_quince = load_intervals(file_path=file_path)
    else:
        for k, v in GAMMAS.items():
            log_gamma = k
            gamma = v
            tau_hat_quince = compute_intervals_Quince(model_args=model_args, ds_train=ds_train,
                                                      ds_valid=ds_valid, ds_test=ds_test, gamma=gamma)
            intervals_quince.update({k: tau_hat_quince})
        save_intervals(intervals=intervals_quince, file_path=file_path)

    return intervals_quince


def compute_intervals_Quince_ensemble(model_args, ds_train, ds_valid, ds_test, gamma, n_ensemble):
    tau_means = []
    tau_uppers = []
    tau_lowers = []

    for n in tqdm(range(n_ensemble)):
        intervals_n = compute_intervals_Quince(model_args=model_args, ds_train=ds_train,
                                               ds_valid=ds_valid, ds_test=ds_test, gamma=gamma)
        tau_means.append(intervals_n["mean"])
        tau_lowers.append(intervals_n["bottom"])
        tau_uppers.append(intervals_n["top"])

    tau_mean = torch.cat(tau_means).mean(0)
    tau_std = torch.cat(tau_means).std(0)
    tau_top = torch.cat(tau_uppers).mean(0)
    tau_bottom = torch.cat(tau_lowers).mean(0)

    tau_top = torch.abs(
        tau_top + 2 * tau_std - tau_mean
    )
    tau_bottom = torch.abs(
        tau_bottom - 2 * tau_std - tau_mean
    )

    tau_hat = {
        "bottom": tau_bottom.unsqueeze(0),
        "top": tau_top.unsqueeze(0),
        "mean": tau_mean.unsqueeze(0),
    }

    return tau_hat


def update_sensitivity(results, intervals, tau_true, pi_true):
    n = len(tau_true)
    pehe = []
    error_rate = []
    defer_rate = []
    tau_hat = intervals["0.0"]
    tau_mean = tau_hat["mean"].mean(0)
    pi_hat = tau_mean > 0.0  # added =
    defer_rate.append(0.0)
    pehe.append(torch.sqrt(torch.square(tau_true - tau_mean).mean(0)).item())
    error_rate.append(1 - (pi_hat == pi_true).float().mean().item())
    deferred = []
    for k in intervals.keys():
        tau_hat = intervals[k]
        tau_mean = tau_hat["mean"].mean(0)
        tau_top = tau_hat["top"].mean(0)
        tau_bottom = tau_hat["bottom"].mean(0)
        defer = (tau_top >= 0) * (tau_bottom <= 0)
        new_deferrals = list(set(np.where(defer)[0]).difference(set(deferred)))
        rank = torch.min(
            torch.abs(torch.cat([tau_top.unsqueeze(0), tau_bottom.unsqueeze(0)])), dim=0
        )[0]
        if len(new_deferrals) > 0:
            for idx in np.argsort(-rank[new_deferrals]):
                deferred.append(new_deferrals[idx])
                defer_rate.append(len(deferred) / n)
                keep = torch.ones_like(tau_true, dtype=torch.bool)
                keep[deferred] = False
                pehe.append(
                    torch.sqrt(
                        torch.square(tau_true[keep] - tau_mean[keep]).mean(0)
                    ).item()
                )
                error_rate.append((pi_hat[keep] != pi_true[keep]).sum().item() / n)
        pi_hat = tau_bottom > 0.0
    new_deferrals = list(set(set(range(n))).difference(set(deferred)))
    if len(new_deferrals) > 0:
        for idx in np.argsort(rank[new_deferrals]):
            deferred.append(new_deferrals[idx])
            defer_rate.append(len(deferred) / n)
            keep = torch.ones_like(tau_true, dtype=torch.bool)
            keep[deferred] = False
            pehe.append(
                torch.sqrt(torch.square(tau_true[keep] - tau_mean[keep]).mean(0)).item()
            )
            error_rate.append((pi_hat[keep] != pi_true[keep]).sum().item() / n)
    results["sweep"]["sensitivity"]["pehe"].append(pehe)
    results["sweep"]["sensitivity"]["error_rate"].append(error_rate)
    results["sweep"]["sensitivity"]["defer_rate"].append(defer_rate)


def interpolate_values(deferral_rate, error_rate):
    means = []
    cis = []
    for i in range(len(deferral_rate)):
        means.append(error_rate[i, :].mean())
        se = stats.sem(error_rate[i, :])
        cis.append(se)
    return np.asarray(means), np.asarray(cis)


def plot_sweep(
        mode, sensitivity_quince=None, sensitivity_nn=None, sensitivity_rf=None,sensitivity_kernel=None, output_dir=""):

    data = {}

    if sensitivity_quince is not None:
        means_se_q, cis_se_q = interpolate_values(sensitivity_quince["defer_rate"], sensitivity_quince[mode])
        data["Quince"] = {
            "mean": np.clip(means_se_q, 1e-5, np.inf),
            "ci": cis_se_q,
            "color": "C0",
            "line_style": "--",
        }

    if sensitivity_nn is not None:
        means_se_nn, cis_se_nn = interpolate_values(sensitivity_nn["defer_rate"], sensitivity_nn[mode])
        data[r"$\widehat{\tau}^\pm$(NN, NN)"] = {
            "mean": np.clip(means_se_nn, 1e-5, np.inf),
            "ci": cis_se_nn,
            "color": "C1",
            "line_style": "--",
        }

    if sensitivity_rf is not None:
        means_rf, cis_rf = interpolate_values(sensitivity_rf["defer_rate"], sensitivity_rf[mode])
        data[r"$\widehat{\tau}^\pm$(RF, RF)"] = {
            "mean": np.clip(means_rf, 1e-5, np.inf),
            "ci": cis_rf,
            "color": "C2",
            "line_style": "--",
        }
    if sensitivity_kernel is not None:
        means_ker, cis_ker = interpolate_values(sensitivity_kernel["defer_rate"], sensitivity_kernel[mode])
        data["Sensitivity Kernel"] = {
            "mean": np.clip(means_ker, 1e-5, np.inf),
            "ci": cis_ker,
            "color": "C3",
            "line_style": "--",
        }
    deferral_rates = [i / len(means_rf) for i in range(len(means_rf))]
    fill_between(
        x=deferral_rates,
        y=data,
        x_label="Deferral Rate",
        y_label="Recommendation Error Rate"
        if mode == "error_rate"
        else r"Standardized $\sqrt{ \epsilon_{PEHE}}$",
        alpha=0.2,
        y_scale="log" if mode == "error_rate" else "linear",
        x_lim=[0, 1.0],
        y_lim=[1e-5, 2e-1] if mode == "error_rate" else None,
        x_pad=-20,
        y_pad=-45,
        legend_loc='lower left',
        file_path=Path(output_dir) / f"{mode}.pdf",
    )

def save_intervals(intervals, file_path):
    tmp_intervals = intervals
    for k, v in tmp_intervals.items():
        for k1, v2 in v.items():
            tmp_intervals[k][k1] = (v2.numpy()).tolist()
    with file_path.open(mode="w") as fp:
        json.dump(tmp_intervals, fp)

def load_intervals(file_path):
    with file_path.open(mode="r") as fp:
        intervals = json.load(fp)
    for k, v in intervals.items():
        for k1, v2 in v.items():
            intervals[k][k1] = torch.Tensor(v2)
    return intervals


if __name__ == "__main__":


    model_args_ihdp = {
        "dim_hidden": 200,
        "depth": 4,
        "negative_slope": 0.3,
        "layer_norm": False,
        "dropout_rate": 0.5,
        "learning_rate": 5e-4,
        "spectral_norm": False,
        "batch_size": 50,
        "max_epochs": 500,
        "verbose": True,
        "num_components": 5,
        "patience": 50,
        # "job_dir": project_path,
    }

    final_nn_args = {
        "dim_hidden": 200,
        "depth": 4,
        "negative_slope": 0.3,
        "layer_norm": False,
        "dropout_rate": 0.5,
        "learning_rate": 5e-4,
        "spectral_norm": False,
        "batch_size": 50,
        "max_epochs": 500,
        "verbose": True,
        "num_components": 5,
        "patience": 50,
    }

    num_trials = 434

    summary_quince = {
        "sweep": {"sensitivity": {"pehe": [], "error_rate": [], "defer_rate": [], }, },
    }

    summary_nn = {
        "sweep": {"sensitivity": {"pehe": [], "error_rate": [], "defer_rate": [], }, },
    }

    summary_rf = {
        "sweep": {"sensitivity": {"pehe": [], "error_rate": [], "defer_rate": [], }, },
    }
    summary_kernel = {
        "sweep": {"sensitivity": {"pehe": [], "error_rate": [], "defer_rate": [], }, },
    }
    results_list_nn = []
    results_list_rf = []
    results_list_quince = []
    results_list_kernel = []
    results_nn = []
    results_rf = []
    results_quince = []
    results_kernel = []
    tau_true_list = []

    # Computing the intervals for all trials in parallel
    for i in range(num_trials):

        # These trials have no overlap for some covariates and yield nonsensical results
        # TODO: fix this
        if i in [4, 18, 99, 127, 135, 349, 273]:
            continue

        # IHDP dataset
        ihdp_train_ds = IHDP(root=None, split="train", mode='mu', seed=i, hidden_confounding=True)
        ihdp_val_ds = IHDP(root=None, split="valid", mode='mu', seed=i, hidden_confounding=True)
        ihdp_test_ds = IHDP(root=None, split="test", mode='mu', seed=i, hidden_confounding=True)
        tau_true = torch.tensor(ihdp_test_ds.mu1 - ihdp_test_ds.mu0)
        tau_true_list.append(tau_true)

        results_list_nn.append(compute_all_intervals_NNBLearner.remote(trial=i, model_args=model_args_ihdp,
                                                                         final_nn_args=final_nn_args,
                                                                         ds_train=ihdp_train_ds,
                                                                         ds_valid=ihdp_val_ds,
                                                                         ds_test=ihdp_test_ds))

        results_list_rf.append(compute_all_intervals_BLearner.remote(trial=i, ds_train=ihdp_train_ds,
                                                                       ds_valid=ihdp_val_ds,
                                                                       ds_test=ihdp_test_ds))
        results_list_quince.append(
            compute_all_intervals_Quince.remote(trial=i, model_args=model_args_ihdp, ds_train=ihdp_train_ds,
                                                ds_valid=ihdp_val_ds, ds_test=ihdp_test_ds))
        results_list_kernel.append(compute_all_intervals_Kernel.remote(trial=i, ds_train=ihdp_train_ds,
                                                ds_valid=ihdp_val_ds, ds_test=ihdp_test_ds))

    results_nn = ray.get(results_list_nn)
    results_rf = ray.get(results_list_rf)
    results_quince = ray.get(results_list_quince)
    results_kernel = ray.get(results_list_kernel)

    for i in range(len(results_rf)):
        tau_true = tau_true_list[i]
        pi_true = tau_true > 0.0

        intervals_nn = results_nn[i]
        update_sensitivity(
            results=summary_nn,
            intervals=intervals_nn,
            tau_true=tau_true,
            pi_true=pi_true,
        )
        inetrvals_rf = results_rf[i]
        update_sensitivity(
            results=summary_rf,
            intervals=inetrvals_rf,
            tau_true=tau_true,
            pi_true=pi_true,
        )
        intervals_quince = results_quince[i]
        update_sensitivity(
            results=summary_quince,
            intervals=intervals_quince,
            tau_true=tau_true,
            pi_true=pi_true,
        )
        intervals_kernel = results_kernel[i]
        update_sensitivity(
            results=summary_kernel,
            intervals=intervals_kernel,
            tau_true=tau_true,
            pi_true=pi_true,
        )

    sensitivity_quince = summary_quince["sweep"]["sensitivity"]
    for k in sensitivity_quince.keys():
        sensitivity_quince[k] = np.nan_to_num(np.asarray(sensitivity_quince[k]).transpose())

    sensitivity_nn = summary_nn["sweep"]["sensitivity"]
    for k in sensitivity_nn.keys():
        sensitivity_nn[k] = np.nan_to_num(
            np.asarray(sensitivity_nn[k]).transpose()
        )

    sensitivity_rf = summary_rf["sweep"]["sensitivity"]
    for k in sensitivity_rf.keys():
        sensitivity_rf[k] = np.nan_to_num(
            np.asarray(sensitivity_rf[k]).transpose()
        )

    sensitivity_kernel = summary_kernel["sweep"]["sensitivity"]
    for k in sensitivity_kernel.keys():
        sensitivity_kernel[k] = np.nan_to_num(
            np.asarray(sensitivity_kernel[k]).transpose()
        )

    plot_sweep(
        sensitivity_quince=sensitivity_quince,
        sensitivity_nn=sensitivity_nn,
        sensitivity_rf=sensitivity_rf,
        sensitivity_kernel=sensitivity_kernel,
        mode="error_rate",
        output_dir=project_path
    )

