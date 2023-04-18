import math
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl

from ..modules import dense, variational
from sklearn.model_selection import train_test_split

_eps = 1e-7


def _alpha_func(p, _lambda):
    return 1 / (_lambda * p) + 1 - 1 / (_lambda)


def _beta_func(p, _lambda):
    return _lambda / (p) + 1 - _lambda


class Quince(pl.LightningModule):
    def __init__(
        self,
        num_components=8,
        dim_hidden=100,
        depth=3,
        negative_slope=-1,
        layer_norm=False,
        dropout_rate=0.05,
        spectral_norm=False,
        max_epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        patience=10,
        verbose=False,
        job_dir="/tmp/quince",
        feature_extractor=None,
        propensity_dist=None,
        outcomes_density=None,
    ) -> None:
        super().__init__()
        self.num_components = num_components
        self.job_dir = Path(job_dir)
        self.job_dir.mkdir(exist_ok=True, parents=True)
        # Architecture
        self.dim_hidden = dim_hidden
        self.depth = depth
        self.negative_slope = negative_slope
        self.layer_norm = layer_norm
        self.dropout_rate = dropout_rate
        self.spectral_norm = spectral_norm
        # Optimization
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        # Logging
        self.verbose = verbose
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        if propensity_dist is not None:
            self.propensity_dist = propensity_dist
        if outcomes_density is not None:
            self.outcomes_density = outcomes_density

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def training_step(self, batch, batch_idx):
        self.feature_extractor.train()
        self.propensity_dist.train()
        self.outcomes_density.train()
        x, a, y = batch
        z = self.feature_extractor(x)
        a_dist = self.propensity_dist(z)
        y_dist = self.outcomes_density(
            [
                z,
                torch.nn.functional.one_hot(a, self.num_actions).float(),
            ]
        )
        a_loss = -a_dist.log_prob(a).mean()
        y_loss = -y_dist.log_prob(y).mean()
        loss = a_loss + y_loss
        self.log("train_loss", loss)
        self.log("a_loss", a_loss)
        self.log("y_loss", y_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        self.feature_extractor.eval()
        self.propensity_dist.eval()
        self.outcomes_density.eval()
        x, a, y = batch
        z = self.feature_extractor(x)
        a_dist = self.propensity_dist(z)
        y_dist = self.outcomes_density(
            [
                z,
                torch.nn.functional.one_hot(a, self.num_actions).float(),
            ]
        )
        val_loss = -a_dist.log_prob(a).mean() - y_dist.log_prob(y).mean()
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1 / (2 * (1 - self.dropout_rate) * len(self.train_inputs)),
        )

    def train_dataloader(self):
        len_ds = len(self.train_inputs)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                self.train_inputs, self.train_actions, self.train_outcomes
            ),
            batch_size=self.batch_size if self.batch_size < len_ds else len_ds,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        len_ds = len(self.valid_inputs)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                self.valid_inputs, self.valid_actions, self.valid_outcomes
            ),
            batch_size=self.batch_size if self.batch_size < len_ds else len_ds,
            shuffle=False,
            drop_last=False,
        )

    def fit(self, x, a, y, x_val=None, a_val=None, y_val=None):
        if x_val is None or a_val is None or y_val is None:
            x_train, x_val, a_train, a_val, y_train, y_val = train_test_split(x, a, y, test_size=0.1, random_state=42)
        else:
            x_train, a_train, y_train = x, a, y
        self.train_inputs = torch.tensor(x_train).float()
        self.train_actions = torch.tensor(a_train).long()
        self.train_outcomes = torch.tensor(y_train).float()

        self.valid_inputs = torch.tensor(x_val).float()
        self.valid_actions = torch.tensor(a_val).long()
        self.valid_outcomes = torch.tensor(y_val).float()

        self.num_actions = int(a_train.max() + 1)
        if self.train_outcomes.ndim == 1:
            self.train_outcomes = self.train_outcomes.unsqueeze(-1)
        if self.valid_outcomes.ndim == 1:
            self.valid_outcomes = self.valid_outcomes.unsqueeze(-1)

        self.dim_outcomes = self.train_outcomes.shape[-1]
        self.feature_extractor = dense.DenseFeatureExtractor(
            dim_input=x_train.shape[-1],
            dim_hidden=self.dim_hidden,
            depth=self.depth,
            negative_slope=self.negative_slope,
            layer_norm=self.layer_norm,
            dropout_rate=self.dropout_rate,
            spectral_norm=self.spectral_norm,
            activate_output=True,
            architecture="resnet",
        )
        self.propensity_dist = variational.Categorical(
            dim_input=self.feature_extractor.dim_output,
            dim_output=self.num_actions,
        )
        self.outcomes_density = variational.GroupGMM(
            num_components=self.num_components,
            dim_input=self.feature_extractor.dim_output,
            dim_output=self.dim_outcomes,
            groups=self.num_actions,
        )
        logger = pl.loggers.TensorBoardLogger(self.job_dir) if self.verbose else None
        log_every_n_steps = min(
            int(math.ceil(self.train_inputs.shape[0] / self.batch_size)), 50
        )
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else None,
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            callbacks=[
                pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1),
                pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", mode="min", patience=self.patience)
            ],
            enable_progress_bar=self.verbose,
            enable_model_summary=self.verbose,
            default_root_dir=self.job_dir,
        )
        trainer.fit(self)

    def preprocess_for_prediction(self):
        if torch.cuda.is_available():
            self.feature_extractor.cuda()
            self.propensity_dist.cuda()
            self.outcomes_density.cuda()
        self.feature_extractor.eval()
        self.propensity_dist.eval()
        self.outcomes_density.eval()

    def process_prediction(self, y):
        if self.dim_outcomes == 1:
            return y.squeeze(-1).to("cpu").numpy()
        return y.to("cpu").numpy()

    def predict_mu(self, x, a):
        self.preprocess_for_prediction()
        if isinstance(a, int):
            a = torch.nn.functional.one_hot(
                a * torch.ones(size=torch.Size([len(x)]), dtype=torch.long),
                self.num_actions,
            )
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(x).float(), torch.tensor(a).float()
            ),
            batch_size=2 * self.batch_size,
            shuffle=False,
        )
        with torch.no_grad():
            means = []
            for x, a in dl:
                if torch.cuda.is_available():
                    x = x.cuda()
                    a = a.cuda()
                z = self.feature_extractor(x)
                means.append(self.outcomes_density([z, a]).mean)
            means = torch.cat(means, dim=0)
        return self.process_prediction(means)

    def predict_pi(self, x):
        self.preprocess_for_prediction()
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(x).float()),
            batch_size=2 * self.batch_size,
            shuffle=False,
        )
        with torch.no_grad():
            means = []
            for batch in dl:
                x = batch[0]
                if torch.cuda.is_available():
                    x = x.cuda()
                z = self.feature_extractor(x)
                means.append(self.propensity_dist(z).probs)
            means = torch.cat(means, dim=0)
        return self.process_prediction(means)

    def predict_quantile(self, x, a, q, num_samples=1000):
        self.preprocess_for_prediction()
        if isinstance(a, int):
            a = torch.nn.functional.one_hot(
                a * torch.ones(size=torch.Size([len(x)]), dtype=torch.long),
                self.num_actions,
            )
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(x).float(), torch.tensor(a).float()
            ),
            batch_size=2 * self.batch_size,
            shuffle=False,
        )
        with torch.no_grad():
            quantile = []
            for x, a in dl:
                if torch.cuda.is_available():
                    x = x.cuda()
                    a = a.cuda()
                z = self.feature_extractor(x)
                y_samples = self.outcomes_density([z, a]).sample(
                    torch.Size([num_samples])
                )
                quantile.append(torch.quantile(y_samples, q, dim=0))
            quantile = torch.cat(quantile, dim=0)
        return self.process_prediction(quantile)

    def predict_cvar(self, x, a, q, num_samples=1000):
        self.preprocess_for_prediction()
        if isinstance(a, int):
            a = torch.nn.functional.one_hot(
                a * torch.ones(size=torch.Size([len(x)]), dtype=torch.long),
                self.num_actions,
            )
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(x).float(), torch.tensor(a).float()
            ),
            batch_size=2 * self.batch_size,
            shuffle=False,
        )
        with torch.no_grad():
            lower, upper = [], []
            for x, a in dl:
                if torch.cuda.is_available():
                    x = x.cuda()
                    a = a.cuda()
                z = self.feature_extractor(x)
                y_samples = self.outcomes_density([z, a]).sample(
                    torch.Size([num_samples])
                )
                if q >= 0.5:
                    condition_upper = torch.quantile(y_samples, q=q, dim=0, keepdims=True)
                    condition_lower = torch.quantile(y_samples, q=1-q, dim=0, keepdims=True)
                else:
                    condition_upper = torch.quantile(y_samples, q=1-q, dim=0, keepdims=True)
                    condition_lower = torch.quantile(y_samples, q=q, dim=0, keepdims=True)
                mask_lower = (y_samples <= condition_lower).float()
                lower.append(
                    (y_samples * mask_lower).sum(0) / (mask_lower.sum(0) + 1e-7)
                )
                mask_upper = (y_samples > condition_upper).float()
                upper.append(
                    (y_samples * mask_upper).sum(0) / (mask_upper.sum(0) + 1e-7)
                )
            lower = torch.cat(lower, dim=0)
            upper = torch.cat(upper, dim=0)
        return self.process_prediction(lower), self.process_prediction(upper)

    def predict_rho(self, x, a, q, num_samples=1000):
        self.preprocess_for_prediction()
        if isinstance(a, int):
            a = torch.nn.functional.one_hot(
                a * torch.ones(size=torch.Size([len(x)]), dtype=torch.long),
                self.num_actions,
            )
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(x).float(), torch.tensor(a).float()
            ),
            batch_size=2 * self.batch_size,
            shuffle=False,
        )
        _lambda = q/(1-q)
        with torch.no_grad():
            lower, upper = [], []
            for x, a in dl:
                if torch.cuda.is_available():
                    x = x.cuda()
                    a = a.cuda()
                z = self.feature_extractor(x)
                y_samples = self.outcomes_density([z, a]).sample(
                    torch.Size([num_samples])
                )
                quantile_plus_batch = torch.quantile(y_samples, q, dim=0)
                quantile_minus_batch = torch.quantile(y_samples, 1-q, dim=0)
                R_minus = (1 / _lambda) * y_samples + (1 - (1 / _lambda)) * (
                quantile_minus_batch + (1 / (1 - q)) * (torch.minimum(y_samples, quantile_minus_batch) - quantile_minus_batch)
                )
                lower.append(torch.mean(R_minus, dim=0))
                R_plus = (1 / _lambda) * y_samples + (1 - (1 / _lambda)) * (
                quantile_plus_batch + (1 / (1 - q)) * (torch.maximum(y_samples, quantile_plus_batch) - quantile_plus_batch)
                )
                upper.append(torch.mean(R_plus, dim=0))
            lower = torch.cat(lower, dim=0)
            upper = torch.cat(upper, dim=0)
        return self.process_prediction(lower), self.process_prediction(upper)

    def predict_mu_bounds(
        self,
        x,
        a,
        _lambda,
        num_samples=100,
        batch_size=None,
    ):
        self.preprocess_for_prediction()
        if isinstance(a, int):
            a = torch.nn.functional.one_hot(
                a * torch.ones(size=torch.Size([len(x)]), dtype=torch.long),
                self.num_actions,
            )
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(x).float(), torch.tensor(a).float()
            ),
            batch_size=2 * self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            drop_last=False,
        )
        _lambda = torch.tensor([_lambda]).to(self._device) + _eps
        # predict and sample
        lower, upper = [], []
        for x, a in dl:
            with torch.no_grad():
                if torch.cuda.is_available():
                    x = x.cuda()
                    a = a.cuda()
                z = self.feature_extractor(x)
                y_density = self.outcomes_density([z, a])
                y_samples = y_density.sample(
                    torch.Size([num_samples])
                ).to(self._device)  # [num_samples, batch_size, dy]
                mu = y_density.mean.unsqueeze(0).to(self._device)  # [1, batch_size, dy]
                a_distribution = self.propensity_dist(z)
                pi = (
                    torch.exp(a_distribution.log_prob(torch.argmax(a, -1)))
                    .unsqueeze(0)
                    .unsqueeze(-1)
                )  # [1, batch_size, 1]
                pi = torch.clip(pi, _eps, 1 - _eps)
                # get alpha prime
                alpha = _alpha_func(pi.to(self._device), _lambda.to(self._device))
                beta = _beta_func(pi.to(self._device), _lambda.to(self._device))
                alpha_prime = alpha / (beta - alpha)
                # sweep over upper bounds
                r = y_samples - mu  # [num_samples, batch_size, dy]
                d = y_samples - y_samples.unsqueeze(
                    1
                )  # [num_samples, num_samples, batch_size, dy]
                h_u = torch.heaviside(
                    d.to(self._device), torch.tensor([1.0], device=self._device)
                ).to(self._device)  # [num_samples, num_samples, batch_size, dy]
                numer_upper = (h_u * r.unsqueeze(0).to(self._device)).mean(
                    1
                ).to(self._device)  # [num_samples, batch_size, dy]
                denom_upper = (
                    h_u.mean(1) + alpha_prime + _eps
                ).to(self._device)  # [num_samples, batch_size, dy]
                upper_batch = (
                    mu + numer_upper / denom_upper
                )  # [num_samples, batch_size, dy]
                upper_batch = upper_batch.max(0)[0]  # [batch_size, dy]
                upper.append(upper_batch)
                # sweep over lower bounds
                h_l = torch.heaviside(
                    -d, torch.tensor([1.0], device=self._device)
                )  # [num_samples, num_samples, batch_size, dy]
                numer_lower = (h_l * r.unsqueeze(0)).mean(
                    1
                )  # [num_samples, batch_size, dy]
                denom_lower = (
                    h_l.mean(1) + alpha_prime + _eps
                )  # [num_samples, batch_size, dy]
                lower_batch = (
                    mu + numer_lower / denom_lower
                )  # [num_samples, batch_size, dy]
                lower_batch = lower_batch.min(0)[0]  # [batch_size, dy]
                lower.append(lower_batch)
        # post process
        upper = torch.cat(upper, dim=0)
        lower = torch.cat(lower, dim=0)
        return self.process_prediction(lower), self.process_prediction(upper)