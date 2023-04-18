import math
import numpy as np
from pathlib import Path
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split

import torch
import pytorch_lightning as pl

from ..modules import variational
from ..modules import dense



#######################
# Quantile Regressors #
#######################
class KernelQuantileRegressor:

    def __init__(self, kernel, tau):
        self.kernel = kernel
        self.tau = tau

    def fit(self, X, Y):
        self.sorted_Y_idx = np.argsort(Y)
        self.sorted_Y = Y[self.sorted_Y_idx]
        self.kernel.fit(X[self.sorted_Y_idx], Y[self.sorted_Y_idx])
        return self

    def predict(self, X):
        preds = np.empty(X.shape[0])
        sorted_weights = self.kernel.predict(X)
        for i, x in enumerate(X):
            quantile_idx = np.where((np.cumsum(sorted_weights[i]) >= self.tau) == True)[0][0]
            preds[i] = self.sorted_Y[quantile_idx]
        return preds


##################
# Kernel Methods #
##################
class RFKernel:

    def __init__(self, rf):
        self.rf = rf

    def fit(self, X, Y):
        self.rf.fit(X, Y)
        self.train_leaf_map = self.rf.apply(X)

    def predict(self, X):
        weights = np.empty((X.shape[0], self.train_leaf_map.shape[0]))
        leaf_map = self.rf.apply(X)
        for i, x in enumerate(X):
            P = (self.train_leaf_map == leaf_map[[i]])
            weights[i] = (1. * P / P.sum(axis=0)).mean(axis=1)
        return weights


class RBFKernel:
    def __init__(self, scale=1):
        self.kernel = RBF(length_scale=scale)

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y
        return self

    def predict(self, X):
        weights = self.kernel(X, self.X_train)
        # Normalize weights
        norm_weights = weights / weights.sum(axis=1).reshape(-1, 1)
        return norm_weights @ self.Y_train


############################
# Superquantile regressors #
############################
class KernelSuperquantileRegressor:

    def __init__(self, kernel, tau, tail='left'):
        self.kernel = kernel
        self.tau = tau
        if tail not in ["left", "right"]:
            raise ValueError(
                f"The 'tail' parameter can only take values in ['left', 'right']. Got '{tail}' instead.")
        self.tail = tail

    def fit(self, X, Y):
        self.sorted_Y_idx = np.argsort(Y)
        self.sorted_Y = Y[self.sorted_Y_idx]
        self.kernel.fit(X[self.sorted_Y_idx], Y[self.sorted_Y_idx])
        return self

    def predict(self, X):
        preds = np.empty(X.shape[0])
        sorted_weights = self.kernel.predict(X)
        for i, x in enumerate(X):
            if self.tail == "right":
                idx_tail = np.where((np.cumsum(sorted_weights[i]) >= self.tau) == True)[0]
                preds[i] = np.sum(self.sorted_Y[idx_tail] * sorted_weights[i][idx_tail]) / (1 - self.tau)
            else:
                idx_tail = np.where((np.cumsum(sorted_weights[i]) <= self.tau) == True)[0]
                preds[i] = np.sum(self.sorted_Y[idx_tail] * sorted_weights[i][idx_tail]) / self.tau
        return preds


#########################
# Neural Net regressors #
#########################
class _Net(pl.LightningModule):
    def __init__(
        self,
        tau=0.5,
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
        model=None,
        feature_extractor=None,
    ) -> None:
        super(_Net, self).__init__()
        self.tau = tau
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
        if model is not None:
            self.model = model

    def training_step(self, batch, batch_idx):
        self.model.train()
        output = self.model(batch[0])
        loss = -output.log_prob(batch[1]).mean()
        tensorboard_logs = {"train_loss": loss}
        self.log("train_loss", loss)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        self.model.eval()
        x, y = batch
        y_dist = self.model(x)
        val_loss = - y_dist.log_prob(y).mean()
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1 / (2 * (1 - self.dropout_rate) * len(self.train_inputs)),
        )

    def train_dataloader(self):
        len_ds = len(self.train_inputs)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.train_inputs, self.train_targets),
            batch_size=self.batch_size if self.batch_size < len_ds else len_ds,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        len_ds = len(self.valid_inputs)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                self.valid_inputs, self.valid_targets
            ),
            batch_size=self.batch_size if self.batch_size < len_ds else len_ds,
            shuffle=False,
            drop_last=False,
        )

    def fit(self, x, y, x_val=None, y_val=None):
        if x_val is None or y_val is None:
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42)
        else:
            x_train, y_train = x, y
        self.train_inputs = torch.tensor(x_train).float()
        self.train_targets = torch.tensor(y_train).float()

        self.valid_inputs = torch.tensor(x_val).float()
        self.valid_targets = torch.tensor(y_val).float()

        if self.train_targets.ndim == 1:
            self.train_targets = self.train_targets.unsqueeze(-1)
        if self.valid_targets.ndim == 1:
            self.valid_targets = self.valid_targets.unsqueeze(-1)
        self.dim_output = self.train_targets.shape[-1]
        feature_extractor = dense.DenseFeatureExtractor(
            dim_input=x.shape[-1],
            dim_hidden=self.dim_hidden,
            depth=self.depth,
            negative_slope=self.negative_slope,
            layer_norm=self.layer_norm,
            dropout_rate=self.dropout_rate,
            spectral_norm=self.spectral_norm,
            activate_output=True,
            architecture="resnet",
        )
        if isinstance(self, MuNet):
            output_dist = variational.GMM(
                num_components=self.num_components, ##check this
                dim_input=feature_extractor.dim_output,
                dim_output=self.dim_output,
            )
        elif isinstance(self, PropensityNet):
            output_dist = variational.Categorical(
                dim_input=feature_extractor.dim_output,
                dim_output=self.dim_output,
            )
        elif isinstance(self, QuantileNet):
            output_dist = variational.GMM(
                num_components=8,
                dim_input=feature_extractor.dim_output,
                dim_output=self.dim_output,
            )
        elif isinstance(self, CVaRNet):
            output_dist = variational.GMM(
                num_components=8,
                dim_input=feature_extractor.dim_output,
                dim_output=self.dim_output,
            )
        self.model = torch.nn.Sequential(
            feature_extractor,
            output_dist,
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
        self.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            model=self.model,
        )
        Path(trainer.checkpoint_callback.best_model_path).unlink()
       
    def preprocess_for_prediction(self):
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def process_prediction(self, y):
        if self.dim_output == 1:
            return y.squeeze(-1).to("cpu").numpy()
        return y.to("cpu").numpy()


class MuNet(_Net):
    def predict(self, x):
        self.preprocess_for_prediction()
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(x).float()),
            batch_size=2 * self.batch_size,
            shuffle=False,
        )
        with torch.no_grad():
            means = []
            for batch in dl:
                batch = batch[0].to("cuda" if torch.cuda.is_available() else "cpu")
                means.append(self.model(batch).mean)
            means = torch.cat(means, dim=0)
        return self.process_prediction(means)


class PropensityNet(_Net):
    def predict_proba(self, x):
        self.preprocess_for_prediction()
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(x).float()),
            batch_size=2 * self.batch_size,
            shuffle=False,
        )
        with torch.no_grad():
            means = []
            for batch in dl:
                batch = batch[0].to("cuda" if torch.cuda.is_available() else "cpu")
                means.append(self.model(batch).mean)
            means = torch.cat(means, dim=0)
        probs1 = self.process_prediction(means).reshape(-1, 1)
        return np.hstack((1-probs1, probs1))

    def predict(self, x):
        probs = self.predict_proba(x)
        return (probs[:, 1] > 0.5).astype("int")


class QuantileNet(_Net):
    def predict(self, x):
        self.preprocess_for_prediction()
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(x).float()),
            batch_size=2 * self.batch_size,
            shuffle=False,
        )
        with torch.no_grad():
            quantile = []
            for batch in dl:
                batch = batch[0].to("cuda" if torch.cuda.is_available() else "cpu")
                y_samples = self.model(batch).sample(torch.Size([1000]))
                quantile.append(torch.quantile(y_samples, self.tau, dim=0))
            quantile = torch.cat(quantile, dim=0)
        return self.process_prediction(quantile)


class CVaRNet(_Net):
    def __init__(
        self,
        tau=0.5,
        tail="right",
        num_components=8,
        dim_hidden=100,
        depth=3,
        negative_slope=1,
        layer_norm=False,
        dropout_rate=0.05,
        spectral_norm=False,
        max_epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        verbose=False,
        job_dir="/tmp/quince",
        model=None):
        self.tail = tail
        super().__init__(
            tau=tau,
            num_components=num_components,
            dim_hidden=dim_hidden,
            depth=depth,
            negative_slope=negative_slope,
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
            spectral_norm=spectral_norm,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=verbose,
            job_dir=job_dir,
            model=model)
    def predict(self, x):
        self.preprocess_for_prediction()
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(x).float()),
            batch_size=2 * self.batch_size,
            shuffle=False,
        )
        with torch.no_grad():
            cvar = []
            for batch in dl:
                batch = batch[0].to("cuda" if torch.cuda.is_available() else "cpu")
                y_samples = self.model(batch).sample(torch.Size([1000]))
                quantiles = torch.quantile(y_samples, self.tau, dim=0, keepdims=True)
                if self.tail == "right":
                    mask = (y_samples >= quantiles).float()
                elif self.tail == "left":
                    mask = (y_samples <= quantiles).float()
                else:
                    raise ValueError("'tail' parameter can only be 'left' or 'right'.")
                cvar.append(
                    (y_samples * mask).sum(0) / (mask.sum(0) + 1e-7)
                )
            cvar = torch.cat(cvar, dim=0)
        return self.process_prediction(cvar)