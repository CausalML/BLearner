import numpy as np

from torch.utils import data


class Synthetic(data.Dataset):
    def __init__(
        self,
        num_examples,
        gamma_star,
        mode,
        p_u="bernoulli",
        theta=4.0,
        beta=0.75,
        sigma_y=1.0,
        domain=2,
        seed=1331,
        split=None,
    ):
        super(Synthetic, self).__init__()
        rng = np.random.RandomState(seed=seed)
        self.num_examples = num_examples
        self.dim_input = 1
        self.dim_treatment = 1
        self.dim_output = 1
        self.x = rng.uniform(-domain, domain, size=(num_examples, 1)).astype("float32")

        if p_u == "bernoulli":
            self.u = rng.binomial(1, 0.5, size=(num_examples, 1)).astype("float32")
        elif p_u == "uniform":
            self.u = rng.uniform(size=(num_examples, 1)).astype("float32")
        elif p_u == "beta_bi":
            self.u = rng.beta(0.5, 0.5, size=(num_examples, 1)).astype("float32")
        elif p_u == "beta_uni":
            self.u = rng.beta(2, 5, size=(num_examples, 1)).astype("float32")
        else:
            raise NotImplementedError(f"{p_u} is not a supported distribution")

        self.pi = (
            complete_propensity(x=self.x, u=self.u, gamma=gamma_star, beta=beta)
            .astype("float32")
            .ravel()
        )
        self.t = rng.binomial(1, self.pi).astype("float32")
        eps = (sigma_y * rng.normal(size=self.t.shape)).astype("float32")
        self.mu0 = (
            f_mu(x=self.x, t=0.0, u=self.u, theta=theta).astype("float32").ravel()
        )
        self.mu1 = (
            f_mu(x=self.x, t=1.0, u=self.u, theta=theta).astype("float32").ravel()
        )
        self.y0 = self.mu0 + eps
        self.y1 = self.mu1 + eps
        self.y = self.t * self.y1 + (1 - self.t) * self.y0
        self.tau = self.mu1 - self.mu0
        if mode == "pi":
            self.inputs = self.x
            self.targets = self.t
        elif mode == "mu":
            self.inputs = np.hstack([self.x, np.expand_dims(self.t, -1)])
            self.targets = self.y
        else:
            raise NotImplementedError(
                f"{mode} not supported. Choose from 'pi'  for propensity models or 'mu' for expected outcome models"
            )
        self.y_mean = np.array([0.0], dtype="float32")
        self.y_std = np.array([1.0], dtype="float32")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index : index + 1]

    def tau_fn(self, x):
        return f_mu(x=x, t=1.0, u=1.0, theta=0.0) - f_mu(x=x, t=0.0, u=1.0, theta=0.0)


def complete_propensity(x, u, gamma, beta=0.75):
    logit = beta * x + 0.5
    nominal = (1 + np.exp(-logit)) ** -1
    alpha = alpha_fn(nominal, gamma)
    beta = beta_fn(nominal, gamma)
    return (u / alpha) + ((1 - u) / beta)


def f_mu(x, t, u, theta=4.0):
    mu = (
        (2 * t - 1) * x
        + (2.0 * t - 1)
        - 2 * np.sin((4 * t - 2) * x)
        - (theta * u - 2) * (1 + 0.5 * x)
    )
    return mu


def alpha_fn(pi, lambda_):
    return (pi * lambda_) ** -1 + 1.0 - lambda_**-1


def beta_fn(pi, lambda_):
    return lambda_ * (pi) ** -1 + 1.0 - lambda_