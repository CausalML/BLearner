import math

import torch
from torch import nn, distributions


class Normal(torch.nn.Module):
    def __init__(
        self,
        dim_input,
        dim_output,
    ):
        super(Normal, self).__init__()
        self.mu = torch.nn.Linear(
            in_features=dim_input,
            out_features=dim_output,
            bias=True,
        )
        self.log_var = torch.nn.Parameter(-0.8 * torch.ones(dim_output))

    def forward(self, inputs):
        var = torch.nn.functional.softplus(self.log_var)
        return torch.distributions.MultivariateNormal(
            loc=self.mu(inputs), covariance_matrix=torch.diag(var + 1e-7)
        )


class MixtureSameFamily(torch.distributions.MixtureSameFamily):
    def log_prob(self, inputs):
        loss = torch.exp(self.component_distribution.log_prob(inputs.unsqueeze(1)))
        loss = torch.sum(loss * self.mixture_distribution.probs, dim=1)
        return torch.log(loss + 1e-7)

    def cdf(self, x):
        x = self._pad(x)
        cdf_x = self.component_distribution.base_dist.cdf(x).squeeze(-1)
        mix_prob = self.mixture_distribution.probs
        return torch.sum(cdf_x * mix_prob, dim=-1)


class GMM(nn.Module):
    def __init__(
        self,
        num_components,
        dim_input,
        dim_output,
    ):
        super(GMM, self).__init__()
        self.mu = nn.Linear(
            in_features=dim_input,
            out_features=num_components * dim_output,
            bias=True,
        )
        sigma = nn.Linear(
            in_features=dim_input,
            out_features=num_components * dim_output,
            bias=True,
        )
        self.pi = nn.Linear(
            in_features=dim_input,
            out_features=num_components,
            bias=True,
        )
        self.sigma = nn.Sequential(sigma, nn.Softplus())
        self.num_components = num_components
        self.dim_output = dim_output

    def forward(self, inputs):
        loc = self.mu(inputs).reshape(-1, self.num_components, self.dim_output)
        scale = (
            self.sigma(inputs).reshape(-1, self.num_components, self.dim_output) + 1e-7
        )
        return MixtureSameFamily(
            mixture_distribution=distributions.Categorical(logits=self.pi(inputs)),
            component_distribution=distributions.Independent(
                base_distribution=distributions.Normal(
                    loc=loc,
                    scale=scale,
                ),
                reinterpreted_batch_ndims=1,
            ),
        )


class GroupLinear(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        num_groups: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super(GroupLinear, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_groups = num_groups
        self.weight = nn.parameter.Parameter(
            torch.empty((self.num_groups, dim_input, dim_output), **factory_kwargs)
        )
        if bias:
            self.bias = nn.parameter.Parameter(
                torch.empty((self.num_groups, dim_output), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.dim_input)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, inputs) -> torch.Tensor:
        x, g = inputs
        w = torch.matmul(
            g,
            self.weight.view(self.num_groups, self.dim_input * self.dim_output),
        ).reshape(-1, self.dim_input, self.dim_output)
        return torch.bmm(x.unsqueeze(1), w).squeeze(1) + torch.matmul(g, self.bias)

    def extra_repr(self) -> str:
        return f"num_groups={self.num_groups}, dim_input={self.dim_input}, dim_output={self.dim_output}, bias={self.bias}"


class GroupGMM(nn.Module):
    def __init__(self, num_components, dim_input, dim_output, groups):
        super(GroupGMM, self).__init__()
        self.mu = GroupLinear(
            dim_input=dim_input,
            dim_output=dim_output * num_components,
            num_groups=groups,
            bias=True,
        )
        self.sigma = nn.Sequential(
            GroupLinear(
                dim_input=dim_input,
                dim_output=dim_output * num_components,
                num_groups=groups,
                bias=True,
            ),
            nn.Softplus(),
        )
        self.pi = GroupLinear(
            dim_input=dim_input,
            dim_output=num_components,
            num_groups=groups,
            bias=True,
        )
        self.dim_output = dim_output
        self.num_components = num_components

    def forward(self, inputs):

        logits = self.pi(inputs)
        loc = self.mu(inputs)
        scale = self.sigma(inputs) + 1e-7

        component_distribution = distributions.Independent(
            distributions.Normal(
                loc=loc.reshape(-1, self.num_components, self.dim_output),
                scale=scale.reshape(-1, self.num_components, self.dim_output),
            ),
            reinterpreted_batch_ndims=1,
        )
        return MixtureSameFamily(
            mixture_distribution=distributions.Categorical(logits=logits),
            component_distribution=component_distribution,
        )


class Categorical(torch.nn.Module):
    def __init__(
        self,
        dim_input,
        dim_output,
    ):
        super(Categorical, self).__init__()
        self.logits = torch.nn.Linear(
            in_features=dim_input,
            out_features=dim_output,
            bias=True,
        )
        self.distribution = (
            torch.distributions.Bernoulli
            if dim_output == 1
            else torch.distributions.Categorical
        )

    def forward(self, inputs):
        logits = self.logits(inputs)
        return self.distribution(logits=logits)