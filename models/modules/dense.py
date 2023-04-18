import torch
from torch import nn


class ConsistentDropout(torch.nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(ConsistentDropout, self).__init__()
        self.q = 1 - p
        self.inplace = inplace

    def forward(self, x, seed=None):
        mask = torch.distributions.Bernoulli(probs=self.q).sample(
            torch.Size([1]) + x.shape[1:]
        ).to(x.device) / (self.q)
        return x * mask


class DenseActivation(nn.Module):
    def __init__(
        self,
        dim_input,
        negative_slope,
        dropout_rate,
        layer_norm,
        consistent_dropout=False,
    ):
        super(DenseActivation, self).__init__()
        self.op = nn.Sequential(
            nn.LayerNorm(dim_input) if layer_norm else nn.Identity(),
            nn.LeakyReLU(negative_slope=negative_slope)
            if negative_slope >= 0.0
            else nn.GELU(),
            ConsistentDropout(p=dropout_rate)
            if consistent_dropout
            else nn.Dropout(p=dropout_rate),
        )

    def forward(self, inputs):
        return self.op(inputs)


class DensePreactivation(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_output,
        bias,
        negative_slope,
        dropout_rate,
        layer_norm,
        spectral_norm=False,
        consistent_dropout=False,
    ):
        super(DensePreactivation, self).__init__()
        self.op = nn.Sequential(
            DenseActivation(
                dim_input=dim_input,
                negative_slope=negative_slope,
                dropout_rate=dropout_rate,
                layer_norm=layer_norm,
                consistent_dropout=consistent_dropout,
            )
        )
        linear = nn.Linear(in_features=dim_input, out_features=dim_output, bias=bias)
        self.op.add_module(
            "linear",
            nn.utils.spectral_norm(linear) if spectral_norm else linear,
        )

    def forward(self, inputs):
        return self.op(inputs)


class DenseResidual(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_output,
        bias,
        negative_slope,
        dropout_rate,
        layer_norm,
        spectral_norm=False,
        consistent_dropout=False,
    ):
        super(DenseResidual, self).__init__()
        if dim_input != dim_output:
            self.shortcut = nn.Sequential(
                ConsistentDropout(p=dropout_rate)
                if consistent_dropout
                else nn.Dropout(p=dropout_rate)
            )
            linear = nn.Linear(
                in_features=dim_input, out_features=dim_output, bias=bias
            )
            self.shortcut.add_module(
                "linear",
                nn.utils.spectral_norm(linear) if spectral_norm else linear,
            )
        else:
            self.shortcut = nn.Identity()

        self.op = DensePreactivation(
            dim_input=dim_input,
            dim_output=dim_output,
            bias=bias,
            negative_slope=negative_slope,
            dropout_rate=dropout_rate,
            layer_norm=layer_norm,
            spectral_norm=spectral_norm,
            consistent_dropout=consistent_dropout,
        )

    def forward(self, inputs):
        return self.op(inputs) + self.shortcut(inputs)


MODULES = {"basic": DensePreactivation, "resnet": DenseResidual}


class DenseLinear(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_output,
        layer_norm,
        spectral_norm=False,
    ):
        super(DenseLinear, self).__init__()
        self.op = nn.Linear(
            in_features=dim_input,
            out_features=dim_output,
            bias=not layer_norm,
        )
        if spectral_norm:
            self.op = nn.utils.spectral_norm(self.op)

    def forward(self, inputs):
        return self.op(inputs)


class DenseFeatureExtractor(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_hidden,
        depth,
        negative_slope,
        layer_norm,
        dropout_rate,
        spectral_norm=False,
        consistent_dropout=False,
        activate_output=True,
        architecture="resnet",
    ):
        super(DenseFeatureExtractor, self).__init__()
        self.op = nn.Sequential()
        hidden_module = MODULES[architecture]
        if depth == 0:
            self.op.add_module(
                name="hidden_layer_0",
                module=nn.Identity(),
            )
        else:
            for i in range(depth):
                if i == 0:
                    self.op.add_module(
                        name="input_layer",
                        module=DenseLinear(
                            dim_input=dim_input,
                            dim_output=dim_hidden,
                            layer_norm=layer_norm,
                            spectral_norm=spectral_norm,
                        ),
                    )
                else:
                    self.op.add_module(
                        name="hidden_layer_{}".format(i),
                        module=hidden_module(
                            dim_input=dim_hidden,
                            dim_output=dim_hidden,
                            bias=not layer_norm,
                            negative_slope=negative_slope,
                            dropout_rate=dropout_rate,
                            layer_norm=layer_norm,
                            spectral_norm=spectral_norm,
                            consistent_dropout=consistent_dropout,
                        ),
                    )
        if activate_output:
            self.op.add_module(
                name="output_activation",
                module=DenseActivation(
                    dim_input=dim_hidden,
                    negative_slope=negative_slope,
                    dropout_rate=dropout_rate,
                    layer_norm=layer_norm,
                    consistent_dropout=consistent_dropout,
                ),
            )
        self.dim_output = dim_hidden

    def forward(self, inputs):
        return self.op(inputs)