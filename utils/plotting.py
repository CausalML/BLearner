import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 50
import matplotlib.pyplot as plt
import seaborn as sns
rc = {
    "figure.constrained_layout.use": True,
    "axes.titlesize": 20,
}
sns.set_theme(style="darkgrid", palette="colorblind", rc=None)

sns.set(style="whitegrid", palette="colorblind")
params = {
    "figure.constrained_layout.use": True,
    "axes.labelsize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 18,
    "legend.title_fontsize": 18,
    "font.size": 18,
}
plt.rcParams.update(params)

_FUNCTION_COLOR = "#ad8bd6"


def ggplot_log_style_deferral(figsize, log_y=False, loc_maj_large=True):
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # Give plot a gray background like ggplot.
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 16
    ax.set_facecolor('#EBEBEB')
    # Remove border around plot.
    [ax.spines[side].set_visible(False) for side in ax.spines]
    # Style the grid.
    ax.grid(which='major', color='white', linewidth=1.2)
    ax.tick_params(which='minor', bottom=False, left=False)

    return ax

def fill_between(
    x,
    y,
    x_label,
    y_label,
    alpha=0.2,
    y_scale="log",
    x_lim=[0, 1.0],
    y_lim=[3e-4, 5],
    x_pad=-20,
    y_pad=-45,
    legend_loc="upper right",
    file_path=None,
):
    ax = ggplot_log_style_deferral(figsize=(682 / 72, 512 / 72), log_y=True)
    # _ = plt.figure(figsize=(682 / 72, 512 / 72), dpi=72)
    for k, v in y.items():
        _ = plt.plot(x, v["mean"], c=v["color"], ls=v["line_style"], label=k)
        _ = plt.fill_between(
            x,
            v["mean"] - v["ci"],
            v["mean"] + v["ci"],
            alpha=alpha,
            color=v["color"],
        )
    # _ = plt.xlabel(x_label)
    # _ = plt.ylabel(y_label)
    _ = plt.yscale(y_scale)
    _ = plt.xlim(x_lim)
    _ = plt.ylim(y_lim)
    _ = plt.tick_params(axis="x", direction="in", pad=x_pad)
    _ = plt.tick_params(axis="y", direction="in", pad=y_pad)
    _ = plt.legend(loc=legend_loc)
    plt.legend(fontsize=12)
    _ = plt.savefig(file_path, dpi=100)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    _ = plt.show()


def errorbar(
    x,
    y,
    y_err,
    x_label,
    y_label,
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
    _ = plt.savefig(file_path, dpi=300)
    _ = plt.close()
