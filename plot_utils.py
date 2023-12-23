import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


# Plot learned posterior P(theta | x_obs)
def pairplot_with_groundtruth_2d(
    samples_list,
    labels,
    colors,
    theta_true=None,
    prior_bounds=None,
    param_names=None,
    plot_bounds=None,
):
    columns = [r"$\theta_1$", rf"$\theta_2$"]
    if param_names is not None:
        columns = param_names

    dfs = []
    for samples, label in zip(samples_list, labels):
        df = pd.DataFrame(samples, columns=columns)
        df["Distribution"] = label
        dfs.append(df)

    dfs = pd.concat(dfs, ignore_index=True)

    pg = sns.pairplot(
        dfs,
        hue="Distribution",
        corner=True,
        palette=dict(zip(labels, colors)),
    )

    if theta_true is not None:
        pg.axes.ravel()[0].axvline(x=theta_true[0], ls="--", linewidth=2, c="black")
        pg.axes.ravel()[3].axvline(x=theta_true[1], ls="--", linewidth=2, c="black")
        pg.axes.ravel()[2].scatter(
            theta_true[0], theta_true[1], marker="o", c="black", s=50, edgecolor="white"
        )

    if prior_bounds is not None:
        pg.axes.ravel()[0].axvline(x=prior_bounds[0][0], ls="--", linewidth=1, c="red")
        pg.axes.ravel()[0].axvline(x=prior_bounds[0][1], ls="--", linewidth=1, c="red")
        pg.axes.ravel()[3].axvline(x=prior_bounds[1][0], ls="--", linewidth=1, c="red")
        pg.axes.ravel()[3].axvline(x=prior_bounds[1][1], ls="--", linewidth=1, c="red")

    if plot_bounds is not None:
        pg.axes.ravel()[2].set_xlim(plot_bounds[0])
        pg.axes.ravel()[3].set_xlim(plot_bounds[1])
        pg.axes.ravel()[2].set_ylim(plot_bounds[1])

    return pg
