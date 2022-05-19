import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_text_feature(
    df: pd.DataFrame, feat: str, k: int = 20, logY: bool = False
) -> None:
    """
    This function's aim is to plot the repartition of a textual feature of the data
    df : The dataframe which contains the data
    feat: The feature we want to plot
    k : The maximum number of values of the feature we want to see on the plot
    logY: Parameter which set if the scale must be in log
    returns -> None
    """
    # Set seaborn theme
    sns.set_theme()

    # Subplot
    fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios": [3, 1]})
    fig.set_figheight(15)
    fig.set_figwidth(25)

    # Keep the k values with more occurencies
    vc = df[feat].value_counts()
    keys = list(vc.keys())[: min(len(vc), k)]

    # Bar chart of the number of occurences wrt. each value (from the top k)
    ax[0].bar(keys, [vc[key] for key in keys], log=logY)
    ax[0].set_xticklabels(labels=keys, rotation=45, ha="right")
    ax[0].set_title(f"Repartition of values among {feat}")

    # Cumulative repartition of the feature
    cumu_sum_list = [list(vc.items())[i][1] for i in range(len(vc.keys()))]
    ax[1].plot(np.arange(len(vc)), np.cumsum(cumu_sum_list / np.sum(cumu_sum_list)))
    ax[1].set_title(f"Cumulative repartition of {feat} wrt. number of values")


def plot_continous_feature(
    df: pd.DataFrame, feat: str, k: int = 20, Log: bool = False
) -> None:
    """
    This function's aim is to plot the repartition of a numeric feature of the data
    df : The dataframe which contains the data
    feat: The feature we want to plot
    k : The maximum number of values of the feature we want to see on the plot
    logY: Parameter which set if the scale must be in log
    returns -> None
    """
    # Set seaborn theme
    sns.set_theme()

    # Subplot
    fig, ax = plt.subplots(1, 4)
    fig.set_figheight(15)
    fig.set_figwidth(25)

    # Plot histogramm of feature
    ax[0].set_xscale("log")
    ax[0].hist(df[feat], bins=100, log=Log)
    ax[0].set_title(f"Histogramm of {feat}")

    # Plot repartition function of feature
    sns.kdeplot(df[feat], bw_method=0.1, ax=ax[1])
    ax[1].set_xscale("log")
    ax[1].set_title(f"Repartition funtion of {feat}")

    # Plot cumulative repartition function of feature
    sns.kdeplot(df[feat], bw_method=0.1, cumulative=True, ax=ax[2])
    ax[2].set_xscale("log")
    ax[2].set_title(f"Cumulative repartition funtion of {feat}")

    # Boxplot of feature
    ax[3].boxplot(df[feat])
    ax[3].set_yscale("log")
    ax[3].set_title(f"Boxplot of {feat}")
