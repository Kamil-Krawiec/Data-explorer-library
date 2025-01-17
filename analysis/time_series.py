import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def time_series_plot(df, time_column, value_column, connect_points=True, marker=True, filepath=None):
    """
    Plots time series data to identify trends, seasonal patterns, and anomalies over time.

    Parameters:
        df (pd.DataFrame): The data frame containing the time series data.
        time_column (str): The name of the column containing the time data.
        value_column (str): The name of the column containing the values to plot.
        connect_points (bool): Whether to connect the points with lines. Default is True.
        marker (bool): Whether to show markers (circles) around the points. Default is True.
        filepath (str or None): If provided, the plot will be saved to this file. Default is None (plot is not saved).

    Returns:
        None: Displays or saves the plot.
    """
    df[time_column] = pd.to_datetime(df[time_column])

    df = df.sort_values(by=time_column)

    sns.set(style="whitegrid")

    plt.figure(figsize=(14, 7))
    if connect_points:
        if marker:
            plt.plot(df[time_column], df[value_column], marker="o", linestyle="-", label=value_column, alpha=0.7)
        else:
            plt.plot(df[time_column], df[value_column], linestyle="-", label=value_column, alpha=0.7)
    else:
        if marker:
            plt.scatter(df[time_column], df[value_column], marker="o", label=value_column, alpha=0.7)
        else:
            plt.scatter(df[time_column], df[value_column], label=value_column, alpha=0.7)

    plt.title("Time Series Plot", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Value", fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=12)

    if filepath:
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Plot saved to {filepath}")
    plt.show()