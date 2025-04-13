import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.py.utils import utils, config


def compare_load() -> None:
    """
    Compare the time it takes to load data from root and csv files
    Returns:
        None
    """
    start = time.time()
    data = utils.load('full')
    time_root = time.time() - start

    data.to_csv("./temp.csv", index=False)

    start = time.time()
    data = utils.load('./temp.csv')
    time_csv = time.time() - start

    os.remove("./temp.csv")

    print(f"Time to load from root in seconds: {time_root}")  # ~5s
    print(f"Time to load from csv in seconds: {time_csv}")  # ~25s
    print(f"Root is {time_csv/time_root} times faster than csv")


def plot_hist(data_series: np.array, title: str, xlabel: str = None, output_dir: str = config.plot_dirs[1]) -> None:
    """
    Plot histogram of data_series and save it to output_dir
    Args:
        data_series: list or pd.Series or np.array. Data to plot
        title: str. Title of the plot
        xlabel: str. Label of x-axis
        output_dir: str. Directory to save the plot

    Returns:
        None
    """
    plt.figure()
    plt.hist(data_series, bins=100)
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)

    filename = title.lower().replace(' ', '_')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.close()

    utils.log(f"plotting {title} done")


def plot_dist_all(data: pd.DataFrame) -> None:
    """
    Plot histogram of all columns in data
    Args:
        data: pd.DataFrame. Data to plot

    Returns:
        None
    """
    for col in data.columns:
        col_data = data[col]
        col_data = col_data.dropna()
        if col_data.dtype in [np.float64, np.int64] and len(col_data) > 0:
            # plot_hist(col_data, col, output_dir="../../plots/1_prac_py/_dists")
            plot_hist(col_data, col, output_dir=config.plot_dirs[1] + '/_dists')


def plot_cdf(data: pd.DataFrame) -> None:
    """
    Plot CDF of PID_K
    Args:
        data: pd.DataFrame. Data to plot

    Returns:
        None
    """
    col1, col2 = "K_PID_K", "pi1_PID_K"
    # select -150 < PID_K < 150
    data = data[(data[col1] > -150) & (data[col1] < 150) & (data[col2] > -150) & (data[col2] < 150)]
    data1 = data[col1].value_counts().sort_index().cumsum()
    data2 = data[col2].value_counts().sort_index().cumsum()
    plt.figure()
    plt.plot(max(data1) - data1, label="K_PID_K")
    plt.plot(max(data2) - data2, label="pi1_PID_K")
    plt.title(f"Inverse CDF of PID_K")
    plt.xlabel("PID_K")
    plt.xlim(-150, 150)
    plt.ylabel("CDF")
    plt.legend()
    plt.savefig(f"{config.plot_dirs[1]}/cdf_PID_K.png")
    plt.close()
    utils.log(f"plotting CDF of PID_K done")


def plot_lldiff(data: pd.DataFrame) -> None:
    """
    Plot the difference between K and PI for PID and PROBNN
    Args:
        data: pd.DataFrame. Data to plot

    Returns:
        None
    """
    for particle in ["K", "pi1", "pi2", "pi3"]:
        for data_type in ["PID", "PROBNN"]:
            base_col = f"{particle}_{data_type}_"
            col1, col2 = base_col + "K", base_col + "PI"
            particle_data = data[[col1, col2]]
            particle_data = particle_data[(particle_data[col1] > -150) & (particle_data[col1] < 150) & (particle_data[col2] > -150) & (particle_data[col2] < 150)]
            data_diff = particle_data[col1] - particle_data[col2]
            plot_hist(data_diff, f"{base_col}K - {base_col}PI", f"{base_col}K - {base_col}PI")


def plot_pid_diff(data: pd.DataFrame) -> None:
    """
    Plot the difference between K_PID and max(PIDX) for X in {K, pi1, pi2, pi3}
    Args:
        data: pd.DataFrame. Data to plot

    Returns:
        None
    """
    for particle in ["K", "pi1", "pi2", "pi3"]:
        for data_type in ["PID", "PROBNN"]:
            base_col = f"{particle}_{data_type}_"
            target = "K" if particle == "K" else "PI"
            col_target = base_col + target
            col_others = [col for col in data.columns if base_col in col and col != col_target]
            data_target = data[col_target]
            data_other = data[col_others].max(axis=1)
            data_diff = data_target - data_other
            plot_hist(data_diff, f"{col_target} - max({base_col}X)", f"{col_target} - max({base_col}X)")


@utils.alert
def main():
    # compare_load()
    # utils.log("start")
    data = utils.load('long')
    # plot_dist_all(data)
    # plot_cdf(data)
    # plot_lldiff(data)
    # plot_pid_diff(data)


if __name__ == "__main__":
    main()
