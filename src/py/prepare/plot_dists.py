import numpy as np
import matplotlib.pyplot as plt
from math import pi
from src.py.utils import utils, consts, config


def plot_2dhist(df, col1, col2, file_path, plot_boundary=False, plot_range=None):
    """
    Plot 2D histogram of col1 and col2 of df and save it as file_path
    Args:
        df: pd.DataFrame. Data to plot
        col1: str. Name of the first column
        col2: str. Name of the second column
        file_path: str. Path to save the plot
        plot_boundary: bool. Whether to plot the boundary of the ellipse
        plot_range: list. Range of the plot

    Returns:
        None
    """
    plt.figure()
    plt.hist2d(df[col1], df[col2], bins=100)
    plt.title(f"{col1} vs {col2}")
    plt.xlabel(col1)
    plt.ylabel(col2)
    if plot_boundary:
        if config.cut_method == 'ellipse':
            t = np.linspace(0, 2*pi, 100)
            x = consts.mc_stats.dmradius * np.cos(t) + consts.mc_stats.dmmean
            y = consts.mc_stats.deltamradius * np.sin(t) + consts.mc_stats.deltammean
            plt.plot(x, y, color='red')
        else:  # rectangle
            from src.py.utils.plotter import plot_box
            plot_box([consts.mc_stats.dmmean, consts.mc_stats.deltammean],
                     [consts.mc_stats.dmradius * 2, consts.mc_stats.deltamradius * 2],
                     color='red')

    if plot_range:
        plt.xlim(plot_range[0], plot_range[1])
        plt.ylim(plot_range[2], plot_range[3])
    plt.savefig(file_path)
    plt.close()
    utils.log(f"plotting {col1} vs {col2} done")


def main():
    target_files = [config.mc_proced_file(ratio) for ratio in consts.train_bkgsig_ratios]
    target_files += [config.mc_file(), config.target_file, config.target_core_file]
    plot_range = [1800, 1920, 138, 155]
    for target_file in target_files:
        data = utils.load(target_file)
        if 'delta_M' not in data.columns:
            data['delta_M'] = data['Dst_M'] - data['D_M']
        file_path = config.plot_dirs[2] + f'/{target_file.split("/")[-1].split(".")[0]}' + '.png'
        plot_2dhist(data, 'D_M', 'delta_M', file_path, True, plot_range)


if __name__ == "__main__":
    main()