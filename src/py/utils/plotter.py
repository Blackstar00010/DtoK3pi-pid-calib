import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve, roc_auc_score
from src.py.utils import utils, config
from src.py.utils.consts import prototype


# default settings for plotting
figsize = (10, 6)
plotstyle = "tight"
filename_particle_first = False  # True -> "sb_K_P.png", False -> "sb_P_K.png"

# default values for the D0 mass and delta mass
dmmean = prototype.dmmean
deltamean = prototype.deltammean
dmwidtho = prototype.dmwidtho
deltawidtho = prototype.deltamwidtho
dmwidthi = prototype.dmwidthi
deltawidthi = prototype.deltamwidthi


def check_filename(filename: str, check_ext=True, check_relpath=True, check_exists=True) -> str:
    """
    Check if the filename is valid. If not, correct it.
    Args:
        filename: str. Filename to check.
        check_ext: bool. If True, checks if the filename has an extension. If not, adds '.png'.
        check_relpath: bool. If True, checks if the filename starts with './plots/'.
        check_exists: bool. If True, checks if the enclosing folder of the filename exists.

    Returns:
        str. Corrected filename.
    """
    if check_ext:
        if not filename.endswith('.png'):
            filename += '.png'
        # if filename.count('.') == 0:
        #     filename += ".png"
        # elif not any([filename.lower().endswith(ext) for ext in [".png", ".pdf", ".jpg"]]):
        #     raise ValueError(f"Invalid extension: {filename}. Must be one of '.png', '.pdf', '.jpg'.")

    if check_relpath:
        # if filename = subdir/filename.ext, convert to ./plots/subdir/filename.ext where subdir starts with a number
        
        # for legacy code
        if filename.startswith('./plots/'):
            filename = config.proj_dir + filename[1:]
        elif filename.startswith('plots/'):
            filename = config.proj_dir + '/' + filename

        good = False
        for value in config.plot_dirs.values():
            if filename.startswith(value):
                good = True
                break
        if not good:
            continue_q = utils.loginput(f"Is {filename} a valid filename? (y/n): ")
            if continue_q.lower() != 'y':
                raise ValueError(f"Invalid filename: {filename}.")

    if check_exists and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    return filename


def plot_1dhist(data: pd.Series | pd.DataFrame, bins=300, cut_range: tuple=None, plot_range: tuple=None,
                log=False, density=False, style="alpha", color: str=None,
                x_label: str=None, y_label: str=None, title: str=None, filename: str=None) -> tuple[float, float]:
    """
    Plots a 1D histogram of the data.
    Args:
        data: pd.Series | pd.DataFrame. Data to plot. If more than one series is to be plotted, pass as a pd.DataFrame.
        bins: int. Number of bins to use in the histogram.
        cut_range: tuple. Range to cut data to. e.g. (xmin, xmax).
        plot_range: tuple. Range to plot in. e.g. (xmin, xmax).
        log: bool. If True, plots the y-axis in log scale.
        density: bool. If True, plots the density instead of counts.
        style: str. Style of the plot if data is a pd.DataFrame. Default is "alpha". Other options are "sbs" (side-by-side) and "stacked".
        color: str. Color of the plot. Default is None. Only used if data is a pd.Series.
        x_label: str. Label for x-axis. Default is "Value" if data is not a pd.Series, else the name of the series.
        y_label: str. Label for y-axis. Default is "Count" if density is False, else "Density".
        title: str. Title of the plot. Default is "<x_label> distribution".
        filename: str. Filename to save the plot as. Ideally full relative path (e.g. "./plots/0_nice_plots/plot1.png").

    Returns:
        tuple. The x limits of the plot.
    """
    if filename is None:
        raise ValueError("Filename must be provided.")
    if cut_range:
        data = data[(data > cut_range[0])
                    & (data < cut_range[1])
                    & (data.notna())]
    if density:
        return plot_1dhist_density(data, nbins=bins,
                                   cut_range=cut_range, plot_range=plot_range, log=log, style=style,
                                   x_label=x_label, y_label=y_label, title=title, filename=filename)
    fig = plt.figure(figsize=figsize)
    fig.set_label('1dHist_' + filename if filename is not None else '1dHist')
    if isinstance(data, pd.Series):
        plt.hist(data, bins=bins, density=density, label=str(data.name), color=color)
    elif style == "sbs":
        plt.hist(data, bins=bins, density=density, label=data.columns)
    elif style == "stacked":
        plt.hist(data, bins=bins, density=density, stacked=True, label=data.columns)
    elif style == "alpha":
        alpha = 1 / len(data.columns) ** (1 / 1.5)
        for col in data.columns:
            if data[col].isna().all():
                continue
            plt.hist(data[col], bins=bins, density=density, alpha=alpha, label=col)
    else:
        raise ValueError(f"Invalid style: {style}. Must be one of 'sbs', 'stacked', 'alpha'.")
    plt.legend()

    if log:
        plt.yscale("log")
    if x_label is None:
        if isinstance(data, pd.Series) and data.name is not None:
            x_label = data.name
        else:
            x_label = "Value"
    plt.xlabel(x_label)
    y_label = 'Count' if y_label is None else y_label
    if log:
        y_label += " (log scale)"
    plt.ylabel(y_label)
    title = title if title else f"{x_label} distribution"
    plt.title(title)
    if plot_range:
        plt.xlim(plot_range)
    plt.savefig(check_filename(filename), bbox_inches=plotstyle)
    ret = plt.xlim()
    plt.close()
    return ret

# todo: implement 1d hist density for pd.Series
def plot_1dhist_density(data: pd.DataFrame, nbins=300,
                        cut_range: tuple=None, plot_range: tuple=None, log=False, style="alpha",
                        x_label: str=None, y_label: str=None, title: str=None, filename: str=None) -> tuple[float, float]:
    """
    Plots a 1D histogram of the data with density.
    Args:
        data: pd.DataFrame. Data to plot.
        nbins: int. Number of bins to use in the histogram.
        cut_range: tuple. Range to cut data to. e.g. (xmin, xmax).
        plot_range: tuple. Range to plot in. e.g. (xmin, xmax).
        log: bool. If True, plots the y-axis in log scale.
        style: str. Style of the plot. Default is "alpha". Other options are "sbs" (side-by-side) and "stacked".
        x_label: str. Label for x-axis.
        y_label: str. Label for y-axis.
        title: str. Title of the plot. Default is "<x_label> distribution".
        filename: str. Filename to save the plot as. Ideally full relative path (e.g. "./plots/0_nice_plots/plot1.png").

    Returns:
        tuple. The x limits of the plot.
    """
    if filename is None:
        raise ValueError("Filename must be provided.")
    if cut_range:
        data = data[(data > cut_range[0]) & (data < cut_range[1])
                    & (data.notna())]

    # Plot histogram and get counts and bin edges
    fig = plt.figure(figsize=figsize)
    fig.set_label('1dHistDensity_' + filename if filename is not None else '1dHistDensity')
    alpha = 1 / len(data.columns) ** (1 / 1.5) if style == "alpha" else 1
    stacked = style == "stacked"

    bin_entries, bins, barcontainers = plt.hist(data, bins=nbins, alpha=alpha, label=data.columns, stacked=stacked)
    normalized_counts = bin_entries / bin_entries.sum(axis=1, keepdims=True)
    if stacked:
        normalized_heights = normalized_counts.cumsum(axis=0)
        for i, barcontainer in enumerate(barcontainers):
            for j, bar in enumerate(barcontainer):
                bottom = 0 if i == 0 else normalized_heights[i - 1][j]
                top = normalized_heights[i][j]
                bar.set_height(top - bottom)
                bar.set_y(bottom)
                bar.set_fill(True)
        max_height = normalized_heights.max()
    else:
        for i, barcontainer in enumerate(barcontainers):
            if style == "alpha" and len(data.columns) > 2 and i >= (len(data.columns) // 2):
                alpha = 0
            for j, bar in enumerate(barcontainer):
                bar.set_height(normalized_counts[i][j])
                if alpha == 0:
                    bar.set_edgecolor(bar.get_facecolor())
                    bar.set_fill(False)
                else:
                    bar.set_fill(True)
                if style == "alpha":
                    bar.set_x(bins[j])
                    bar.set_width(bins[j + 1] - bins[j])
        max_height = normalized_counts.max()

    # Adjust axes to reflect the normalized scale
    if plot_range:
        plt.xlim(plot_range)
    plt.ylim(0, max_height * 1.1)
    if log:
        plt.yscale("log")

    # Labels and title
    title = title if title else f"{x_label} distribution"
    plt.title(title)
    x_label = "Value" if x_label is None else x_label
    plt.xlabel(x_label)
    y_label = 'Density' if y_label is None else y_label
    plt.ylabel(y_label)
    plt.legend()

    # Save the plot
    plt.savefig(check_filename(filename), bbox_inches=plotstyle)
    ret = plt.xlim()
    plt.close()
    return ret

def plot_box(center, width, color="red"):
    center_x, center_y = center
    width_x, width_y = width
    x1 = center_x - width_x / 2
    x2 = center_x + width_x / 2
    y1 = center_y - width_y / 2
    y2 = center_y + width_y / 2
    plt.plot([x1, x2], [y1, y1], color=color)
    plt.plot([x1, x2], [y2, y2], color=color)
    plt.plot([x1, x1], [y1, y2], color=color)
    plt.plot([x2, x2], [y1, y2], color=color)

def plot_2dmass(data: pd.DataFrame, cut_range: tuple = None, plot_range: tuple = None, bins=300, box=True, box_range=None,
                filename=None) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Plots a 2D histogram of D0 mass vs (D0 mass - D* mass).
    Args:
        data: pd.DataFrame. Data to plot.
        cut_range: tuple. Range to cut data to. e.g. (xmin, xmax).
        plot_range: tuple. Range to plot in. e.g. ((xmin, xmax), (ymin, ymax)).
        bins: int. Number of bins to use in the histogram. Not used if `plot_range` is given.
        box: bool. If True, plots a box around the mean values.
        box_range: tuple. Range of the box. Default is None. Should be in the form of [(dmmean, deltamean), (dmwidth, deltawidth)]
        filename: str. Filename to save the plot as. Ideally full relative path (e.g. "./plots/0_nice_plots/plot1.png").

    Returns:
        tuple. The x and y limits of the plot.
    """
    if filename is None:
        raise ValueError("Filename must be provided.")
    data = data[["D_M", "delta_M"]].dropna()

    binsize_x = (1920 - 1800) / 300
    binsize_y = (155 - 140) / 300
    binx = bins if cut_range is None else int((data["D_M"].max() - data["D_M"].min()) / binsize_x)
    biny = bins if cut_range is None else int((data["delta_M"].max() - data["delta_M"].min()) / binsize_y)

    fig = plt.figure(figsize=figsize)
    fig.set_label('2dMass_' + filename if filename is not None else '2dMass')
    plt.hist2d(data["D_M"], data["delta_M"], bins=(binx, biny))
    if box:
        if box_range is None:
            plot_box((dmmean, deltamean), (dmwidtho, deltawidtho), color="red")
            plot_box((dmmean, deltamean), (dmwidthi, deltawidthi), color="black")
        else:
            plot_box(*box_range, color='red')
    plt.title(f"(D0 mass - D* mass) vs D0 mass")
    plt.xlabel("D0 mass [MeV/c^2]")
    plt.ylabel("D0 mass - D* mass [MeV/c^2]")
    if plot_range:
        plt.xlim(plot_range[0])
        plt.ylim(plot_range[1])
    plt.colorbar()
    plt.savefig(check_filename(filename), bbox_inches=plotstyle)
    ret = plt.xlim(), plt.ylim()
    plt.close()
    return ret

def plot_2dhist(data: pd.DataFrame, colx, coly, cut_range: tuple = None, plot_range: tuple = None,
                bins: int | list=300, filename: str = None) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Plots a 2D histogram of colx vs coly.
    Args:
        data: pd.DataFrame. Data to plot.
        colx: any (usually str or numeric). Column to plot on x-axis.
        coly: any (usually str or numeric). Column to plot on y-axis.
        cut_range: tuple. Range to cut data to. e.g. (xmin, xmax).
        plot_range: tuple. Range to plot in. e.g. ((xmin, xmax), (ymin, ymax)).
        bins: int or list. Number of bins to use in the histogram. Not used if `plot_range` is given.
        filename: str. Filename to save the plot as. Ideally full relative path (e.g. "./plots/0_nice_plots/plot1.png").

    Returns:
        tuple. The x and y limits of the plot.
    """
    if filename is None:
        raise ValueError("Filename must be provided.")
    if cut_range:
        data = data[(data[colx] > cut_range[0][0]) & (data[colx] < cut_range[0][1]) &
                    (data[coly] > cut_range[1][0]) & (data[coly] < cut_range[1][1])]
    fig = plt.figure(figsize=figsize)
    fig.set_label('2dHist_' + filename if filename is not None else '2dHist')

    # 3d histogram with z axis being the number of entries in each bin
    if isinstance(bins, int):
        bins = (bins, bins)
    plt.hist2d(data[colx], data[coly], bins=bins)
    plt.title(f"{colx} vs {coly}")
    plt.xlabel(colx)
    plt.ylabel(coly)
    if plot_range:
        plt.xlim(plot_range[0])
        plt.ylim(plot_range[1])
    plt.colorbar()
    plt.savefig(check_filename(filename), bbox_inches=plotstyle)
    ret = plt.xlim(), plt.ylim()
    plt.close()
    return ret


def plot_scatter(data: pd.DataFrame, colx, coly,
                 cut_range: tuple = None, plot_range: tuple = None, filename: str = None) -> tuple:
    """
    Plots a scatter plot of colx vs coly.
    Args:
        data: pd.DataFrame. Data to plot.
        colx: any (usually str or numeric). Column to plot on x-axis.
        coly: any (usually str or numeric). Column to plot on y-axis.
        cut_range: tuple. Range to cut data to. e.g. (xmin, xmax).
        plot_range: tuple. Range to plot in. e.g. ((xmin, xmax), (ymin, ymax)).
        filename: str. Filename to save the plot as. Ideally full relative path (e.g. "./plots/0_nice_plots/plot1.png").

    Returns:
        tuple. The x and y limits of the plot.
    """
    if filename is None:
        raise ValueError("Filename must be provided.")
    if cut_range:
        data = data[(data[colx] > cut_range[0][0]) & (data[colx] < cut_range[0][1]) &
                    (data[coly] > cut_range[1][0]) & (data[coly] < cut_range[1][1])]
    fig = plt.figure(figsize=figsize)
    fig.set_label('Scatter_' + filename if filename is not None else 'Scatter')
    plt.scatter(data[colx], data[coly], s=0.1)
    if plot_range:
        plt.xlim(plot_range[0])
        plt.ylim(plot_range[1])
    plt.xlabel(colx)
    plt.ylabel(coly)
    plt.title(f"{colx} vs {coly}")
    plt.savefig(check_filename(filename), bbox_inches=plotstyle)
    ret = plt.xlim(), plt.ylim()
    plt.close()
    return ret


def plot_probs(probs, test_y, filename):
    """
    Plot the probabilities of the test data
    Args:
        probs: list | np.array. Probabilities of the test data
        test_y: list | np.array. True labels of the test data
        filename: str. Filename to save the plot as. Ideally full relative path (e.g. "./plots/0_nice_plots/plot1.png").

    Returns:
        None
    """
    sig_test = probs[test_y == 1]
    bkg_test = probs[test_y == 0]
    longer_len = max(len(sig_test), len(bkg_test))
    sig_test = np.pad(sig_test, (0, longer_len - len(sig_test)), constant_values=np.nan)
    bkg_test = np.pad(bkg_test, (0, longer_len - len(bkg_test)), constant_values=np.nan)
    to_plot = pd.DataFrame({"sig": sig_test, "bkg": bkg_test})
    plot_1dhist(to_plot, bins=100, log=False, density=True, style="alpha",
                x_label="Probability", filename=filename)


def compare_probs(train_probs: list | np.ndarray | pd.Series,
                  train_y : list | np.ndarray | pd.Series,
                  test_probs: list | np.ndarray | pd.Series,
                  test_y: list | np.ndarray | pd.Series,
                  filename: str) -> None:
    """
    Compare the probabilities of the training and test data. Plots three histograms:\n
    - 1. Training signal vs test signal
    - 2. Training background vs test background
    - 3. Training signal vs training background
    Args:
        train_probs: list | np.array | pd.Series. Probabilities of the training data
        train_y: list | np.array | pd.Series. True labels of the training data
        test_probs: list | np.array | pd.Series. Probabilities of the test data
        test_y: list | np.array | pd.Series. True labels of the test data
        filename: str. Filename to save the plot as. Ideally full relative path (e.g. "./plots/0_nice_plots/plot1.png"). If "./plots/example.png" is given, plots are saved as "./plots/example.png", "./plots/example_sig.png", "./plots/example_bkg.png".

    Returns:
        None
    """
    if filename is None:
        raise ValueError("Filename must be provided.")
    sig_train = np.array(train_probs)[np.array(train_y) == 1]
    bkg_train = np.array(train_probs)[np.array(train_y) == 0]
    sig_test = np.array(test_probs)[np.array(test_y) == 1]
    bkg_test = np.array(test_probs)[np.array(test_y) == 0]

    # K-S test on sig_train vs sig_test, bkg_train vs bkg_test
    ks_sig = ks_2samp(sig_train[~np.isnan(sig_train)], sig_test[~np.isnan(sig_test)])
    ks_bkg = ks_2samp(bkg_train[~np.isnan(bkg_train)], bkg_test[~np.isnan(bkg_test)])

    # for matching lengths
    longer_len = max(len(sig_train), len(bkg_train), len(sig_test), len(bkg_test))
    sig_train = np.pad(sig_train, (0, longer_len - len(sig_train)), constant_values=np.nan)
    bkg_train = np.pad(bkg_train, (0, longer_len - len(bkg_train)), constant_values=np.nan)
    sig_test = np.pad(sig_test, (0, longer_len - len(sig_test)), constant_values=np.nan)
    bkg_test = np.pad(bkg_test, (0, longer_len - len(bkg_test)), constant_values=np.nan)

    to_plot = pd.DataFrame({"sig_train": sig_train, "bkg_train": bkg_train, "sig_test": sig_test, "bkg_test": bkg_test})
    filename = filename.replace('.png', '')
    plot_1dhist(to_plot, bins=100, log=False, density=True, style="alpha", title="Prediction distribution comparison",
                x_label="Predictions", filename=f"{filename}.png")
    plot_1dhist(to_plot[["sig_train", "sig_test"]], bins=100, log=False, density=True, style="alpha",
                x_label="Probability",
                title=f"Prediction distribution of signal data\np value of KS test: {np.round(ks_sig.pvalue, 2)}",
                filename=f"{filename}_sig.png")
    plot_1dhist(to_plot[["bkg_train", "bkg_test"]], bins=100, log=False, density=True, style="alpha",
                x_label="Probability",
                title=f"Prediction distribution of background data\np value of KS test: {np.round(ks_bkg.pvalue, 2)}",
                filename=f"{filename}_bkg.png")


def plot_roc(ans: pd.Series, proba: pd.Series, filename: str):
    """
    Plot the ROC curve
    Args:
        ans: pd.Series. True labels
        proba: pd.Series. Probabilities
        filename: str. Filename to save the plot as. Ideally full relative path (e.g. "./plots/0_nice_plots/plot1.png").

    Returns:
        None
    """
    if filename is None:
        raise ValueError("Filename must be provided.")
    fpr, tpr, _ = roc_curve(ans, proba)
    auc = roc_auc_score(ans, proba)
    fig = plt.figure(figsize=figsize)
    fig.set_label('ROC')
    plt.plot(fpr, tpr, label=f"AUC: {auc}")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
    plt.savefig(check_filename(filename), bbox_inches=plotstyle)
    plt.close()


if __name__ == "__main__":
    raise RuntimeError("Module plotter.py is not meant to be run as a script. Import it and use the functions.")
