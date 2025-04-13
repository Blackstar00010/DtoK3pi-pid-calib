import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import src.utils.utils as utils
from src.utils import plotter, config, consts
import fns


# this file performs a 2D fit on the D_M and delta_M data
# NOT DONE YET


def flatten2dto1d(df: pd.DataFrame | np.ndarray) -> pd.Series:
    """
    Flatten a 2D DataFrame to a 1D Series.
    Args:
        df: pd.DataFrame. the 2D DataFrame to flatten.

    Returns:
        pd.Series. the 1D Series.
    """
    if isinstance(df, np.ndarray):
        return pd.Series(df.flatten())
    return pd.Series(df.values.flatten())


def mul_outer(data1: pd.Series | np.ndarray, data2: pd.Series | np.ndarray) -> pd.DataFrame:
    """
    Find the outer product of two 1D Series. That is, ret_ij = data1_i * data2_j, or in dataframe notation, ret.iloc[i, j] = data1.iloc[i] * data2.iloc[j].
    Args:
        data1: pd.Series | np.ndarray. the first data.
        data2: pd.Series | np.ndarray. the second data.

    Returns:
        pd.DataFrame. the outer product.
    """
    data1 = pd.Series(data1, name='data1') if not isinstance(data1, pd.Series) else data1
    data2 = pd.Series(data2, name='data2') if not isinstance(data2, pd.Series) else data2
    return pd.DataFrame(np.outer(data1, data2), index=data1.index, columns=data2.index)


def plot_mountain(df):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    x, y = np.meshgrid(df.columns, df.index)  # Grid of x and y
    x = x.ravel()  # Flatten for bar3d
    y = y.ravel()  # Flatten for bar3d
    z = np.zeros_like(x)  # Base z-axis is 0
    values = df.values.ravel()  # Flatten the DataFrame values for bar heights

    # Define bar widths
    dx = dy = (df.columns[1] - df.columns[0]) * 0.8  # Bar width based on column spacing
    dz = values  # Bar heights are the data values

    # Plot the bars
    ax.bar3d(x, y, z, dx, dy, dz, shade=True)

    # Set axis labels
    # ax.set_xlabel('Columns (X-axis)')
    # ax.set_ylabel('Index (Y-axis)')
    # ax.set_zlabel('Values (Z-axis)')
    plt.title('residuals')

    plt.show()


def find_2dfit(data: pd.Series, cols: list):
    """
    Find the best fit for a 2D histogram.
    Args:
        data: pd.Series. the data to fit.
        cols: list. the columns to fit.

    Returns:
        dict. the best fit parameters.
    """
    # fit
    hist, xedges, yedges = np.histogram2d(data[cols[0]], data[cols[1]], bins=100)
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    x, y = np.meshgrid(xcenters, ycenters)
    # x = x.flatten()
    # y = y.flatten()
    z = hist.flatten() / 1000  # to match the scale of the other functions

    # todo: separate sig into sig1 and sig2
    sig_fn, sig_nparam = fns.gaussian_1d, fns.no_params['gaussian']
    sig1_p0 = [data[cols[0]].mean(), data[cols[0]].std()/3, max(z)**0.5]
    sig2_p0 = [data[cols[1]].mean(), data[cols[1]].std()/3, max(z)**0.5]
    bkg1_fn, bkg1_nparam = fns.linear_1d, fns.no_params['lin']
    bkg1_p0 = [0.01, z.mean()]
    bkg2_fn, bkg2_nparam = fns.linear_1d, fns.no_params['lin']
    bkg2_p0 = [0.01, z.mean()]

    def target_fn_unflattened(xy_values, *params):
        x_values, y_values = xy_values

        # todo: there should be some better way to do this
        # signal pion + signal D
        count_sig = sig_fn(x_values, *params[:sig_nparam]) * sig_fn(y_values, *params[sig_nparam:(sig_nparam * 2)])
        # background pion + signal D
        count_bkg_pion = (bkg1_fn(x_values, *params[(sig_nparam * 2):(sig_nparam * 2 + bkg1_nparam)])
                          * sig_fn(y_values, *params[(sig_nparam * 2 + bkg1_nparam):(sig_nparam * 3 + bkg1_nparam)]))
        # signal pion + background D
        count_bkg_d = (sig_fn(x_values, *params[(sig_nparam * 3 + bkg1_nparam):
                                                (sig_nparam * 4 + bkg1_nparam)])
                       * bkg2_fn(y_values, *params[(sig_nparam * 4 + bkg1_nparam):
                                                   (sig_nparam * 4 + bkg1_nparam + bkg2_nparam)]))
        # background pion + background D
        count_bkg_pure = (bkg1_fn(x_values,
                                  *params[(sig_nparam * 4 + bkg1_nparam + bkg2_nparam):
                                          (sig_nparam * 4 + bkg1_nparam * 2 + bkg2_nparam)])
                          * bkg2_fn(y_values,
                                    *params[(sig_nparam * 4 + bkg1_nparam * 2 + bkg2_nparam):
                                            (sig_nparam * 4 + bkg1_nparam * 2 + bkg2_nparam * 2)]))
        return count_sig + count_bkg_pion + count_bkg_d + count_bkg_pure

    num_calls = 0
    def target_fn(xy_values, *params):
        nonlocal num_calls
        num_calls += 1
        return target_fn_unflattened(xy_values, *params).ravel()

    # initial guess
    initial_guess = sig1_p0 + sig2_p0 + bkg1_p0 + sig2_p0 + sig1_p0 + bkg2_p0 + bkg1_p0 + bkg2_p0
    print(f'initial_guess: {initial_guess}')

    # fit
    fit = curve_fit(target_fn, (x, y), z, p0=initial_guess, maxfev=100000, method='dogbox', full_output=True)
    print(f'num_calls: {num_calls}')
    popt = fit[0]

    fitted_data = target_fn_unflattened((x, y), *popt) * 1000  # revert the scaling

    return popt, fitted_data




@utils.alert
def main():
    utils.log(f'Running {__file__}...')
    full_df = utils.load('full')[['D_M', "delta_M"]]

    # apply sneha's mass cuts
    full_df = full_df[(full_df['D_M'] < consts.sneha_masscuts['dmmax']) &
                      (full_df['D_M'] > consts.sneha_masscuts['dmmin'])]
    full_df = full_df[(full_df['delta_M'] < consts.sneha_masscuts['deltammax']) &
                      (full_df['delta_M'] > 138.5)]

    plotter.plot_2dhist(full_df, 'D_M', 'delta_M',
                        cut_range=None, plot_range=None, bins=100,
                        filename=config.plotdir4() + f"/2dhist.png")

    # find the best fit
    popt = find_2dfit(full_df, ['D_M', 'delta_M'])
    utils.log(f'popt:{list(popt[0])}')

    # plot the residuals
    hists, xedges, yedges = np.histogram2d(full_df['D_M'], full_df['delta_M'], bins=100)
    residuals = pd.DataFrame(hists - popt[1], index=xedges[:-1], columns=yedges[:-1])
    plot_mountain(residuals)
    # plt.imshow(residuals, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')
    # plt.colorbar()
    # plt.show()



if __name__ == "__main__":
    main()
