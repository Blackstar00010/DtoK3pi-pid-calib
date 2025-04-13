import os
from src.py.utils import utils, plotter, config
from src.py.intro import common_fns as cf

# This file is used to plot the mass distributions of the full data and the mc data.
# The mass distributions are plotted in 1D and 2D histograms.
# The plots are saved under `plot_dir`.


def plot_mass_full():
    """
    Plots full mass distribution of mc data and full data. Plots saved under `plot_dir`.
    Returns:
        None
    """
    twodrange, dmrange, dstmrange, deltamrange = None, None, None, None
    for data_type in ['full', 'mc', 'mc_proced_1']:
        data = utils.load(data_type)
        if 'sig' in data.columns:  # only sig data for mc
            data = data[data['sig'] == 1]
        if 'delta_M' not in data.columns:
            data['delta_M'] = abs(data['Dst_M'] - data['D_M'])
        data = data[['D_M', 'delta_M', "Dst_M"]]
        utils.log(f"{data_type} loaded.")

        data_type = data_type.replace('full', 'act').replace('_proced_1', '_filtered')
        twodrange = plotter.plot_2dmass(
            data, plot_range=twodrange, box=True,
            filename=f'{config.plotdir1("general")}/2dhist_mass_{data_type}'
        )
        dmrange = plotter.plot_1dhist(
            data['D_M'], plot_range=dmrange,
            x_label="D0 mass [MeV/c^2]", y_label='Counts',
            title=f'D0 mass distribution ({data_type})',
            filename=f"{config.plotdir1('general')}/hist_{data_type}_D"
        )
        dstmrange = plotter.plot_1dhist(
            data['Dst_M'], plot_range=dstmrange,
            x_label="D* mass [MeV/c^2]",
            title=f'D* mass distribution ({data_type})',
            filename=f"{config.plotdir1('general')}/hist_{data_type}_Dst"
        )
        deltamrange = plotter.plot_1dhist(
            data['delta_M'], plot_range=deltamrange,
            x_label="D*-D0 mass [MeV/c^2]",
            title=f'D*-D0 mass distribution ({data_type})',
            filename=f"{config.plotdir1('general')}/hist_{data_type}_delta"
        )
        utils.log(f"{data_type} mass plots done.")


@utils.alert
def main():
    mode = "plain"
    # mode_dir = f"{cf.plot_dir}/{mode}"
    mode_dir = config.plotdir1(mode)
    if not os.path.exists(mode_dir):
        os.makedirs(mode_dir)

    # load data
    data = utils.load('full')
    utils.log("Data loaded.")
    data = data[utils.useful_cols_plot(data.columns)]
    data["D_P"] = cf.pD_finder(data)
    data_pi = cf.concat_pis(data, conserve_index=True)
    utils.log("pi columns concatenated.")

    # split data
    sig_index, bkg_index = cf.split_data(data[['D_M', 'delta_M']])
    utils.log("Data split.")

    utils.log("Plotting main plots.")
    cf.plot_stuff(data, data_pi, sig_index, bkg_index, mode)
    utils.log("Main plots done.")

    utils.log("Plotting full mass plots.")
    plot_mass_full()
    utils.log("Full mass plots done.")


if __name__ == "__main__":
    # main()
    plot_mass_full()
