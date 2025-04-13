from src.py.utils import config, plotter, utils
from src.py.utils.consts import dmmean, dmwidtho, dmwidthi, deltamean, deltawidtho, deltawidthi
import pandas as pd
import matplotlib.pyplot as plt


plot_dir = config.plot_dirs[2]
# dmmean, dmwidtho, dmwidthi = consts.dmmean, consts.dmwidtho, consts.dmwidthi
# deltamean, deltawidtho, deltawidthi = consts.deltamean, consts.deltawidtho, consts.deltawidthi
filename_particle_first = False  # True -> "sb_K_P.png", False -> "sb_P_K.png"

def filename_generator(mode: str, style: str, particle: str, quantity: str, plotrange=None, ext=None) -> str:
    """
    Generates a filename for the plots. e.g. "./plots/2_sigbkg_py/plain/sb_K_P.png".
    Args:
        mode: str. The mode of the plot. E.g. "plain" or "gmm".
        style: str. The style of the plot. E.g. "sb" or "scatter".
        particle: str. The particle to plot. E.g. "K" or "pi".
        quantity: str. The quantity to plot. E.g. "P" or "ETA".
        plotrange: tuple. The range of the plot. Default is None.
        ext: str. The extension of the file. Default is None.

    Returns:
        str. The relative path of the plot.
    """
    ret = f"{mode}/{style}_"
    if filename_particle_first:
        ret += particle + "_" + quantity
    else:
        ret += quantity + "_" + particle
    if plotrange:
        ret += f"_{plotrange[1]}"
    if ext:
        ret += f".{ext.replace('.', '').lower()}"
    ret = f"{plot_dir}/{ret}"
    return ret

def split_data(data: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """
    Splits the data into signal and background based on the D_M and delta_M
    Args:
        data: pd.DataFrame. Data to split.

    Returns:
        tuple(pd.Index, pd.Index). The signal and background indices.
    """
    data = data[["D_M", "delta_M"]]
    data_sig = data.index[(data["D_M"] > dmmean - dmwidthi/2) &
                          (data["D_M"] < dmmean + dmwidthi/2) &
                          (data["delta_M"] > deltamean - deltawidthi/2) &
                          (data["delta_M"] < deltamean + deltawidthi/2)]
    data_bkg = data.index[(data["D_M"] < dmmean - dmwidtho/2) |
                          (data["D_M"] > dmmean + dmwidtho/2) |
                          (data["delta_M"] < deltamean - deltawidtho/2) |
                          (data["delta_M"] > deltamean + deltawidtho/2)]
    return data_sig, data_bkg

def concat_pis(data: pd.DataFrame, conserve_index=True) -> pd.DataFrame:
    """
    Concatenates the pi1, pi2, pi3 columns into a single pi column and returns pi-only data.
    Args:
        data: pd.DataFrame. Contains data of pi1, pi2, pi3.
        conserve_index: bool. If True, conserves the original index of the data.

    Returns:
        pd.DataFrame. The concatenated pi columns.
    """
    to_concat = []
    for i in range(1, 4):
        this_pis_cols = [col for col in data.columns if f"pi{i}_" in col]
        this_pis_df = data[this_pis_cols]
        this_pis_df = this_pis_df.rename(columns={col: col.replace(f"pi{i}_", "pi_") for col in this_pis_cols})
        to_concat.append(this_pis_df)
    pis_df = pd.concat(to_concat, axis=0)
    if conserve_index:
        pis_df = pis_df.sort_index()
    return pis_df

def plot_sigbkg(data_sig: pd.Series, data_bkg: pd.Series, plot_range=None, log=False,
                plot_quantity="characteristic", x_label=None, filename=None) -> tuple[float, float]:
    """
    Plots the signal and background distributions of a given quantity and saves the plot under `plot_dir`.
    Args:
        data_sig: pd.Series. Signal data.
        data_bkg: pd.Series. Background data.
        plot_range: tuple. Range to plot in.
        log: bool or list. If True, plots the y-axis in log scale. If list, first element is for x-axis, second for y-axis.
        plot_quantity: str. Name of the quantity being plotted. Default is "characteristic".
        x_label: str. Label for x-axis. Default is `plot_quantity`.
        filename: str. Filename to save the plot as. Ideally full relative path (e.g. "./plots/2_sigbkg_py/plain/sb_K_P.png").

    Returns:
        tuple. The x limits of the plot.
    """
    if filename is None:
        raise ValueError("Filename cannot be None.")
    if plot_range:
        data_sig = data_sig[(data_sig > plot_range[0]) & (data_sig < plot_range[1])]
        data_bkg = data_bkg[(data_bkg > plot_range[0]) & (data_bkg < plot_range[1])]
    if len(data_sig) == 0 and len(data_bkg) == 0:
        return 0, 0
    fig = plt.figure(figsize=plotter.figsize)
    fig.set_label("sig vs bkg of " + plot_quantity)
    plt.hist(data_sig, bins=300, density=True, alpha=0.5, label="signal")
    plt.hist(data_bkg, bins=300, density=True, alpha=0.5, label="background")
    if log is not None:
        if isinstance(log, list):
            plt.xscale("log") if log[0] else None
            plt.yscale("log") if log[1] else None
        elif type(log) == bool and log:
            plt.yscale("log")
    x_label = x_label if x_label else plot_quantity
    plt.xlabel(x_label)
    y_label = "Density (log scale)" if log else "Density"
    plt.ylabel(y_label)
    plt.legend()
    plt.title(f"Normalised {plot_quantity} distribution")
    plt.savefig(plotter.check_filename(filename), bbox_inches=plotter.plotstyle)
    ret = plt.xlim()
    plt.close()
    return ret

def pD_finder(data: pd.DataFrame) -> pd.Series:
    """
    Returns the pD column for the given data.

    Args:
        data: pd.DataFrame. Data to calculate pD for.

    Returns:
        pd.Series. pD column.
    """
    target_particles = ["K", "pi1", "pi2", "pi3"]
    # m_D^2 = E_D^2 - P_D^2 = (E_K + E_pi1 + E_pi2 + E_pi3)^2 - P_D^2
    # P_D^2 = (E_K + E_pi1 + E_pi2 + E_pi3)^2 - m_D^2
    energies = data[[f"{particle}_P" for particle in target_particles]].sum(axis=1)
    pD = (energies**2 - data["D_M"]**2)**0.5
    return pD

def plot_stuff(data_all: pd.DataFrame, data_pis: pd.DataFrame,
               sigrows: pd.Index | pd.Series | list, bkgrows: pd.Index | pd.Series | list,
               mode: str, verbose=True) -> None:
    """
    Plots the following:\n
    - 2D histogram of D0 mass vs (D0 mass - D* mass)
    - scatter plot of D0 mass vs (D0 mass - D* mass)
    - sig-vs-bkg of \n
      - D0 mass and (D0 mass - D* mass)
      - momenta (K, pi, pi1, pi2, pi3)
      - eta (K, pi, pi1, pi2, pi3)
      - IP (K, pi, pi1, pi2, pi3)
      - IPCHI2 (K, pi, pi1, pi2, pi3, D, Dst)
      - BPVFDCHI2 (D, Dst)
      - CHI2 (K, pi, pi1, pi2, pi3, D, Dst)

    Args:
        data_all: pd.DataFrame. Data to plot.
        data_pis: pd.DataFrame. Data of concatenated pi columns. Should have the same index as `data_all`.
        sigrows: pd.Index | pd.Series | list. Rows of signal data.
        bkgrows: pd.Index | pd.Series | list. Rows of background data.
        mode: str. Mode of the plot. E.g. "plain" or "gmm".
        verbose: bool. If True, prints logs. Default is True

    Returns:
        None
    """
    data_sig = data_all.loc[sigrows]
    data_bkg = data_all.loc[bkgrows]
    data_sig_pis = data_pis.loc[sigrows]
    data_bkg_pis = data_pis.loc[bkgrows]
    utils.log("Data split.") if verbose else None

    # plot D0 mass vs (D0 mass - D* mass)
    plot_range = plotter.plot_2dmass(data_all, filename=config.plotdir2(mode) + "/2dhist_mass_all")
    plotter.plot_2dmass(data_sig, plot_range=plot_range, filename=config.plotdir2(mode) + "/2dhist_mass_sig", bins=20)
    plotter.plot_2dmass(data_bkg, plot_range=plot_range, filename=config.plotdir2(mode) + "/2dhist_mass_bkg")
    utils.log("Mass plots done.") if verbose else None

    # plot m histogram
    for particle in ["D", "delta", "Dst"]:
        particlem = f"{particle}_M"
        plot_sigbkg(data_sig[particlem], data_bkg[particlem],
                    plot_quantity=particlem,
                    filename=filename_generator(mode, "sb", particle, "M"))

        # # plot full mass distribution
        # plt.figure(figsize=plotter.figsize)
        # plt.hist(data_all[particlem], bins=300)
        # plt.title(f"{particle} mass distribution")
        # plt.xlabel(f"{particle} mass [MeV/c^2]")
        # plt.ylabel("Density")
        # plt.savefig(config.plotdir2('general') + f"/hist_act_{particle}.png", bbox_inches=plotter.plotstyle)
        # plt.close()

    # plot p
    # plot_scatter(data[["K_P", "K_TRACK_P"]], "K_P", "K_TRACK_P")  # exactly the same
    for particle in ["K", "pi", "pi1", "pi2", "pi3", "D"]:
        mom_range = (0, 150000) if particle == "D" else (0, 100000)
        particlep = f"{particle}_P"
        sig, bkg = (data_sig_pis, data_bkg_pis) if particle == "pi" else (data_sig, data_bkg)
        plot_sigbkg(sig[particlep], bkg[particlep],
                    plot_range=mom_range, plot_quantity=particlep,
                    filename=filename_generator(mode, "sb", particle, "P"))
    del particle, mom_range, particlep, sig, bkg
    utils.log("Momentum plots done.") if verbose else None

    # plot eta
    for particle in ["K", "pi", "pi1", "pi2", "pi3"]:
        particleeta = f"{particle}_ETA"
        sig, bkg = (data_sig_pis, data_bkg_pis) if particle == "pi" else (data_sig, data_bkg)
        plot_sigbkg(sig[particleeta], bkg[particleeta],
                    plot_quantity=particleeta,
                    filename=filename_generator(mode, "sb", particle, "ETA"))
    del particle, particleeta, sig, bkg
    utils.log("Eta plots done.") if verbose else None

    # plot ip
    for particle in ["K", "pi", "pi1", "pi2", "pi3"]:
        data, sig, bkg = (data_pis, data_sig_pis, data_bkg_pis) if particle == "pi" else (data_all, data_sig, data_bkg)
        plotter.plot_scatter(data[[f"{particle}_BPVIP", f"{particle}_MINIP"]],
                     f"{particle}_MINIP", f"{particle}_BPVIP",
                     filename=filename_generator(mode, "scatter", particle, "IP"))
        for ip in ["BPVIP", "MINIP"]:
            for iprange in [(0, 0.5), (0, 2.5), None]:
                particleip = f"{particle}_{ip}"
                plot_sigbkg(sig[particleip], bkg[particleip],
                            plot_range=iprange, log=True, plot_quantity=particleip,
                            filename=filename_generator(mode, "sb", particle, ip, iprange))
    del particle, data, sig, bkg, ip, iprange, particleip
    utils.log("IP plots done.") if verbose else None

    # plot IPCHI2
    for particle in ["K", "pi", "pi1", "pi2", "pi3", "D", "Dst"]:
        data, sig, bkg = (data_pis, data_sig_pis, data_bkg_pis) if particle == "pi" else (data_all, data_sig, data_bkg)
        plotter.plot_scatter(data[[f"{particle}_BPVIPCHI2", f"{particle}_MINIPCHI2"]],
                     f"{particle}_MINIPCHI2", f"{particle}_BPVIPCHI2",
                     filename=filename_generator(mode, "scatter", particle, "IPCHI2"))
        for ipchi2 in ["BPVIPCHI2", "MINIPCHI2"]:
            for ipchi2range in [(0, 100), (0, 1000), (0, 10000), None]:
                particleipchi2 = f"{particle}_{ipchi2}"
                if particle == "Dst" and ipchi2range is None:
                    ipchi2range = (0, 8e6)
                plot_sigbkg(sig[particleipchi2], bkg[particleipchi2],
                            plot_range=ipchi2range, log=True, plot_quantity=particleipchi2,
                            filename=filename_generator(mode, "sb", particle, ipchi2, ipchi2range))
    del particle, data, sig, bkg, ipchi2, ipchi2range, particleipchi2
    utils.log("IPCHI2 plots done.") if verbose else None

    # BPVFDCHI2
    for particle in ["D", "Dst"]:
        for fdchi2range in [(0, 100), (0, 1000), (0, 10000), None]:
            particlefdchi2 = f"{particle}_BPVFDCHI2"
            if particle == "Dst" and fdchi2range is None:
                fdchi2range = (0, int(8e6))
            plot_sigbkg(data_sig[particlefdchi2], data_bkg[particlefdchi2],
                        plot_range=fdchi2range, log=True, plot_quantity=particlefdchi2,
                        filename=filename_generator(mode, "sb", particle, "BPVFDCHI2", fdchi2range))
    del particle, fdchi2range, particlefdchi2
    utils.log("BPVFDCHI2 plots done.") if verbose else None

    # plot CHI2
    for particle in ["K", "pi", "pi1", "pi2", "pi3", "D", "Dst"]:
        sig, bkg = (data_sig_pis, data_bkg_pis) if particle == "pi" else (data_sig, data_bkg)
        for chi2range in [(0, 1), (0, 10), (0, 100), None]:
            particlechi2 = f"{particle}_CHI2"
            plot_sigbkg(sig[particlechi2], bkg[particlechi2],
                        plot_range=chi2range, log=True, plot_quantity=particlechi2,
                        filename=filename_generator(mode, "sb", particle, "CHI2", chi2range))
    del particle, sig, bkg, chi2range, particlechi2
    utils.log("CHI2 plots done.") if verbose else None
    