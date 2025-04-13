import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from src.py.prepare.dataprep import find_inout
from src.py.utils import utils, config, plotter, consts


def calc_purisigni(
        proba: pd.Series, ans: pd.Series,
        power_tp: int | float, power_pos: int | float,
        resolution=300
) -> np.ndarray:
    """
    Calculates purity, significance, or related values for a given model that can be calculated as (TP)^power_tp / (TP + FP)^power_tpfn
    Args:
        proba: pd.Series. Probabilities of the model
        ans: pd.Series. True answers
        power_tp: int | float. Power of true positives in the formula
        power_pos: int | float. Power of positives in the formula
        resolution: int. Number of points to plot

    Returns:
        np.ndarray. Values of the formula for different cuts
    """
    thresholds = np.linspace(0, 1, resolution)
    positives = (proba.values[:, None] > thresholds)
    tps = (positives & (ans.values[:, None] == 1)).sum(axis=0)
    positives = positives.sum(axis=0)

    tps = np.where(positives == 0, 1, tps)
    positives = np.where(positives == 0, 1, positives)
    return (tps ** power_tp) / (positives ** power_pos)


def plot_purisigni(
        proba: pd.Series, ans: pd.Series, inout: pd.Series = None,
        resolution=300, logx=True, model_info: dict=None
) -> None:
    """
    Plots purity, significance, and purity*significance for a given model
    Args:
        proba: pd.Series. Probabilities of the model. Index should match ``ans``
        ans: pd.Series. True answers. 1 for signal, 0 for background
        inout: pd.Series. 1 for central region, -1 for outer region. If None, only full region is plotted
        resolution: int. Number of points to plot
        logx: bool. Whether to plot x-axis in log scale
        model_info: dict. Information about the model. Must contain key "model", and optionally "ratio" and "join_type"

    Returns:
        None
    """
    model_name = model_info["model"]
    purisignis = {
        "purity": [1, 1],
        "significance": [1, 0.5],
        "purisigni": [2, 1.5]
    }
    if len(proba) > 100000:
        proba = proba.sample(100000, random_state=42)
        ans = ans.loc[proba.index]
        if inout is not None:
            inout = inout.loc[proba.index]
    for name, powers in purisignis.items():
        kwargs = {"power_tp": powers[0], "power_pos": powers[1], "resolution": resolution}

        fig = plt.figure(figsize=plotter.figsize)
        fig.set_label(f"{name}_" + model_name if model_name is not None else f"{name}")
        cuts = np.linspace(0, 1, resolution)
        values_full = calc_purisigni(proba, ans, **kwargs)
        plt.plot(cuts, values_full, label="Full")
        if inout is not None:
            values_in = calc_purisigni(proba[inout == 1], ans[inout == 1], **kwargs)
            values_out = calc_purisigni(proba[inout == -1], ans[inout == -1], **kwargs)
            plt.plot(cuts, values_in, label="Central region")
            plt.plot(cuts, values_out, label="Outer region")
            plt.plot(cuts, values_in - values_out, label="Central - Outer")
        plt.xlabel("Cut")
        if name == "purisigni":
            plt.ylabel("Purity * Significance")
            plt.title("Purity * Significance")
        else:
            plt.ylabel(name.capitalize())
            plt.title(name.capitalize())
        plt.legend()
        if logx:
            plt.xscale("log")

        if model_name is not None:
            file_path = config.plotdir3(**model_info, plot_type=name)
            plotter.check_filename(file_path)
            plt.savefig(file_path, bbox_inches=plotter.plotstyle)
        else:
            plt.show()
        plt.close()


def plot_masses(mass_df: pd.DataFrame, proba: pd.Series, model_info: dict=None) -> None:
    """
    Plots masses for D_M and delta_M for different cuts of the model
    Args:
        mass_df: pd.DataFrame. Dataframe with D_M and delta_M columns
        proba: pd.Series. Probabilities of the model
        model_info: dict. Information about the model. Must contain key "model", and optionally "ratio" and "join_type"

    Returns:
        None
    """
    mass_df = mass_df[["D_M", "delta_M"]]
    cuts = [digit / 10 for digit in range(1, 10)] + [digit/100 for digit in range(1, 10)]
    cuts = [cut for cut in cuts if proba.max() > cut > proba.min()]  # otherwise all bkg or all sig
    if len(proba) != len(mass_df):
        raise ValueError("proba and mass_df must have the same length")
    for cut in cuts:
        bkg = mass_df[proba < cut]
        sig = mass_df[proba > cut]
        deltam_df = pd.DataFrame({"bkg": bkg["delta_M"], "sig": sig["delta_M"]})
        dm_df = pd.DataFrame({"bkg": bkg["D_M"], "sig": sig["D_M"]})
        masses = [["deltam", deltam_df, "D*-D0 mass"], ["dm", dm_df, "D0 mass"]]
        for this_masstype, this_massdf, this_masslabel in masses:
            # e.g. ./plots/3_nn/bdt10/dm/10percent, but might change. see config.py
            filename = config.plotdir3(**model_info, plot_type=this_masstype) + f"/{int(cut * 100)}percent"
            common_kwargs = {
                "log": False,
                "density": False,
                "x_label": f"{this_masslabel} [MeV/c^2]",
                # "cut_range": [consts.sneha_masscuts[f"{this_masstype}min"],
                #               consts.sneha_masscuts[f"{this_masstype}max"]],
                "cut_range": [consts.mc_stats.get(f"{this_masstype}min"),
                              consts.mc_stats.get(f"{this_masstype}max")],
            }
            plotter.plot_1dhist(
                this_massdf,
                **common_kwargs,
                style="stacked",
                filename=f"{filename}.png",
            )
            if len(this_massdf["sig"].dropna()) != 0:
                plotter.plot_1dhist(
                    this_massdf["sig"],
                    **common_kwargs,
                    color='orange',
                    filename=f"{filename}_sig.png",
                )
            if len(this_massdf["bkg"].dropna()) != 0:
                plotter.plot_1dhist(
                    this_massdf["bkg"],
                    **common_kwargs,
                    color='blue',
                    filename=f"{filename}_bkg.png",
                )


def plot_sigdensity(
        proba: pd.Series, inout: pd.Series, resolution=300, logx=False, model_info: dict=None
) -> None:
    """
    Plots sig_in, sig_out, and (sig_in-sig_out) on the same plot
    Args:
        proba: pd.Series. Probabilities of the model
        inout: pd.Series. True for central region, False for outer region
        resolution: int. Number of points to plot
        logx: bool. Whether to plot x-axis in log scale
        model_info: dict. Information about the model. Must contain key "model", and optionally "ratio" and "join_type"

    Returns:
        None
    """
    model_name = model_info["model"]
    if len(proba) > 100000:
        proba = proba.sample(100000, random_state=42)
    thresholds = np.linspace(0, 1, resolution)
    sig_in = (proba[inout].values[:, None] > thresholds).sum(axis=0) / len(proba[inout])
    sig_out = (proba[inout].values[:, None] > thresholds).sum(axis=0) / len(proba[~inout])
    sig_diff = sig_in - sig_out
    fig = plt.figure(figsize=plotter.figsize)
    fig.set_label("SigDensity_" + model_name if model_name is not None else "SigDensity")
    plt.plot(np.linspace(0, 1, resolution), sig_in, label="Central region")
    plt.plot(np.linspace(0, 1, resolution), sig_out, label="Outer region")
    plt.plot(np.linspace(0, 1, resolution), sig_diff, label="Central - Outer")
    plt.xlabel("Cut")
    plt.ylabel("Signal density")
    plt.title("Signal density")
    plt.legend()
    if logx:
        plt.xscale("log")
    if model_name is not None:
        plottype = "sigdensity" + ("_log" if logx else "")
        file_path = config.plotdir3(**model_info, plot_type=plottype)
        plotter.check_filename(file_path)
        plt.savefig(file_path, bbox_inches=plotter.plotstyle)
    else:
        plt.show()
    plt.close()


def plot_probs_roc(proba: pd.Series, ans: pd.Series, tt_flag: pd.Series, model_info: dict=None) -> None:
    """
    Plots two plots: one with probabilities of the model for train and test samples, and the other with ROC curve
    Args:
        proba: pd.Series. Probabilities of the model
        ans: pd.Series. True answers
        tt_flag: pd.Series. 0 for train, 1 for test
        model_info: dict. Information about the model. Must contain key "model", and optionally "ratio" and "join_type"

    Returns:
        None
    """
    train_proba = proba[tt_flag == 0]
    train_y = ans[tt_flag == 0]
    test_proba = proba[tt_flag == 1]
    test_y = ans[tt_flag == 1]
    plotter.compare_probs(
        train_probs=train_proba,
        train_y=train_y,
        test_probs=test_proba,
        test_y=test_y,
        filename=config.plotdir3(**model_info, plot_type="probs"),
    )
    plotter.plot_roc(ans=test_y, proba=test_proba,
                     filename=config.plotdir3(**model_info, plot_type="roc"))


@utils.alert
def main() -> None:
    """
    Main function that plots purisignis, masses, and signal densities for each model
    Returns:
        None
    """
    # Plot 1. purity, significance, and purity*significance for each model
    # 2. masses for D_M and delta_M for each model
    # probas_tt = pd.read_csv(f"{config.probas_dir}/proba_tt.csv")
    probas_tt = utils.load(config.tt_score_file)
    models_cols = [item for item in probas_tt.columns if not item.startswith("_")]
    for col in models_cols:
        model_info = {
            "model": col.split("_")[0],
            "ratio": col.split("_")[-2 if col.count("_") == 2 else -1],
            "join_type": col.split("_")[-1] if col.count("_") == 2 else None
        }
        tt_flag = probas_tt[f"_is_test_{model_info['ratio']}"]
        ans_series = probas_tt[f"_ans_{model_info['ratio']}"]
        proba_tt = probas_tt[col]
        plot_purisigni(proba=proba_tt.loc[tt_flag == 1],
                       ans=ans_series.loc[tt_flag == 1],
                       logx=False, model_info=model_info)
        utils.log(f"Plotting purisignis for {col} done")
        plot_probs_roc(proba=proba_tt, ans=ans_series, tt_flag=tt_flag, model_info=model_info)
        utils.log(f"Plotting roc and predictions for {col} done")

    # Plot 3. masses for D_M and delta_M for each model
    # 4. signal density for each model
    # probas_full = pd.read_csv(f"{config.probas_dir}/proba_full.csv")
    probas_full = utils.load(config.target_score_file)
    full_df = utils.load(config.target_core_file)
    for col in probas_full.columns:
        model_info = {
            "model": col.split("_")[0],
            "ratio": col.split("_")[-2 if col.count("_") == 2 else -1],
            "join_type": col.split("_")[-1] if col.count("_") == 2 else None
        }
        proba_full = probas_full[col]
        utils.log(f"Plotting masses for {col}")
        mass_df = full_df[["D_M", "delta_M"]]
        plot_masses(mass_df=mass_df, proba=proba_full, model_info=model_info)
        utils.log(f"Plotting masses for {col} done")

        # # as we are using the core file, we do not plot the signal density comparison
        # utils.log(f"Plotting sigdensity for {col}")
        # inouts = find_inout(mass_df)
        # for logx in [True, False]:
        #     plot_sigdensity(proba=proba_full, inout=inouts, logx=logx, model_info=model_info)
        # utils.log(f"Plotting sigdensity for {col} done")

    # # lines below is for the case when og_ind column might be used
    # for ind in main03_1.available_bkgsig_ratios():
    #     proced = utils.load(f'mc_proced_{ind}')
    #     og_ind = proced[proced["origin"] == "full"]["og_index"]
    #     for model in probas_full.columns:
    #         if not model.endswith(str(ind)) and f"_{str(ind)}_" not in model:
    #             continue
    #         utils.log(f"Plotting masses for {model}")
    #         proba_full = probas_full[model]
    #         mass_df = full_df[["D_M", "delta_M"]]
    #         plot_masses(mass_df=mass_df, proba=proba_full, model_name=model)
    #         # filtered_mass_df = full_df[~full_df.index.isin(og_ind)][["D_M", "delta_M"]]
    #         # plot_masses(mass_df=mass_df, proba=proba_full[filtered_mass_df.index], model_name=model)
    #         utils.log(f"Plotting masses for {model} done")
    #
    #         utils.log(f"Plotting sigdensity for {model}")
    #         inouts = find_inouts(mass_df)
    #         for logx in [True, False]:
    #             plot_sigdensity(proba=proba_full, inout=inouts, logx=logx, model_name=model)
    #         utils.log(f"Plotting sigdensity for {model} done")


if __name__ == "__main__":
    main()
