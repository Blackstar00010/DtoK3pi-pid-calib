import os
from typing import Literal
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import MinMaxScaler
from src.utils import utils, plotter, config
import common_fns as cf


# This file is used to split the data into signal and background based on the mass and other features.
# It uses the data from the previous file and preprocesses it to remove outliers and select signal-likely data points.
# It then splits the data based on the mass using various methods like GMM, ellipse, KMeans, and DBSCAN.
# The data is then plotted to show the signal and background points.


def split_data_ellipse(df: pd.DataFrame,
                       center_x: float, center_y: float,
                       radius_out_x: float, radius_out_y: float,
                       radius_in_x: float, radius_in_y: float) -> tuple[pd.Index, pd.Index]:
    """
    Split data into signal and background based on the distance from the origin.
    Args:
        df: pd.DataFrame. The data to split.
        center_x: float. The x-coordinate of the center of the ellipse.
        center_y: float. The y-coordinate of the center of the ellipse.
        radius_out_x: float. The x-radius of the outer ellipse.
        radius_out_y: float. The y-radius of the outer ellipse.
        radius_in_x: float. The x-radius of the inner ellipse.
        radius_in_y: float. The y-radius of the inner ellipse.

    Returns:
        tuple. The signal and background indices.
    """
    x, y = df["D_M"], df["delta_M"]
    dist_out = np.sqrt((x - center_x) ** 2 / radius_out_x ** 2 + (y - center_y) ** 2 / radius_out_y ** 2)
    dist_in = np.sqrt((x - center_x) ** 2 / radius_in_x ** 2 + (y - center_y) ** 2 / radius_in_y ** 2)
    sig = df[dist_in < 1]
    bkg = df[dist_out > 1]
    return sig.index, bkg.index

def split_data_gmm(df: pd.DataFrame, target_size: tuple=None, threshold: tuple=None) -> tuple[pd.Index, pd.Index]:
    """
    Perform Gaussian Mixture Model clustering based on the D_M and delta_M columns.

    Args:
        df: pd.DataFrame. Data to split.
        target_size: tuple. The number of signal and background events to return.
        threshold: tuple. The lower and upper thresholds for the log probability of the data.

    Returns:
        tuple. The signal and background indices.
    """
    if (target_size is not None) and (threshold is not None):
        raise ValueError('Cannot specify both target_size and threshold')
    if (target_size is None) and (threshold is None):
        raise ValueError('Must specify either target_size or threshold')
    if target_size and (sum(target_size) > len(df)):
        raise ValueError('Sum of target_size must be less than or equal to the length of the data')

    # normalise
    # df["D_M"] = (df["D_M"] - dmmean) / df["D_M"].std()
    # df["delta_M"] = (df["delta_M"] - deltamean) / df["delta_M"].std()

    cols = ['D_M', 'delta_M']
    lencols = len(cols)
    pca = PCA(n_components=lencols)
    pca.fit(df[cols])
    pcaed_df = pd.DataFrame(pca.transform(df[cols]))
    pcaed_df = pcaed_df.rename(columns={i: f'PC{i + 1}' for i in range(lencols)})
    # plot_scatter(pcaed_df, "PC1", "PC2", filename=f"gmm/scatter_PCA_before")

    gmm = GaussianMixture(n_components=1, random_state=0, init_params='kmeans', max_iter=10000,
                            means_init=((0, 0),), n_init=5)
    gmm.fit(pcaed_df)
    utils.log(f"Fitted GMM with means={gmm.means_} (target means: (0, 0))")

    log_probs = gmm.score_samples(pcaed_df)
    df["log_probs"] = log_probs

    if threshold is None:
        df = df.sort_values(by='log_probs', ascending=False)
        sig_df = df.head(int(target_size[0])).sort_index()
        bkg_df = df.tail(int(target_size[1])).sort_index()
        # pcaed_df = pcaed_df.sort_values(by='log_probs', ascending=False)
        # plot_scatter(pcaed_df.head(int(target_size[0])), "PC1", "PC2", filename=f"gmm/scatter_PCA_after_sig")
        # plot_scatter(pcaed_df.tail(int(target_size[1])), "PC1", "PC2", filename=f"gmm/scatter_PCA_after_bkg")
    else:
        sig_df = df[log_probs >= threshold[0]]
        bkg_df = df[log_probs < threshold[1]]
        # plot_scatter(pcaed_df[pcaed_df["log_probs"] >= threshold[0]], "PC1", "PC2", filename=f"gmm/scatter_PCA_after_sig")
        # plot_scatter(pcaed_df[pcaed_df["log_probs"] < threshold[1]], "PC1", "PC2", filename=f"gmm/scatter_PCA_after_bkg")
    sig_df = sig_df.drop(columns='log_probs')
    bkg_df = bkg_df.drop(columns='log_probs')
    return sig_df.index, bkg_df.index


def split_data_dbscan(df: pd.DataFrame, min_size: tuple=None) -> tuple[pd.Index, pd.Index]:
    """
    Perform DBSCAN clustering based on the D_M and delta_M columns. Best among the clustering methods.
    Args:
        df: pd.DataFrame. Data to split.
        min_size: tuple. The minimum number of signal and background events to return.

    Returns:
        tuple. The signal and background indices.
    """
    if min_size is None:
        min_size = [100000, 100000]

    iscut = False
    df = df[['D_M', 'delta_M']]
    cut_to = 500000
    if len(df) > cut_to:
        df = df.sample(cut_to, random_state=0)
        iscut = True
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=['D_M', 'delta_M'])

    eps = 0.02
    sig = []
    min_size = [int(min_size[0]/len(df)*cut_to), int(min_size[1]/len(df)*cut_to)]
    labels = None
    while len(sig) < min_size[0]:
        utils.log(f"\tDBSCAN - eps={eps}, min_size={min_size[0]}")
        dbscan = DBSCAN(eps=eps, min_samples=min_size[0])
        labels = dbscan.fit_predict(df)
        sig = df.index[labels != -1]
        eps += 0.02

    if labels is None:
        raise ValueError("DBSCAN failed to find signal points.")
    bkg = df.index[labels == -1]
    if iscut:
        sig, bkg = df[sig], df[bkg]
        mean_point = sig.mean()
        max_sig_dist = sig.apply(lambda x: np.linalg.norm(x - mean_point), axis=1).max()
        max_bkg_dist = bkg.apply(lambda x: np.linalg.norm(x - mean_point), axis=1).max()
        sig = df[(df.apply(lambda x: np.linalg.norm(x - mean_point), axis=1) < max_sig_dist)].index
        bkg = df[(df.apply(lambda x: np.linalg.norm(x - mean_point), axis=1) < max_bkg_dist)].index

    if len(bkg) < min_size[1]:  # too few background points
        mean_point = df[sig].mean()
        bkg = df[labels == -1]
        bkg['dist'] = bkg.apply(lambda x: np.linalg.norm(x - mean_point), axis=1)
        bkg = bkg.sort_values('dist').head(min_size[1]).index

    return sig, bkg


def split_data_kmeans(df: pd.DataFrame, target_size: tuple=None,) -> tuple[pd.Index, pd.Index]:
    """
    Perform K-Means clustering based on the D_M and delta_M columns.
    Args:
        df: pd.DataFrame. Data to split.
        target_size: tuple. The number of signal and background events to return.

    Returns:
        tuple. The signal and background indices.
    """

    df = df[['D_M', 'delta_M']]

    # Fit K-Means with one cluster
    kmeans = KMeans(n_clusters=1, random_state=0)
    kmeans.fit(df)
    utils.log(f"Fitted KMeans with centers={kmeans.cluster_centers_} (target centers: ({plotter.dmmean}, {plotter.deltamean}))")

    # Calculate distances from each point to the cluster center
    distances = np.linalg.norm(df - kmeans.cluster_centers_, axis=1)

    if target_size:
        # Sort the points by distance and select the target number of signal and background points
        df['distance'] = distances
        df = df.sort_values('distance')
        sig = df.head(target_size[0]).index
        bkg = df.tail(target_size[1]).index
        return sig, bkg
    # Set a threshold distance to classify points as outliers (e.g., 2 standard deviations from the mean distance)
    threshold_distance = np.mean(distances) + 2 * np.std(distances)
    core_points = df[distances < threshold_distance]
    outliers = df[distances >= threshold_distance]
    return core_points.index, outliers.index


def preprocess_auto(data: pd.DataFrame, guide_sig_ind: pd.Index, guide_bkg_ind: pd.Index,
                    anomalies_rate=0.01, target_snr=0.5) -> pd.Index:
    """
    Preprocess the data by removing outliers and selecting signal-likely data points based on the 1/2-snr-rule.
    Args:
        data: pd.DataFrame. The data to preprocess.
        guide_sig_ind: pd.Index. The indices of the signal guide data.
        guide_bkg_ind: pd.Index. The indices of the background guide data.
        anomalies_rate: float. The fraction of data to remove from the top and bottom of the distribution.
        target_snr: float. The signal-to-noise ratio to use for selecting signal-likely data points.

    Returns:
        pd.Index. The signal-likely data points.
    """
    if anomalies_rate < 0 or anomalies_rate > 0.5:
        raise ValueError('anomalies_rate must be between 0.0 and 0.5')
    top = data.quantile(1 - anomalies_rate)
    bottom = data.quantile(anomalies_rate)
    data = data[(data < top) & (data > bottom)]
    data = data.dropna()
    guide_sig = data.loc[guide_sig_ind[guide_sig_ind.isin(data.index)]]
    guide_bkg = data.loc[guide_bkg_ind[guide_bkg_ind.isin(data.index)]]

    for col in data.columns:
        if col.endswith("_M") or ("_MIN" in col) or ("pi1" in col) or ("pi2" in col) or ("pi3" in col):
            continue
        # bkg_density[i] is the density of between bkg_bins[i] and bkg_bins[i+1], where 0<i<bins
        bkg_density, bkg_bins = np.histogram(guide_bkg[col], bins=100)
        bkg_density = bkg_density / len(guide_bkg)
        # use the same bins for signal
        sig_density, _ = np.histogram(guide_sig[col], bins=bkg_bins)
        sig_density = sig_density / len(guide_sig)
        snr = sig_density / bkg_density  # ratio[i] is the snr between bkg_bins[i] and bkg_bins[i+1]
        # select bins with snr > target_snr
        likely = np.where(snr > target_snr)[0]  # length == bins, values ranging from 0 to bins-1
        # select data points in those bins
        data['bin'] = np.digitize(data[col], bkg_bins)  # values ranging from 0 to bins+1 (0 and bins+1 are outliers)
        data['bin'] = data['bin'] - 1  # values ranging from -1 to bins (-1 and bins are outliers); bin=i means bkg_bins[i] <= data < bkg_bins[i+1]
        data = data[data['bin'].isin(likely)]
        data = data.drop(columns='bin')
    return data.index

def preprocess_manual(data: pd.DataFrame) -> pd.Index:
    """
    Returns manually-cut signal-likely datapoints which is then used to apply GMM or hard cut to separate signal and background.
    Args:
        data: pd.DataFrame. The data to preprocess.

    Returns:
        pd.Index. The signal-likely datapoints' indices.
    """
    # BPVFDCHI2
    data = data[(data["D_BPVFDCHI2"] < 15000) & (data["D_BPVFDCHI2"] > 50)]
    data = data[data["Dst_BPVFDCHI2"] < 8000]

    # BPVIP
    data = data[data["K_BPVIP"] < 2.2]
    for i in range(1, 4):
        data = data[data[f"pi{i}_BPVIP"] < 2.7]

    # BPVIPCHI2
    data = data[data["D_BPVIPCHI2"] < 5000]
    data = data[data["Dst_BPVIPCHI2"] < 5000]
    data = data[data["K_BPVIPCHI2"] < 8000]
    for i in range(1, 4):
        data = data[data[f"pi{i}_BPVIPCHI2"] < 20000]

    # MINIP
    data = data[data["K_MINIP"] < 2.5]
    for i in range(1, 4):
        data = data[data[f"pi{i}_MINIP"] < 3]

    # MINIPCHI2
    data = data[data["D_MINIPCHI2"] < 5000]
    data = data[data["Dst_MINIPCHI2"] < 6000]
    data = data[data["K_MINIPCHI2"] < 8000]
    for i in range(1, 4):
        data = data[data[f"pi{i}_MINIPCHI2"] < 20000]

    return data.index


def main(misc_cut: Literal["auto", "manual", None]= "auto",
         mass_cut: Literal["gmm", "ellipse", "kmeans", "dbscan", "svm", None]=None, print_logs=True,
         data: pd.DataFrame=None, data_pis: pd.DataFrame=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divides the data into signal and background based on the method specified. If both `preprocess` and `splitter` are None, the function is the same as `main02_1.py`'s `main` function.
    Args:
        misc_cut: str. The method to split the data based on ex-mass data. "auto" uses the 1/2-snr-rule, "manual" uses manual cuts.
        mass_cut: str or None. The method to split the data based on DM & deltaM. "gmm" uses Gaussian Mixture Model, "ellipse" uses ellipse method, "svm" uses Support Vector Machine, None uses hard cut.
        print_logs: bool. Whether to print logs.
        data: pd.DataFrame. The data to use. If None, loads the data.
        data_pis: pd.DataFrame. The data with pion information. If None, loads the data.

    Returns:
        tuple. The data and data_pis.
    """
    if mass_cut == "svm":
        raise NotImplementedError("SVM not implemented yet.")
    if (data is None) != (data_pis is None):
        raise ValueError("Both data and data_pi must be provided.")
    utils.log(f"Starting main02_2.py/main() with misc_cut={misc_cut}, "
          f"mass_cut={mass_cut}, "
          f"{'data length='+str(len(data)) if data is not None else ''}") if print_logs else None

    folder_name = config.plotdir2(mode=f"{misc_cut if misc_cut else 'default'}", 
                                  method=f"{mass_cut if mass_cut else 'default'}")

    if data is None:
        data = utils.load('full')
        data = data[utils.useful_cols_plot(data.columns)]
        data["D_P"] = cf.pD_finder(data)
        data_pis = cf.concat_pis(data)
    utils.log(f"Loaded data") if print_logs else None

    # preprocessing
    if misc_cut == "manual":
        sig_candidates: pd.Index = preprocess_manual(data)
    elif misc_cut == "auto":
        sig, bkg = cf.split_data(data)  # split with hard cutting on DM & deltaM
        sig_candidates: pd.Index = preprocess_auto(data, sig, bkg)
    else:
        sig_candidates = data.index
    massplotrange = plotter.plot_2dmass(data, filename=f"{folder_name}/temp2dmass")  # already in plot_stuff
    os.remove(f"{folder_name}/temp2dmass.png")
    plotter.plot_2dmass(data.loc[sig_candidates], plot_range=massplotrange,
                        filename=f"{folder_name}/2dhist_mass_sig_candidates")
    # semitodo: after adding grey area, uncomment this
    # plotter.plot_2dmass(data.loc[asdfasdfasdf], filename=f"{folder_name}/2dmass_misccut_bkg", plot_range=massplotrange)
    utils.log(f"Preprocessed data") if print_logs else None

    # splitting by masses
    if mass_cut == "ellipse":
        sig, bkg = split_data_ellipse(data, cf.dmmean, cf.deltamean,
                                      cf.dmwidtho, cf.deltawidtho,
                                      cf.dmwidthi, cf.deltawidthi)
    elif mass_cut == "gmm":
        sig, bkg = split_data_gmm(data.loc[sig_candidates],
                                  target_size=(int(0.5*len(sig_candidates)), int(0.3*len(sig_candidates))))
    elif mass_cut == "kmeans":
        sig, bkg = split_data_kmeans(data.loc[sig_candidates])
    elif mass_cut == "dbscan":
        sig, bkg = split_data_dbscan(data.loc[sig_candidates])
    else:
        sig, bkg = cf.split_data(data.loc[sig_candidates])
    sig = sig[sig.isin(sig_candidates)]
    # semitodo: if grey area added, change below line to remove both sig and grey
    bkg = bkg.join(data.index[~data.index.isin(sig_candidates)], how="outer").unique().sort_values()
    utils.log(f"Split data") if print_logs else None
    # Tree
    # .
    # ├── sig_candidates
    # │   ├── sig                       -> sig
    # │   ├── (grey area)
    # │   └── bkg                       -> bkg
    # ├── (semitodo: grey area here too)
    # └── non-sig_candidates            -> bkg

    cf.plot_stuff(data, data_pis, sig, bkg, mode=folder_name, verbose=print_logs)
    utils.log(f"Plotted data") if print_logs else None

    return data, data_pis


if __name__ == '__main__':
    @utils.alert
    def mainloop():
        for misc_cut in ["auto", "manual", None, ]:
            # for mass_cut in ["gmm", "ellipse", "svm", None]:
            for mass_cut in [ None, "ellipse", "kmeans", "gmm", "dbscan", ]:
                if mass_cut == 'dbscan':
                    continue
                if "data" not in locals():
                    data, data_pis = main(misc_cut, mass_cut)
                else:
                    main(misc_cut, mass_cut, data=data, data_pis=data_pis)
    mainloop()
