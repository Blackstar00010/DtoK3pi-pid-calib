import os
import pandas as pd
from src.py.utils import utils, config, consts

# file for preparing data, especially for mc data


preloaded_full = pd.DataFrame()  # to avoid loading full data multiple times
core_saved = False  # to avoid saving core data multiple times


def concat_mc() -> pd.DataFrame:
    """
    Concatenate all mc files into one DataFrame
    Returns:
        pd.DataFrame. Concatenated data from all mc files.
    """
    mc_files = utils.listdir(config.mc_dir, show_hidden=False, sort=True, end=config.better_ext)
    dfs = []
    for file in mc_files:
        if 'proced' in file:
            continue
        if file == f'mc.{config.better_ext}':
            continue
        data = utils.load(f"{config.mc_dir}/{file}")
        data['file'] = file.lower().replace(f'.{config.better_ext}', '')
        dfs.append(data)
    mc_data = pd.concat(dfs)
    return mc_data


def load_mc(save=True, verbose=False) -> pd.DataFrame:
    """
    Load mc data in the `mc_dir` specified in `config.py`. If not saved, save it first as `mc.{utils.better_ext}`.
    Returns:
        pd.DataFrame. Loaded data
    """
    # mc_filename = f'{config.mc_dir}/mc.{utils.better_ext}'
    mc_filename = config.mc_file()
    if not os.path.exists(mc_filename):
        utils.log(f"{mc_filename} does not exist. Concatenating...") if verbose else None
        mc_data = concat_mc()
        utils.log("MC data concatenated") if verbose else None
        utils.export(mc_data, mc_filename, config.better_ext) if save else None
        utils.log("MC data saved") if verbose and save else None
        return mc_data
    utils.log("Concatenated MC data found. Loading...") if verbose else None
    return utils.load(mc_filename)


def filter_mc(mc_data: pd.DataFrame, apply_trueid=True, apply_mothertrueid=True, apply_bkgcat=False,
              cut_cols=True) -> pd.DataFrame:
    """
    Filters rows and columns of mc data by trueid and bkgcat
    Args:
        mc_data: pd.DataFrame. Data to filter
        apply_trueid: bool. Whether to filter by trueid. Default: True
        apply_mothertrueid: bool. Whether to filter by mothertrueid. Default: True
        apply_bkgcat: bool. Whether to filter by bkgcat. Default: False
        cut_cols: bool. Whether to cut columns. Default: True

    Returns:
        pd.DataFrame. Filtered data
    """
    if apply_trueid:
        pid = {"K": 321, "pis": 211, "pi1": 211, "pi2": 211, "pi3": 211, "D": 421, "Dst": 413}
        trueid = [(mc_data[f"{particle}_TRUEID"].abs() == pid[particle]) for particle in pid]
        trueid = pd.concat(trueid, axis=1).all(axis=1)
        mc_data = mc_data.loc[trueid]
    if apply_mothertrueid:
        mid = {"K": 421, "pi1": 421, "pi2": 421, "pi3": 421, "pis": 413, "D": 413}
        mothertrueid = [(mc_data[f"{particle}_MC_MOTHER_ID"].abs() == mid[particle]) for particle in mid]
        mothertrueid = pd.concat(mothertrueid, axis=1).all(axis=1)
        mc_data = mc_data.loc[mothertrueid]
    if apply_bkgcat:
        utils.logwarn("`apply_bkgcat` is deprecated. Skipping...")
        # particles = ["K", "pis", "pi1", "pi2", "pi3", "D", "Dst"]
        # bad_list = [20, 40, 50, 60]
        # bkgcat = [(~mc_data[f"{particle}_BKGCAT"].isin(bad_list)) for particle in particles]
        # bkgcat = pd.concat(bkgcat, axis=1).all(axis=1)
        # mc_data = mc_data.loc[bkgcat]
    mc_data = mc_data.reset_index(drop=True)

    # filtering columns
    if cut_cols:
        raise ValueError(
            "`cut_cols` option for `filter_mc` function is deprecated. Columns should be selected right before training.")
        # good_cols, bad_cols = utils.useful_cols_ml(mc_data.columns)
        # utils.log(f"Bad columns: {bad_cols}")
        # if 'sig' not in good_cols and 'sig' in mc_data.columns:
        #     good_cols.append('sig')
        # mc_data = mc_data[good_cols]
    mc_data = mc_data.dropna(subset=['D_BPVFDCHI2', 'D_BPVIPCHI2', 'Dst_BPVFDCHI2', 'Dst_BPVIPCHI2'],
                             axis=0)  # FD&IP contains very few NaNs (<1%)
    # mc_data = mc_data.dropna(axis=1)

    return mc_data


def find_inout(data: pd.DataFrame) -> pd.Series:
    """
    Find in and out data by applying cuts on D_M and delta_M, and return a boolean series
    Args:
        data: pd.DataFrame. Data to filter. Must contain D_M and delta_M columns

    Returns:
        pd.Series. Boolean series indicating in and out data (True for in, False for out)
    """
    if config.mode == 'ellipse':
        # ellipse
        x = (data['D_M'] - consts.mc_stats.dmmean) / consts.mc_stats.dmstd  # ~N(0, 1)
        y = (data['delta_M'] - consts.mc_stats.deltammean) / consts.mc_stats.deltamstd  # ~N(0, 1)
        ret = (x ** 2) + (y ** 2) <= consts.cut_std ** 2
    else:  # box
        x = data['D_M']
        y = data['delta_M']
        ret = (x >= consts.mc_stats.dmmean - consts.mc_stats.dmradius)
        ret &= (x <= consts.mc_stats.dmmean + consts.mc_stats.dmradius)
        ret &= (y >= consts.mc_stats.deltammean - consts.mc_stats.deltamradius)
        ret &= (y <= consts.mc_stats.deltammean + consts.mc_stats.deltamradius)
    return ret


def filter_bkg(full_data: pd.DataFrame, target_count: int = -1, verbose=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract background data from full data by selected data farthest from the mean of D_M and delta_M
    Args:
        full_data: pd.DataFrame. Data to filter
        target_count: int. Target count of signal data. If -1, use all data with . Default: -1
        verbose: bool. Whether to print messages. Default: False

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]. Core and sideband data
    """
    # TODO: save core here and only return the sideband data
    inouts = find_inout(full_data[['D_M', 'delta_M']])
    core, sideband = full_data[inouts], full_data[~inouts]
    if 0 < target_count < len(sideband):
        sideband = sideband.sample(n=target_count, random_state=42)
    utils.log("Background data filtered") if verbose else None
    return core, sideband


def concat_mc_bkg(save=True, verbose=False, bkgsig_ratio: int | str = 2) -> pd.DataFrame:
    """
    Concatenate mc and full data into one DataFrame with a given ratio of background to signal
    Args:
        save: bool. Whether to save the data. Default: True
        verbose: bool. Whether to print messages. Default: False
        bkgsig_ratio: int or str. Ratio of background to signal. If "all", use all sideband data. Default: 2

    Returns:
        pd.DataFrame. Concatenated data
    """
    if bkgsig_ratio not in consts.train_bkgsig_ratios:
        raise ValueError(f"bkgsig_ratio should be one of {consts.train_bkgsig_ratios}, not {bkgsig_ratio}")
    # load mc data and filter columns
    mc_data = load_mc(save=save, verbose=verbose)
    mc_data['og_index'] = mc_data.index
    mc_data = filter_mc(mc_data, apply_trueid=True, apply_bkgcat=False, cut_cols=False)
    pointless_cols = mc_data.columns[mc_data.nunique() == 1]
    mc_data = mc_data.drop(pointless_cols, axis=1)
    mc_data['sig'] = True
    mc_data['sig_int'] = 1
    mc_data['origin'] = 'mc'
    utils.log("MC data filtered") if verbose else None

    # calculate mean and std of D_M and delta_M
    mc_data['delta_M'] = abs(mc_data['Dst_M'] - mc_data['D_M'])
    dmmean = mc_data['D_M'].mean()
    deltammean = mc_data['delta_M'].mean()
    dmstd = mc_data['D_M'].std()
    deltamstd = mc_data['delta_M'].std()
    utils.log(f"Mean of D_M: {dmmean}, delta_M: {deltammean}") if verbose else None
    utils.log(f"Std of D_M: {dmstd}, delta_M: {deltamstd}") if verbose else None

    # load full data and filter columns
    global preloaded_full, core_saved
    full_data = utils.load(config.target_file) if preloaded_full.empty else preloaded_full.copy()
    full_data['og_index'] = full_data.index
    must_have_cols = ['D_M', 'Dst_M', 'D_BPVFDCHI2', 'Dst_BPVFDCHI2', 'D_BPVIPCHI2', 'Dst_BPVIPCHI2']
    must_have_cols += utils.useful_cols_classify(full_data.columns)[0]
    must_have_cols = list(set(must_have_cols))
    full_data = full_data.dropna(subset=must_have_cols, axis=0)
    utils.log("Full data loaded") if verbose else None
    target_count = -1 if bkgsig_ratio == 'all' else len(mc_data) * bkgsig_ratio
    core, full_data = filter_bkg(full_data, target_count=target_count, verbose=verbose)
    # core.to_csv(config.target_core_file.replace('.root', '.csv'), index=False)
    if not core_saved:
        utils.export(core, config.target_core_file, config.better_ext)
        core_saved = True
    full_data['sig'] = False
    full_data['sig_int'] = 0
    full_data['origin'] = config.target_file.split('/')[-1].split('.')[0]
    utils.log("Full data filtered") if verbose else None

    # concatenate data vertically
    full_data_cols = [col for col in full_data.columns if col in mc_data.columns]  # for column ordering
    data = pd.concat([mc_data, full_data], axis=0, join='inner').dropna(axis=1)
    data = data[[col for col in full_data_cols if col in data.columns]]
    utils.log("Data concatenated") if verbose else None

    # save data
    if save:
        # filepath = f'{config.mc_dir}/mc_proced_{bkgsig_ratio}.{utils.better_ext}'
        filepath = config.mc_proced_file(ratio=bkgsig_ratio)
        utils.to_root(data, filepath)
        head_filepath = f'{config.metadata_dir}/mc_proced_{bkgsig_ratio}.csv'
        data.head(10).to_csv(head_filepath, index=False)
        utils.log(f"MC data of ratio {bkgsig_ratio} ( shape: {data.shape} ) saved as {filepath}") if verbose else None
    return data


def load_proced_mc(bkgsig_ratio: int | str = 2, verbose=False) -> pd.DataFrame:
    """
    Load processed mc data with a given ratio of background to signal

    Args:
        bkgsig_ratio: int or str. Ratio of background to signal. If "all", use all sideband data. Default: 2
        verbose: bool. Whether to print messages. Default: False

    Returns:
        pd.DataFrame. Loaded data
    """
    # file_to_find = f'{config.mc_dir}/mc_proced_{bkgsig_ratio}.{config.better_ext}'
    file_to_find = config.mc_proced_file(ratio=bkgsig_ratio)
    if not os.path.exists(file_to_find):
        utils.log(f"{file_to_find} does not exist. Concatenating...") if verbose else None
        return concat_mc_bkg(save=True, verbose=verbose, bkgsig_ratio=bkgsig_ratio)
    utils.log(f"Concatenated MC data found. Loading...") if verbose else None
    return utils.load(file_to_find)


def check_proced_mc(delete_existing: bool = False, verbose: bool = False) -> None:
    for ratio in consts.train_bkgsig_ratios:
        filename = config.mc_proced_file(ratio=ratio)
        if os.path.exists(filename):
            if delete_existing:
                utils.log(f"{filename} already exists. Removing...") if verbose else None
                os.remove(filename) if os.path.exists(filename) else None
            else:
                utils.log(f"{filename} already exists. Skipping...") if verbose else None
                continue
        load_proced_mc(bkgsig_ratio=ratio, verbose=verbose)
    utils.save_sample(config.input_dir)  # just in case


@utils.alert
def main():
    script_filename = "`" + os.path.basename(__file__) + "`"
    if not __name__ == "__main__":
        raise ValueError(f"main function in {script_filename} should be run directly")
    mcproced_filename = config.mc_proced_file(1).replace("1." + config.better_ext, "").replace(config.proj_dir, "...")
    utils.log(f'{script_filename} is used to prepare data for training, meaning running this script will reset `{mcproced_filename}*` files. ')
    res = utils.loginput(f'Do you want to continue? (y/n): ', add_time=False)
    if res != 'y':
        utils.log('Exiting...')
        return
    global preloaded_full
    preloaded_full = utils.load(config.target_file)
    check_proced_mc(True, True)
    utils.log(f"Finished running {script_filename}")


if __name__ == "__main__":
    main()
