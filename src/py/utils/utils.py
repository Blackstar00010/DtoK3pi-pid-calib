import os
import sys
import platform
import pandas as pd
import numpy as np
import time
import uproot
from src.py.utils import config
if 'univ' not in os.path.abspath(__file__):
    from playsound3 import playsound
    from playsound3.playsound3 import PlaysoundException
else:
    from src.py.utils.fallbacks import unable2playsound as playsound
    from src.py.utils.fallbacks import MyPlaysoundException as PlaysoundException
if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from src.py.utils.fallbacks import deprecated


if platform.system() == 'Windows':
    raise NotImplementedError("THIS PROJECT IS NOT MENT TO RUN ON WINDOWS. USE OTHER OS OR INSTALL WSL")


def listdir(target_dir: str, show_hidden: bool = False, sort: bool = True, end: str | list = None,
            recursive=False) -> list:
    """
    List all files in a directory. Similar to `os.listdir`, but with more options.
    Args:
        target_dir: str. Directory to list files in.
        show_hidden: bool. If True, hidden files are included. Default is False.
        sort: bool. If True, files are sorted. Default is True.
        end: str or list. If provided, only files with the given extension(s) are included. Default is None.
        recursive: bool. If True, all files in subdirectories are included. Default is False.

    Returns:
        list. List of files in the directory.
    """
    target_dir = target_dir.rstrip('/')
    if recursive:
        ret = []
        for root, dirs, files in os.walk(target_dir, followlinks=True):
            ret += [f'{root}/{filename}' for filename in files]
            ret += [f'{root}/{dirname}' for dirname in dirs]
        ret = [item.replace(target_dir + '/', '') for item in ret]
    else:
        ret = os.listdir(target_dir)
    ret.sort() if sort else None
    if not show_hidden:
        ret = [item for item in ret if '/.' not in item or '/__' not in item]
        ret = [item for item in ret if not(item.startswith('.') and (not item.startswith('./')))]
    if type(end) == str:
        ret = [item for item in ret if item.endswith(end)]
    elif type(end) == list:
        end = ['.' + item if item.count('.') == 0 else item for item in end]
        ret = [item for item in ret if any([item.endswith(item) for item in end])]
    return ret


def export(df: pd.DataFrame, filename: str, filetype='auto') -> None:
    """
    Exports the file as a `.root` or `.csv` file, depending on the `format` argument.
    Args:
        df: pd.DataFrame. Data to export.
        filename: str. Path to save file. Extension will be ignored and set by `filetype`.
        filetype: str. If `auto`, export format is determined by the cpu architecture.

    Returns:
        None
    """
    filetype = filetype.lower().strip()
    if filetype not in ['auto', 'csv', 'root']:
        raise ValueError(f"Argument `filetype` of `export()` should be either `auto`, `csv`, or `root`. Received: {filetype}")
    filetype = config.better_ext if filetype == 'auto' else filetype
    filename = filename.replace('.csv', '').replace('.root', '') + '.' + filetype
    file_size = df.memory_usage(deep=True).sum() / 1e6
    # approximate; 6540.19MB df -> 4849.30MB .root file -> 340 sec
    eta = file_size /6540.19 * 340
    log(f"Exporting {filename} ({file_size:.2f} MB) in {eta:.2f} seconds")
    if filetype == 'csv':
        df.to_csv(filename, index=False)
    else:
        to_root(df, filename)
    log(f'File saved to {filename}')


def to_root(df: pd.DataFrame, filename: str, treename: str='DecayTree') -> None:
    """
    Save data to a root file
    Args:
        df: pd.DataFrame. Data to save.
        filename: str. Path to save file. If no extension is provided, '.root' is appended.
        treename: str. Name of the tree. Default is 'DecayTree'.
    """
    if filename.count('.') == 0:
        filename += '.root'
    with uproot.recreate(filename) as f:
        f[treename] = df.to_dict(orient='list')


def load(filename: str, drop_awk=True, nrows=None) -> pd.DataFrame:
    """
    Load data from root or csv file. If file not found, search for it in the data directory.
    Args:
        filename: str. Path to file. If no extension is provided, '.root' is assumed as it is faster.
        drop_awk: bool. If True, awkward arrays are dropped instead of converted to list. Default is True.
        nrows: int. Number of rows to read. Default is None, i.e. read all rows.

    Returns:
        pd.DataFrame. Data from root or csv file. Columns have been converted to list if they were awkward arrays.
    """
    if filename.count('.') == 0:
        filename += '.' + config.better_ext
    if filename.endswith('.root'):
        data = _load_root(filename, drop_awk)
        data = data.head(nrows) if nrows is not None else data
    elif filename.endswith('.csv'):
        data = _load_csv(filename, drop_awk, nrows)
    else:
        raise ValueError(f"File extension must be '.root' or '.csv', but got {filename}")
    log(f"Loaded {filename}")
    return data


def _load_root(filename: str, drop_awk=True) -> pd.DataFrame:
    """
    Load data from root file. If file not found, search for it in the data directory.
    Args:
        filename: str. Path to file. If no extension is provided, '.root' is appended.
        drop_awk: bool. If True, awkward arrays are dropped instead of converted to list. Default is True.

    Returns:
        pd.DataFrame. Data from root file. Columns have been converted to list if they were awkward arrays.
    """
    if os.path.exists(filename):
        log(f"Loading {filename}")
        data = uproot.open(filename)
    else:
        log(f"File {filename} not found. Searching for it...")
        candidates = listdir(config.data_dir, end=filename, recursive=True)
        candidates = [item for item in candidates if f'/{os.path.basename(config.metadata_dir)}/' not in item]  # filter head files
        candidates = [item for item in candidates if '.bak' not in item]

        # look for exact match
        exact_candidates = [item for item in candidates if item.split('/')[-1] == filename]
        if len(exact_candidates) > 0:
            candidates = exact_candidates

        if len(candidates) == 0:
            raise FileNotFoundError(f"File {filename} not found.")
        elif len(candidates) > 1:
            raise ValueError(f"Multiple candidate files found: {candidates}")
        else:
            filename = f"{config.data_dir}/{candidates[0]}"
        log(f"Found file at {filename}")
        data = uproot.open(filename)

    try:
        data = data["DecayTree"]
    except KeyError:  # mc data
        data = data.get("DstToD0Pi_D0ToKPiPiPi")["DecayTree"]
    data = data.arrays(filter_name=["*"], library="pd")

    if pd.Series(data.dtypes == "awkward").any():
        awkcols = data.columns[data.dtypes == "awkward"]
        if drop_awk:
            data = data.drop(columns=awkcols)
        else:
            for col in awkcols:
                # 1. replace all-[]-cols to all-""-cols
                if (data[col].apply(len) == 0).all():
                    data[col] = data[col].apply(lambda x: "")
                # 2. convert to list
                else:
                    data[col] = data[col].apply(lambda x: list(x))

    return data


def _load_csv(filename: str, drop_awk=True, nrows=None) -> pd.DataFrame:
    """
    Load data from csv file. If the file does not exist, first check if it is in other directory and if not found, it tries converting from the corresponding root file then saved.
    Args:
        filename: str. Path to file. If no extension is provided, '.csv' is appended.
        drop_awk: bool. Because all csvs are already converted, this option drops all list columns. Default is True.
        nrows: int. Number of rows to read. Default is None, i.e. read all rows.

    Returns:
        pd.DataFrame. Data from csv file.
    """
    if not os.path.exists(filename):
        log(f"File {filename} not found. Searching for it...")
        candidates = listdir(config.data_dir, end=filename, recursive=True)
        if len(candidates) == 0:
            candidates += listdir(config.data_dir, end=filename.replace('.csv', '.root'), recursive=True)
        header_dir_basename = os.path.basename(config.metadata_dir)
        candidates = [item for item in candidates 
                      if f'/{header_dir_basename}/' not in item 
                      and not item.startswith(f'{header_dir_basename}/')]  # filter head files
        candidates = [item for item in candidates if '.bak' not in item]

        # look for exact match
        exact_candidates = [item for item in candidates
                            if item.split('/')[-1] in [filename, filename.replace('.csv', '.root')]]
        if len(exact_candidates) > 0:
            candidates = exact_candidates

        if len(candidates) == 0:
            raise FileNotFoundError(f"File {filename} not found.")
        elif len(candidates) > 1:
            raise ValueError(f"Multiple candidate files found: {candidates}")
        else:
            filename = f"{config.data_dir}/{candidates[0]}"
        log(f"Found file at {filename}")

        if filename.endswith('.root'):
            data = _load_root(filename)
            data = data.head(nrows) if nrows is not None else data
            data.to_csv(filename.replace('.root', '.csv'), index=False)
        else:
            log(f"Loading {filename}")
            data = pd.read_csv(filename, nrows=nrows)

    else:
        log(f"Loading {filename}")
        data = pd.read_csv(filename, nrows=nrows)

    if drop_awk:
        cols = data.columns[data.dtypes == "object"]
        should_drop = pd.Series(cols).apply(lambda col: type(data.loc[0, col]) in [list, str])
        data = data.drop(columns=cols[should_drop])
    return data


def save_sample(target_dir=config.data_dir):
    """
    Save first 10 rows of all root files in `target_dir` to csv under `metadata_dir` specified in `config.py`.
    Returns:
        None
    """
    log("Saving head of all root files to csv")

    dones = []
    for root, dirs, files in os.walk(target_dir):
        if ('_metadata' in root) or ('.bak' in root):
            continue
        for file in files:
            if not file.endswith('.root'):
                continue
            if file in dones:
                raise FileExistsError(f"Files with same names found: {file}")
            data = load(f'{root}/{file}')
            data = pd.concat([data.head(10), data.tail(10)])
            data.to_csv(f'{config.metadata_dir}/{file.replace(".root", ".csv")}', index=False)
            log(f"Saved head of {file}")
            dones.append(file)


def useful_cols_plot(cols: list) -> list:
    """
    Returns the useful columns from the given list of columns.
    Args:
        cols: list. List of columns.

    Returns:
        list. Useful columns.
    """
    ret = ["K_P", "K_TRACK_P", "pi1_P", "pi2_P", "pi3_P"]
    keywords = ["_ETA", "BPVIP", "MINIP"]
    suffixes = ["CHI2", "_M"]
    [ret.append(col) for col in cols 
     if any([keyword in col for keyword in keywords]) 
     or any([col.endswith(suffix) for suffix in suffixes])]
    return ret


def useful_cols_classify(cols: list) -> tuple[list, list]:
    """
    Returns the useful columns from the given list of columns.
    Args:
        cols: list. List of columns.

    Returns:
        tuple[list, list]. Good columns and bad columns.
    """
    good = []
    bad = []

    dynamics_suffixes = ["_ETA", "_P", "_PT", "_PX", "_PY", "_PZ", '_ENERGY']
    for col in cols:
        if col.find("_") == -1:  # non particle-related columns
            bad.append(col)
        elif "_TRUE" in col or "_PID" in col or "_PROBNN_" in col:  # answer data; explicitly mentioned
            bad.append(col)
        # elif (col == "D_M") or (col == "Dst_M") or (col.startswith("D_WM")):  # explicitly mentioned
        elif col.endswith('_M') or col.startswith("D_WM"):  # explicitly mentioned
            bad.append(col)
        elif any([col.endswith(suffix) for suffix in dynamics_suffixes]):  # explicitly mentioned
            bad.append(col)
        elif col.endswith("PROB") or col.endswith("ISMUON"):
            bad.append(col)
        elif col == 'og_index' or col.startswith('sig'):
            bad.append(col)
        else:
            good.append(col)
    return good, bad


def useful_cols_fit(cols: list) -> list:
    """
    Returns the useful columns from the given list of columns.
    Args:
        cols: list. List of columns.

    Returns:
        list. Useful columns.
    """
    ret = []
    keywords = ["ETA", "P", "M", "PID_K"]
    particles = ['delta', 'D', 'K', 'pi1', 'pi2', 'pi3']
    for particle in particles:
        for keyword in keywords:
            if f'{particle}_{keyword}' in cols:
                ret.append(f'{particle}_{keyword}')
    return ret


def log(message: str, end="\n", add_time=True):
    """
    Log message to console
    Args:
        message: str. Message to log
        end: str. End character. Default is '\n'
        add_time: bool. If True, add time stamp to message. Default is True.
    """
    if add_time:
        now = time.strftime("%Y-%m-%d %H:%M:%S") + ": "
    else:
        now = ' ' * 21
    print(f"{now}{message}", end=end)
    # probably import logging and log to file?


def loginput(question: str, end="", add_time=True) -> str:
    """
    Log a question and return the input
    Args:
        question: str. Question to ask
        end: str. End character. Default is ''
        add_time: bool. If True, add time stamp to message. Default is True.

    Returns:
        str. Input
    """
    question = question.strip() + "\t"
    log(question, end=end, add_time=add_time)
    beep('ask')
    return input()


def logwarn(message: str, end="\n", add_time=True) -> None:
    """
    Log a warning message
    Args:
        message: str. Warning message
        end: str. End character. Default is '\n'
        add_time: bool. If True, add time stamp to message. Default is True.
    """
    log(f"WARNING: {message}", end=end, add_time=add_time)
    beep('warn')


@deprecated(reason="This function is deprecated. Use the method in `dataprep.py` instead.")
def find_center(series: pd.Series) -> float:
    """
    Using DBSCAN, find the center of the data.
    Args:
        series: pd.Series. Data to find the center of.

    Returns:
        float. Center of the data.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    X = series.values.reshape(-1, 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    db = DBSCAN(eps=0.05, min_samples=int(0.05 * len(X)), n_jobs=-1).fit(X)
    ret = X[db.core_sample_indices_].mean()
    return scaler.inverse_transform(np.array([ret]).reshape(-1, 1))[0][0]


def find_massmean(data: pd.DataFrame) -> tuple[float, float]:
    """
    Find the mean of the D0 mass and D0-D* mass.
    Args:
        data: pd.DataFrame. Data to find the mean of.

    Returns:
        tuple[float, float]. Mean of the D0 mass and D0-D* mass.
    """
    if len(data) > 200000:
        data = data.sample(200000, random_state=42)
    dmmean = find_center(data['D_M'])
    deltamean = find_center(data['delta_M'])
    return dmmean, deltamean


def beep(mode="good") -> None:
    """
    Makes a beep audio
    """
    if 'univ' in os.path.abspath(__file__):
        log("Beep failed. Check the audio file or check if ffmpeg is installed.")
        return
    # good -> data/audio/nice.mp3
    # bad -> data/audio/no.mp3
    if mode == "good":  # e.g. function successfully completed
        filename = 'nice'
    elif mode == "bad":  # e.g. error raised
        filename = 'no'
    elif mode == "warn":  # e.g. warning raised
        # 440hz for 0.3 seconds
        # to generate, use `ffmpeg -f lavfi -i "sine=frequency=440:duration=0.3" -c:a libmp3lame -q:a 4 440.mp3`
        filename = '440'
    else:  # e.g. asking for input
        filename = 'damn'
    playsound(f'{config.audio_dir}/{filename}.mp3')


# decorators
def alert(func):
    def wrapper(*args, **kwargs):
        ret = None
        try:  # for errors in playing sound
            try:  # for errors in function
                ret = func(*args, **kwargs)
                log(f"{func.__name__}() executed successfully")
                beep("good")
            except Exception as e:
                log(f"Error in {func.__name__}(): {e}")
                beep("bad")
                raise e
        except PlaysoundException as e:
            log(f"Playsound failed: {e}")
        except Exception as e:
            raise e
        return ret

    return wrapper


def log_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        log(f"{func.__name__} took {time.time() - start:.2f} seconds")
        return ret

    return wrapper


if __name__ == "__main__":
    @alert
    def main():
        raise RuntimeError(f"`{__file__}` is not meant to be executed.")

    main()
