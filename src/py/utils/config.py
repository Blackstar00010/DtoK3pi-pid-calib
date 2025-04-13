import os
# import platform


# for arm, root is ~10 times faster; for cisc, root is ~1000 times slower
better_ext = 'root' # if platform.processor() == 'arm' else 'csv'  # we are not using csv
cut_method = 'box'  # 'ellipse' or 'box'


utils_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.dirname(utils_dir)
src_dir = os.path.dirname(py_dir)
proj_dir = os.path.dirname(src_dir)

# subdirectories of proj_dir
data_dir = f"{proj_dir}/data"
models_dir = f"{proj_dir}/models"
plots_dir = f"{proj_dir}/plots"
logs_dir = f"{proj_dir}/logs"

# subdirectories of data_dir
metadata_dir = f"{data_dir}/_samples"  # heads(=first few lines) of data files
audio_dir = f"{data_dir}/audio"        # audio files for alerting function status
input_dir = f"{data_dir}/input"        # input data(actual data, mc data)
ouptut_dir = f"{data_dir}/output"      # output data(sweights for now)
probas_dir = f"{data_dir}/score"       # scores of classifiers
temp_dir = f"{data_dir}/temp"          # temporary files

# subdirectories of input_dir
mc_dir = f"{input_dir}/mc"        # real mc
synmc_dir = f"{input_dir}/synmc"  # synthetic mc

# important files in input_dir; extensions are added while reading and writing files in `common.py`
long_file = f"{input_dir}/long"
long_core_file = f"{input_dir}/long_core"
short_file = f"{input_dir}/short"
wide_file = f"{input_dir}/wide"

# important files in score dir
long_score_file = f"{probas_dir}/proba_long"
short_score_file = f"{probas_dir}/proba_short"
tt_score_file = f"{probas_dir}/proba_tt"
long_with_score_file = f"{probas_dir}/long_with_prob"
short_with_score_file = f"{probas_dir}/short_with_prob"

# target files
# TODO: change mode into boolean and rename to something like `is_long`
mode = 'long'  # 'long' or 'short'
if mode == 'long':
    target_file = long_file
    target_core_file = long_core_file
    target_score_file = long_score_file
    target_with_score_file = long_with_score_file
else:
    target_file = short_file
    target_core_file = short_file
    target_score_file = short_score_file
    target_with_score_file = short_with_score_file

# subdirectories of models_dir
generators_dir = f"{models_dir}/generators"    # synthetic data generators
scalers_dir = f"{models_dir}/scalers"          # scalers for data preprocessing
classifiers_dir = f"{models_dir}/classifiers"  # classifiers for signal-background classification

plot_dirs = {
    0: f"{plots_dir}/0_synth",
    1: f"{plots_dir}/1_prac_py",
    2: f"{plots_dir}/2_dists_py",
    3: f"{plots_dir}/3_classify_py",
    4: f'{plots_dir}/4_massfit',
}

# check if directories exist, if not create them
for adir in [v for k, v in locals().items() if k.endswith("_dir")]:
    if os.path.exists(adir) and os.path.isdir(adir):
        continue
    else:
        os.makedirs(adir)
for adir in plot_dirs.values():
    if os.path.exists(adir) and os.path.isdir(adir):
        continue
    else:
        os.makedirs(adir)


def plotdir1(mode: str, method: str=None):
    if mode not in ['general', 'plain'] and method is None:
        raise ValueError("method must be specified if mode is not 'general' or 'plain'")
    if mode == 'general':
        return f"{plot_dirs[1]}/_general"
    elif mode == 'plain':
        return f"{plot_dirs[1]}/plain"
    return f"{plot_dirs[1]}/{mode}_{method}"


def plotdir3(model: str=None, ratio: int|str=None, join_type: str=None, plot_type: str='dists'):
    if model is None and ratio is None and join_type is None:
        if plot_type == "dists":
            return f"{plot_dirs[3]}/_dists"
        else:
            raise ValueError("model, ratio, join_type must be specified if plot_type is not 'dists'")
    if model is None:
        raise ValueError("model must be specified if plot_type is not 'dists'")
    if plot_type == 'dists':
        raise ValueError("plot_type cannot be dists if model is specified")
    ret = f"{plot_dirs[3]}/{model}"
    if ratio is not None:
        ret += f"_{ratio}"
    if join_type is not None:
        ret += f"_{join_type}"
    if plot_type in ['dm', 'deltam']:
        return f"{ret}/{plot_type}"
    else:
        return f"{ret}/{plot_type}.png"

def plotdir4():
    return plot_dirs[4]

def mc_file(mode: str | None = None, ind: int | None = None) -> str:
    """
    Returns the raw MC file path based on the mode and index. If mode and index are not specified, returns the full MC file path.
    Args:
        mode: 'up' or 'down'
        ind: 1, 2, or 3

    Returns:
        str: path to the MC file
    """
    if (not mode) and (not ind):
        return mc_dir + "/mc.root"
    if mode is None or ind is None:
        raise ValueError("Both mode and ind must be specified")
    if mode.lower().strip() in ['d', 'down']:
        mode = 'd'
    elif mode.lower().strip() in ['u', 'up']:
        mode = 'u'
    else:
        raise ValueError("mode must be either 'up' or 'down'")
    if ind not in range(1, 4):
        raise ValueError("ind must be 1, 2, or 3")
    return f"{mc_dir}/m{mode}{ind}.root"


def mc_proced_file(ratio: int | str | None = None) -> str:
    if isinstance(ratio, str) and ratio.isnumeric():
        ratio = int(ratio)
    if ratio == 'all' or ratio is None or ratio > 100:
        return f"{mc_dir}/mc_proced_all.{better_ext}"
    return f"{mc_dir}/mc_proced_{ratio}.{better_ext}"


if __name__ == "__main__":
    from playsound3 import playsound
    playsound(audio_dir + "/no.mp3")
    raise RuntimeError("Module config.py is not supposed to be executed")
