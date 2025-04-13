import os
import uproot
import pandas as pd


current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '..', '..', '..', 'data')


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

def load_root(filename: str, drop_awk=True) -> pd.DataFrame:
    """
    Load data from root file. If file not found, search for it in the data directory.
    Args:
        filename: str. Path to file. If no extension is provided, '.root' is appended.
        drop_awk: bool. If True, awkward arrays are dropped instead of converted to list. Default is True.

    Returns:
        pd.DataFrame. Data from root file. Columns have been converted to list if they were awkward arrays.
    """
    try:
        data = uproot.open(filename)
    except FileNotFoundError:
        print(f"File {filename} not found. Searching for it...")
        candidates = os.listdir(data_dir, end=filename, recursive=True)
        candidates = [item for item in candidates if '/_metadata/' not in item]
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
            filename = f"{data_dir}/{candidates[0]}"
        print(f"Found file at {filename}")
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

def main():
    """
    Joins two root files(long.root and proba_long.root) and saves the result to a new root file(long_with_prob.root).
    """
    file1 = os.path.join(data_dir, 'input', 'long_core.root')
    file2 = os.path.join(data_dir, 'score', 'proba_long.root')
    df = load_root(file1)[[
        "D_M", "delta_M", 
        "K_P", "K_PT", "K_ETA", "K_PID_K", 
        "pi1_P", "pi1_PT", "pi1_ETA", "pi1_PID_K",
        "pi2_P", "pi2_PT", "pi2_ETA", "pi2_PID_K",
        "pi3_P", "pi3_PT", "pi3_ETA", "pi3_PID_K",
    ]]
    df2 = load_root(file2)['bdt_all']
    df['bdt_all'] = df2
    df = df.loc[(df['D_M'] > 1810) & (df['D_M'] < 1920) & (df['delta_M'] > 139.5) & (df['delta_M'] < 152)]
    to_root(df, data_dir + '/score/long_with_prob.root')


if __name__ == '__main__':
    # main()
    print("joining data with probs is moved to files in src/py/classify/")
    