import os
import pickle
# import numpy as np 

# this file reads the pickled histograms of Kpi to use for comparisons


def main():
    data_dir = '<path_to_your_data_directory>'  # e.g. '/home/user/data/'
    if data_dir == '<path_to_your_data_directory>':
        raise ValueError("Please set the data_dir variable to your data directory.")
    files = os.listdir(data_dir)
    files = [f for f in files if 'effhists' in f and 'block8' in f and 'up' in f]
    files.sort()
    keyword = '-P.ETA.pkl'
    # keyword = '-nPVs.pkl'  # no pi data for this keyword
    target = [f for f in files if f.endswith(keyword)]
    k_npv_data = {}
    pi_npv_data = {}
    for file in target:
        with open(data_dir + file, 'rb') as f:
            # print(file)
            hists = pickle.load(f)
            cut = float(file.split('>')[-1].replace(keyword, ''))
            if 'K-DLLK' in file:
                k_npv_data[cut] = hists.values().sum()
                continue
            elif 'Pi-DLLK' in file:
                pi_npv_data[cut] = hists.values().sum()
    k_npv_data = dict(sorted(k_npv_data.items()))
    k_max = max(k_npv_data.values())
    for key in k_npv_data:
        k_npv_data[key] = k_npv_data[key] / k_max
    pi_npv_data = dict(sorted(pi_npv_data.items()))
    pi_max = max(pi_npv_data.values())
    for key in pi_npv_data:
        pi_npv_data[key] = pi_npv_data[key] / pi_max
    
    print("DLLK, K, pi")
    for key in k_npv_data:
        try:
            k_npv = k_npv_data[key]
        except KeyError:
            raise KeyError(f"key {key} not found in the kaon data (*{keyword})")
        try:
            pi_npv = pi_npv_data[key]
        except KeyError:
            raise KeyError(f"key {key} not found in the pion data (*{keyword})")
        print(f"{key}, {k_npv:.05f}, {pi_npv:.05f}")
    # to save as a csv, use `python roc.py > roc.csv`




if __name__ == '__main__':
    main()
