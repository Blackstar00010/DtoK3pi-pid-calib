import pandas as pd  # type: ignore
import numpy as np  # type: ignore
# from jointworoot import data_dir, load_root
import jointworoot

def main():
    df = jointworoot.load_root(jointworoot.data_dir + '/output/sWeights.root')

    picols = []
    for col in df.columns:
        if col.startswith('pi'):
            picols.append(col)
    print(df[picols].head())
    print()
    # sort pi columns by P
    key_cols = ['pi1_P', 'pi2_P', 'pi3_P']
    keys_subdf = df[key_cols].values
    momentum_ranks = np.argsort(keys_subdf, axis=1)
    df[['pi1_P', 'pi2_P', 'pi3_P']] = np.take_along_axis(keys_subdf, momentum_ranks, axis=1)
    
    objs = [col.replace('pi1_', '') for col in df.columns if col.startswith('pi1') and col != 'pi1_P']
    for obj in objs:
        cols = [f'pi{i}_{obj}' for i in range(1, 4)]
        subdf = df[cols].values
        sorted_subdf = np.take_along_axis(subdf, momentum_ranks, axis=1)
        df[cols] = sorted_subdf
    # now pi1_P <= pi2_P <= pi3_P but the other columns are not sorted
    # other columns' corresponding 
    print(df[picols].head())

    jointworoot.to_root(df, jointworoot.data_dir + '/output/sWeights_sorted.root')
    


if __name__ == "__main__":
    main()