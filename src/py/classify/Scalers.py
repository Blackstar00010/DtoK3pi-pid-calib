import pandas as pd
import numpy as np
import os
from src.py.utils import utils, config
from src.py.prepare import dataprep


class Scaler:
    def __init__(self, scaler_type: str, reset: bool=False):
        self.vars, self.features, self.var_names = None, None, None
        self.scaler_type = scaler_type

        if not os.path.exists(f'{config.scalers_dir}/{self.scaler_type}.csv'):
            reset = True
        
        self.reset() if reset else self.load()

    def reset(self) -> None:
        """
        Reset the scaler instance
        Returns:
            None
        """
        self.vars = pd.DataFrame()  # columns = features, rows = var1, var2, ... where var1 = mean for standard scaler
        self.features = []  # self.vars.columns
        self.var_names = []  # self.vars.index

    def load(self) -> None:
        """
        Load the fitted scaler from a csv file named `{self.scaler_type}.csv` in scalers_dir specified in `config.py`
        Returns:
            None
        """
        self.vars = pd.read_csv(f'{config.scalers_dir}/{self.scaler_type}.csv', index_col=0)
        self.features = self.vars.columns
        self.var_names = self.vars.index

    def save(self) -> str:
        """
        Save the fitted scaler to a csv file named `{self.scaler_type}.csv` in scalers_dir specified in `config.py`
        Returns:
            str. Path of the saved file
        """
        if __name__ != '__main__':
            raise PermissionError("Saving scalers should only be done in `preprocessor.py` file.")
        self.vars.to_csv(f'{config.scalers_dir}/{self.scaler_type}.csv')
        return f'{config.scalers_dir}/{self.scaler_type}.csv'

    def fit(self, df: pd.DataFrame, append: bool=True, overwrite: bool=False) -> None:
        """
        Fit the scaler with the data.
        Args:
            df: pd.DataFrame. Data to fit the scaler
            append: bool. If True, append features not yet fitted.
            overwrite: bool. If True, overwrite the fitted features with the new values.

        Returns:
            None
        """
        df = df.astype(float)
        new_vars = self._get_new_vars(df)
        brandnew_cols = [col for col in new_vars.columns if col not in self.features]
        overwrite_cols = [col for col in new_vars.columns if col not in brandnew_cols]
        if append:
            self.vars = pd.concat([self.vars, new_vars[brandnew_cols]], axis=1)
        if len(overwrite_cols) > 0 and overwrite:
            self.vars.loc[:, overwrite_cols] = new_vars.loc[:, overwrite_cols]
        self.features = self.vars.columns
        self.var_names = self.vars.index

    def _get_new_vars(self, df: pd.DataFrame):
        """
        Calculate new values(mean & std for ``std``, for example) for all columns in `df`. Index are the names of the new values and columns are the features in `df`.
        Args:
            df: pd.DataFrame. Data to calculate new values

        Returns:
            pd.DataFrame. New values of the features with index 'mean' and 'std' for example
        """
        raise NotImplementedError(
            "Private functions of Scaler class should not be run. Make a child class's instance instead.")

    def transform(self, df: pd.DataFrame, fit_extra: bool=None, save_extra: bool=None) -> pd.DataFrame:
        """
        Transform the data using the fitted. Handles extra columns not in the fitted scaler.
        Args:
            df: pd.DataFrame. Data to transform
            fit_extra: bool. If True, fit the scaler with the extra columns. If False, skip the extra columns(not returned). If None, ask to fit the scaler with the extra columns.
            save_extra: bool. If True, save the fitted scaler. If False, do not save the fitted scaler. If None, ask to save the fitted scaler.

        Returns:
            pd.DataFrame. Transformed data
        """
        df = df.astype(float)
        good_cols = [col for col in df.columns if col in self.features]
        bad_cols = [col for col in df.columns if col not in self.features]
        good_features = [feature for feature in self.features if feature in df.columns]
        if len(bad_cols) > 0:
            utils.log(
                f"Columns not in the scaler {self.scaler_type} found: {bad_cols}")  # TODO: warning instead of log
            if fit_extra is None:
                fit_extra = utils.loginput("Fit the scaler with the extra columns? (Y/n) ")
                fit_extra = fit_extra.lower() != 'n'
            if fit_extra:
                self.fit(df[bad_cols], append=True)
                [good_cols.append(col) for col in bad_cols]
                [good_features.append(feature) for feature in bad_cols]
                utils.log("The scaler is fitted with the extra columns.")
                if save_extra is None:
                    save_extra = utils.loginput("Save the fitted scaler? (Y/n) ")
                    save_extra = save_extra.lower() != 'n'
                self.save() if save_extra else utils.log("The fitted scaler is not saved.")
            else:
                utils.log("Extra columns are skipped.")
        ret = self._tf(df[good_cols], good_features)
        ret = ret.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all').dropna()
        return ret[[col for col in df.columns if col in ret.columns]]  # for column order

    def _tf(self, df: pd.DataFrame, good_features: list | pd.Index) -> pd.DataFrame:
        """
        Transform the data using the fitted scaler. For internal use.
        Args:
            df: pd.DataFrame. Data to transform
            good_features: list | pd.Index. Features to transform

        Returns:
            pd.DataFrame. Transformed data
        """
        raise NotImplementedError(
            "Private functions of Scaler class should not be run. Make a child class's instance instead.")

    def untf(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the data using the fitted scaler.
        Args:
            df: pd.DataFrame. Data to inverse transform

        Returns:
            pd.DataFrame. Inverse transformed data
        """
        if "object" in df.dtypes.values:
            raise ValueError("DataFrame contains non-numeric columns")
        good_cols = [col for col in df.columns if col in self.features]
        bad_cols = [col for col in df.columns if col not in self.features]
        good_features = [feature for feature in self.features if feature in df.columns]
        if len(bad_cols) > 0:
            utils.log(f"WARNING: Columns not in the scaler {self.scaler_type} are skipped: {bad_cols}")
        ret = self._untf(df[good_cols], good_features)
        return ret[[col for col in df.columns if col in ret.columns]]

    def _untf(self, df: pd.DataFrame, good_features: list | pd.Index) -> pd.DataFrame:
        """
        Inverse transform the data using the fitted scaler. For internal use.
        Args:
            df: pd.DataFrame. Data to inverse transform
            good_features: list | pd.Index. Features to inverse transform

        Returns:
            pd.DataFrame. Inverse transformed data
        """
        raise NotImplementedError("Functions of Scaler class should not be run. Make a child class's instance instead.")


class StandardScaler(Scaler):
    """
    StandardScaler using mean (row 0) and std (row 1).

    Transform: x -> z = (x - mean) / std
    Inverse transform: z -> x = z * std + mean
    """
    def __init__(self, reset=False):
        super().__init__('std', reset)

    def _get_new_vars(self, df: pd.DataFrame):
        return pd.DataFrame([df.mean(), df.std()], index=pd.Index(['mean', 'std']))

    def _tf(self, df: pd.DataFrame, good_features: list | pd.Index):
        return (df - self.vars.loc['mean', good_features]) / self.vars.loc['std', good_features]

    def _untf(self, df: pd.DataFrame, good_features: list | pd.Index):
        return df * self.vars.loc['std', good_features] + self.vars.loc['mean', good_features]


class MinMaxScaler(Scaler):
    """
    MinMaxScaler using min (row 0) and max (row 1).

    Transform: x -> z = (x - min) / (max - min)
    Inverse transform: z -> x = z * (max - min) + min
    """
    def __init__(self, reset=False):
        super().__init__('minmax', reset)

    def _get_new_vars(self, df: pd.DataFrame):
        return pd.DataFrame([df.min(), df.max()], index=pd.Index(['min', 'max']))

    def _tf(self, df: pd.DataFrame, good_features: list | pd.Index):
        offsetted = df - self.vars.loc['min', good_features]
        scale = self.vars.loc['max', good_features] - self.vars.loc['min', good_features]
        return offsetted / scale

    def _untf(self, df: pd.DataFrame, good_features: list | pd.Index):
        scale = self.vars.loc['max', good_features] - self.vars.loc['min', good_features]
        return df * scale + self.vars.loc['min', good_features]


class RobustScaler(Scaler):
    """
    RobustScaler using median (row 0) and IQR (row 1).

    Transform: x -> z = (x - median) / IQR
    Inverse transform: z -> x = z * IQR + median
    """
    def __init__(self, reset=False):
        super().__init__('robust', reset)

    def _get_new_vars(self, df: pd.DataFrame):
        ret = pd.DataFrame([df.median(), df.quantile(0.75) - df.quantile(0.25)], index=pd.Index(['median', 'iqr']))
        # line below is to avoid division by zero; usually for very unbalanced boolean columns
        ret.loc['iqr'] = ret.loc['iqr'].replace(0, 1)
        return ret

    def _tf(self, df: pd.DataFrame, good_features: list | pd.Index):
        return (df - self.vars.loc['median', good_features]) / self.vars.loc['iqr', good_features]

    def _untf(self, df: pd.DataFrame, good_features: list | pd.Index):
        return df * self.vars.loc['iqr', good_features] + self.vars.loc['median', good_features]


class QuantileTransformer(Scaler):
    """
    QuantileTransformer using q1 (row 0) and q3 (row 1). Similar to RobustScaler but less sensitive to outliers.

    Transform: x -> z = (x - q1) / (q3 - q1)
    Inverse transform: z -> x = z * (q3 - q1) + q1
    """
    def __init__(self, reset=False):
        super().__init__('quantile', reset)

    def _get_new_vars(self, df: pd.DataFrame):
        return pd.DataFrame([df.quantile(0.25), df.quantile(0.75)], index=pd.Index(['q1', 'q3']))

    def _tf(self, df: pd.DataFrame, good_features: list | pd.Index):
        offsetted = df - self.vars.loc['q1', good_features]
        scale = self.vars.loc['q3', good_features] - self.vars.loc['q1', good_features]
        return offsetted / scale

    def _untf(self, df: pd.DataFrame, good_features: list | pd.Index):
        scale = self.vars.loc['q3', good_features] - self.vars.loc['q1', good_features]
        return df * scale + self.vars.loc['q1', good_features]


class PowerTransformer(Scaler):
    """
    PowerTransformer using Yeo-Johnson transformation then standard scaling.

    Transform: x -> z = (yj(x) - mean) / std
    Inverse transform: z -> x = unyj(z * std + mean)

    For yj(x) and unyj(y), see `yj` and `unyj` functions in `Scalers.py`.
    """
    def __init__(self, reset=False):
        """
        PowerTransformer using Yeo-Johnson transformation then standard scaling
        Args:
            reset: bool. If True, reset a fresh scaler instance which needs to be fitted and saved. If False, load the saved scaler instance if exists.
        """
        super().__init__('power', reset)

    def _get_new_vars(self, df: pd.DataFrame):
        lambdas = find_lambda(df)
        return pd.DataFrame([df.mean(), df.std(), lambdas], index=pd.Index(['mean', 'std', 'lambda']))

    def _tf(self, df: pd.DataFrame, good_features: list | pd.Index):
        lambdas = self.vars.loc['lambda', good_features]
        # TODO

    def _untf(self, df: pd.DataFrame, good_features: list | pd.Index):
        lambdas = self.vars.loc['lambda', good_features]
        # TODO


class NoScaler(Scaler):
    """
    NoScaler which does not scale the data. Used for debugging or when the data is already scaled or when the data is not needed to be scaled.

    Transform: x -> x
    Inverse transform: x -> x
    """
    def __init__(self, reset=True):
        # reset option is pointless, but for consistency
        super().__init__('noscaler', reset)

    def _get_new_vars(self, df: pd.DataFrame):
        # placeholders
        ret = pd.DataFrame()
        for col in df.columns:
            ret[col] = [0, 1]
        ret.index = ['mean', 'std']
        return ret

    def _tf(self, df: pd.DataFrame, good_features: list | pd.Index):
        return df[good_features]

    def _untf(self, df: pd.DataFrame, good_features: list | pd.Index):
        return df[good_features]


def get_scaler(scaler_type: str, reset=False) -> Scaler:
    """
    Get a scaler instance. Available types are 'standard', 'minmax', 'robust', 'quantile', 'power'(not implemented yet).
    Args:
        scaler_type: str. Type of the scaler
        reset: bool. If True, reset a fresh scaler instance which needs to be fitted and saved. If False, load the saved scaler instance if exists.

    Returns:
        Scaler. A scaler instance
    """
    scaler_type = scaler_type.lower().strip()
    if scaler_type == 'standard' or scaler_type == 'std':
        return StandardScaler(reset=reset)
    elif scaler_type == 'minmax':
        return MinMaxScaler(reset=reset)
    elif scaler_type == 'robust':
        return RobustScaler(reset=reset)
    elif scaler_type == 'quantile':
        return QuantileTransformer(reset=reset)
    elif scaler_type == 'power' or scaler_type == 'pt':
        # return PowerTransformer(reset=reset)
        raise NotImplementedError("PowerTransformer is not implemented yet")
    elif scaler_type in ['noscaler', 'no', 'none']:
        return NoScaler(reset=reset)
    else:
        raise ValueError(f"Invalid scaler type: {scaler_type}")


def find_lambda(df: pd.DataFrame):
    from scipy.stats import yeojohnson
    # yeojohnson(*) returns the transformed data and lambda
    lambdas = [yeojohnson(df[col])[1] for col in df.columns]
    return pd.Series(lambdas, index=df.columns)


def yj(x: float, lambda_: float):
    if x >= 0:
        if lambda_ == 0:
            return np.log(x + 1)  # > 0
        else:
            return ((x + 1) ** lambda_ - 1) / lambda_  # > 0
    else:
        if lambda_ == 2:
            return -np.log(-x + 1)  # < 0
        else:
            return -((-x + 1) ** (2 - lambda_) - 1) / (2 - lambda_)  # < 0


def unyj(y: float, lambda_: float):
    if y >= 0:
        if lambda_ == 0:
            return np.exp(y) - 1
        else:
            return (y * lambda_ + 1) ** (1 / lambda_) - 1
    else:
        if lambda_ == 2:
            return -np.exp(-y) + 1
        else:
            return -(2 - lambda_) * (-y * (2 - lambda_) + 1) ** (1 / (2 - lambda_)) + 1


def reset_scalers(mc_data_all: pd.DataFrame = None):
    """
    Resets and fits scalers for the mc(all) data
    Args:
        mc_data_all: pd.DataFrame. MC data(all) to fit the scalers. If None, load the mc(all) data

    Returns:
        None
    """
    mc_data = dataprep.load_proced_mc('all') if mc_data_all is None else mc_data_all
    # mc_data = mc_data.drop(columns=["sig"])
    good_cols, _ = utils.useful_cols_classify(mc_data.columns)
    mc_data = mc_data[good_cols]
    for scaler_type in ["std", "minmax", "robust", "quantile", 'noscaler']:
        scaler = get_scaler(scaler_type=scaler_type, reset=True)
        scaler.fit(mc_data)
        scaler.transform(mc_data[[col for col in mc_data.columns if col.startswith("pi3")]])
        scaler.save()
        utils.log(f"{scaler_type} scaler fitted and saved")


if __name__ == '__main__':
    @utils.alert
    def hehe():
        filename = __file__.split('/')[-1]
        res = utils.loginput(f"`{filename}` is used to reset scaler models. Do you want to continue? (y/n): ")
        if res != 'y':
            return
        reset_scalers()
        utils.log(f"Finished running `{filename}`")
    hehe()
