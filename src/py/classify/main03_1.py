import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from myClassifiers import TabTransformer
import Scalers
import optuna
from src.py.utils import config, plotter, utils, consts

epochs, optuna_trials = 100, 10

class ModelsContainer:
    def __init__(self, bkgsig_ratio: int | str=2, scaler_type="std"):
        self.bkgsig_ratio = bkgsig_ratio
        self._data_mc = utils.load(config.mc_proced_file(self.bkgsig_ratio))
        self._og_index = self._data_mc['og_index']
        good_cols, _ = utils.useful_cols_classify(self._data_mc.columns)
        good_cols.append('sig')
        self.data_mc = self._data_mc[good_cols]
        self.train_x, self.test_x, self.train_y, self.test_y = prepare_data(self.data_mc, test_size=0.05, scale=scaler_type)

        self.scaler = Scalers.get_scaler(scaler_type=scaler_type, reset=False)
        self.models = {
            # "tabtransformer": TabTransformer(num_features=self.train_x.shape[1], num_classes=2),
            "torch": torch.nn.Sequential(),
            "bdt": xgb.Booster(),
            "randomforest": RandomForestClassifier(),
            # "svm": None,
        }
        self.loaded = {key: False for key in self.models.keys()}
        self.train_probas = {key: pd.Series() for key in self.models.keys()}
        self.train_probas['_ans'] = self.train_y
        self.test_probas = {key: pd.Series() for key in self.models.keys()}
        self.test_probas['_ans'] = self.test_y

        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _train_torch(self) -> None:
        """
        Train a torch model using the training data, save the model/train probs/test probs as an attribute of the class.
        Returns:
            None
        """
        utils.log(f"Training Torch model for bkg/sig ratio {self.bkgsig_ratio}")
        num_features = self.train_x.shape[1]
        train_x_tensor = torch.tensor(self.train_x.values, dtype=torch.float32)
        train_y_tensor = torch.tensor(self.train_y.values, dtype=torch.float32).view(-1, 1)
        model = torch.nn.Sequential(
            torch.nn.Linear(num_features, 300),
            torch.nn.Tanh(),
            torch.nn.Linear(300, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 1),
            torch.nn.Sigmoid(),
        )

        count1s = sum(self.train_y)
        count0s = len(self.train_y) - count1s
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([count1s / count0s]))
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(train_x_tensor)
            loss = criterion(outputs, train_y_tensor)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                utils.log(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
        self.models["torch"] = model
        self.loaded["torch"] = True
        self.calc_tt_probs("torch")
        utils.log(f"Torch model for bkg/sig ratio {self.bkgsig_ratio} trained")

    def _train_bdt(self) -> None:
        """
        Train a BDT model using the training data and save the model/train probs/test probs as an attribute of the class
        Returns:
            None
        """
        utils.log(f"Training BDT model for bkg/sig ratio {self.bkgsig_ratio}")
        weight = (len(self.train_y) - self.train_y.sum()) / self.train_y.sum()  # > 1 if more bkg than sig
        weights = self.train_y.replace({0: 1, 1: weight})
        dtrain = xgb.DMatrix(data=self.train_x, label=self.train_y, weight=weights)

        def objective(trial: optuna.Trial):
            # Suggest parameters
            param = {
                'objective': 'binary:logistic',
                'eval_metric': ['logloss', 'aucpr', 'auc'],
                'max_depth': trial.suggest_int('max_depth', 2, 5),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
                'max_delta_step': trial.suggest_float('max_delta_step', 0.0, 10.0),
                'scale_pos_weight': weight,
            }

            # Train and validate with cross-validation
            test_cv_kwargs = {"dtrain": dtrain, "nfold": 5,
                              "stratified": True, "metrics": "logloss",
                              "early_stopping_rounds": 10, "seed": 42,
                              "verbose_eval": False, "params": param,
                              "num_boost_round": trial.suggest_int('num_boost_round', 50, 200)}
            cv_results = xgb.cv(**test_cv_kwargs)

            # Return the mean test logloss from cross-validation
            return cv_results['test-logloss-mean'].iloc[-1]

        # Optimize parameters with Optuna
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=optuna_trials)
        best_params = study.best_params
        num_boost_round = best_params.pop("num_boost_round")

        # Train final model using the best parameters and boosting round
        self.models["bdt"] = xgb.train(
            dtrain=dtrain,
            params=best_params,
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train')],
            verbose_eval=False,
        )
        self.loaded["bdt"] = True
        self.calc_tt_probs("bdt")
        utils.log(f"BDT model for bkg/sig ratio {self.bkgsig_ratio} trained")

    def _train_randomforest(self) -> None:
        """
        Train a RandomForest model using the training data and save the model/train probs/test probs as an attribute of the class
        Returns:
            None
        """
        utils.log(f"Training RandomForest model for bkg/sig ratio {self.bkgsig_ratio}")
        def objective(trial: optuna.Trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 2, 5),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
                'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.1),
                'class_weight': 'balanced',
            }

            model = RandomForestClassifier(**param)
            model.fit(self.train_x, self.train_y)
            preds = model.predict(self.train_x)
            f1 = f1_score(self.train_y, preds, average='weighted')
            return f1

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=optuna_trials, n_jobs=-1)
        best_model = RandomForestClassifier(**study.best_params)
        best_model.fit(self.train_x, self.train_y)

        self.models["randomforest"] = best_model
        self.loaded["randomforest"] = True
        self.calc_tt_probs("randomforest")
        utils.log(f"RandomForest model for bkg/sig ratio {self.bkgsig_ratio} trained")

    def _train_svm(self) -> None:
        """
        Train a SVM model using the training data and save the model/train probs/test probs as an attribute of the class
        Returns:
            None
        """
        utils.log(f"Training SVM model for bkg/sig ratio {self.bkgsig_ratio}")

        # variables for parameter tuning
        param_grid = {
            "kernel": ["rbf", "linear", "poly", "sigmoid"],
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto"],
        }
        grid_search = GridSearchCV(SVC(probability=True, class_weight='balanced', random_state=42),
                                   param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(self.train_x, self.train_y)
        best_model = grid_search.best_estimator_

        self.models["svm"] = best_model
        self.loaded["svm"] = True
        self.calc_tt_probs("svm")
        utils.log(f"SVM model for bkg/sig ratio {self.bkgsig_ratio} trained")

    def _train_tabtransformer(self) -> None:
        """
        Train a TabTransformer model using the training data and save the model/train probs/test probs as an attribute of the class
        Returns:
            None
        """
        utils.log(f"Training TabTransformer model for bkg/sig ratio {self.bkgsig_ratio}")

        num_features, num_classes = self.train_x.shape[1], 2

        # Initialize the model, loss, and optimizer
        model = TabTransformer(num_features, num_classes).to(self.torch_device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Converting data to tensors
        X_train_tensor = torch.FloatTensor(self.train_x.values).to(self.torch_device)
        y_train_tensor = torch.LongTensor(self.train_y.values).to(self.torch_device)

        # Training loop
        num_epoch = 100
        for epoch in range(num_epoch):
            optimizer.zero_grad()
            output = model(X_train_tensor)
            loss = criterion(output, y_train_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 1:
                utils.log(f'Epoch [{epoch + 1}/{num_epoch}], Loss: {loss.item():.4f}')

        self.models["tabtransformer"] = model.to('cpu')
        self.loaded["tabtransformer"] = True
        self.calc_tt_probs("tabtransformer")
        utils.log(f"TabTransformer model for bkg/sig ratio {self.bkgsig_ratio} trained")

    def train(self, model_name: str) -> None:  # todo: model_name='all'
        """
        Train a model using the training data. The model will be saved as an attribute of the class as well as the train/test probabilities
        Args:
            model_name: str. The name of the model to train

        Returns:
            None
        """
        model_name = model_name.lower().strip()
        try:
            func = getattr(self, f"_train_{model_name}")
        except AttributeError:
            raise NotImplementedError(f"Model {model_name} not implemented")
        func()

    def _test_torch(self, dataset: pd.DataFrame, scale: bool=False) -> pd.Series:
        """
        Calculates the probability of the dataset being signal using the torch model
        Args:
            dataset: pd.DataFrame. The dataset to test
            scale: bool. Whether to scale the dataset the same way training data was scaled. Default: False

        Returns:
            pd.Series. The probability of the dataset being signal
        """
        if not self.loaded["torch"]:
            raise ValueError("Torch model not loaded or trained")
        torch_model = self.models["torch"]
        if scale:
            dataset = self.scaler.transform(dataset, fit_extra=False, save_extra=False)
        else:
            dataset = dataset[self.scaler.features]
        dataset_index = dataset.index
        dataset = torch.tensor(dataset.values, dtype=torch.float32)

        with torch.no_grad():
            proba = torch_model(dataset).cpu().numpy()
        proba = pd.Series(proba[:, 0], index=dataset_index, name='torch')
        return proba

    def _test_bdt(self, dataset: pd.DataFrame, scale: bool=False) -> pd.Series:
        """
        Calculates the probability of the dataset being signal using the BDT model
        Args:
            dataset: pd.DataFrame. The dataset to test
            scale: bool. Whether to scale the dataset the same way training data was scaled. Default: False

        Returns:
            pd.Series. The probability of the dataset being signal
        """
        if not self.loaded["bdt"]:
            raise ValueError("BDT model not loaded or trained")
        bdt_model = self.models["bdt"]
        if scale:
            dataset = self.scaler.transform(dataset, fit_extra=False, save_extra=False)
        else:
            dataset = dataset[self.scaler.features]
        dataset_index = dataset.index
        proba = bdt_model.predict(xgb.DMatrix(dataset))
        proba = pd.Series(proba, index=dataset_index, name='bdt')
        return proba

    def _test_randomforest(self, dataset: pd.DataFrame, scale: bool=False) -> pd.Series:
        """
        Calculates the probability of the dataset being signal using the RandomForest model
        Args:
            dataset: pd.DataFrame. The dataset to test
            scale: bool. Whether to scale the dataset the same way training data was scaled. Default: False

        Returns:
            pd.Series. The probability of the dataset being signal
        """
        if not self.loaded["randomforest"]:
            raise ValueError("RandomForest model not loaded or trained")
        randfor_model = self.models["randomforest"]
        if scale:
            dataset = self.scaler.transform(dataset, fit_extra=False, save_extra=False)
        else:
            dataset = dataset[self.scaler.features]
        dataset_index = dataset.index
        proba = randfor_model.predict_proba(dataset)[:, 1]
        proba = pd.Series(proba, index=dataset_index, name='randomforest')
        return proba

    def _test_svm(self, dataset: pd.DataFrame, scale: bool=False) -> pd.Series:
        pass

    def _test_tabtransformer(self, dataset: pd.DataFrame, scale: bool=False) -> pd.Series:
        if not self.loaded["tabtransformer"]:
            raise ValueError("TabTransformer model not loaded or trained")
        tabtf_model = self.models["tabtransformer"]
        if scale:
            dataset = self.scaler.transform(dataset, fit_extra=False, save_extra=False)
        else:
            dataset = dataset[self.scaler.features]
        dataset_index = dataset.index
        dataset = torch.FloatTensor(dataset.values).to(self.torch_device)

        with torch.no_grad():
            proba = tabtf_model(dataset).cpu().numpy()
        proba = pd.Series(proba[:, 1], index=dataset_index, name='tabtransformer')
        return proba

    def predict(self, model_name: str, dataset: pd.DataFrame, scale=False) -> pd.Series | tuple:
        """
        Predict the probability of the dataset being signal using the model.
        The model will be loaded if it hasn't been already.
        The probability of the dataset being signal will be returned.
        If model_name is 'all', a dictionary of probabilities for all models will be returned
        where key is model name in string and value is the probability
        Args:
            model_name: str. The name of the model to test. If 'all', all models will be tested
            dataset: pd.DataFrame. The dataset to test
            scale: bool. Whether to scale the dataset the same way training data was scaled. Default: False

        Returns:
            pd.Series | tuple. The probability of the dataset being signal
        """
        model_name = model_name.lower().strip()
        if model_name == "all":
            ret = {}
            for model_name in self.models.keys():
                proba = self.predict(model_name, dataset, scale=scale)
                ret[model_name] = proba
            return ret
        try:
            func = getattr(self, f"_test_{model_name}")
        except AttributeError:
            raise NotImplementedError(f"Model {model_name} not implemented")

        utils.log(f"Predicting probabilities for model {model_name}_{self.bkgsig_ratio}")
        proba = func(dataset, scale=scale)
        return proba

    def calc_tt_probs(self, model_name: str) -> None:
        """
        Calculate the probabilities of the training and testing data for all models.
        The probabilities will be saved as an attribute of the class.
        Args:
            model_name: str. The name of the model to calculate the probabilities for. If 'all', all models will be calculated

        Returns:
            None
        """
        if model_name == "all":
            for model_name in self.models.keys():
                self.calc_tt_probs(model_name)
            return
        self.train_probas[model_name] = self.predict(model_name, self.train_x, scale=False)
        self.test_probas[model_name] = self.predict(model_name, self.test_x, scale=False)
        utils.log(f"Calculated train and test probabilities for model {model_name}_{self.bkgsig_ratio}")

    def load_model(self, model_name: str) -> None:
        """
        Load a model from the './models' directory.
        The model will be saved as an attribute of the class.
        Bkg/sig ratio will be appended to the model name when finding the file.
        Args:
            model_name: str. The name of the model to load (e.g. 'bdt'). If 'all', all models will be loaded

        Returns:
            None
        """
        model_name = model_name.lower().strip()
        if model_name == "all":
            for model_name in self.models.keys():
                self.load_model(model_name)
            return
        if model_name not in self.models.keys():
            raise ValueError(f"Model {model_name} not found")
        self.models[model_name] = joblib.load(f'{config.classifiers_dir}/{model_name}_{self.bkgsig_ratio}.joblib')
        self.loaded[model_name] = True
        utils.log(f"Model {model_name}_{self.bkgsig_ratio} loaded")

    def save_model(self, model_name: str) -> None:
        """
        Save a model to the './models' directory.
        Args:
            model_name: str. The name of the model to save. If 'all', all loaded/trained models will be saved

        Returns:
            None
        """
        model_name = model_name.lower().strip()
        if model_name == "all":
            for model_name in self.models.keys():
                if self.loaded[model_name]:
                    self.save_model(model_name)
                else:
                    utils.log(f"Model {model_name} not loaded, hence not saved")
            return
        if model_name not in self.models.keys():
            raise ValueError(f"Model {model_name} not found")
        if not self.loaded[model_name]:
            raise ValueError(f"Model {model_name} not loaded")
        model = self.models[model_name]
        joblib.dump(model, f'{config.classifiers_dir}/{model_name}_{self.bkgsig_ratio}.joblib')
        utils.log(f"Model {model_name}_{self.bkgsig_ratio} saved")

    def save_probs(self, model_name: str) -> None:
        """
        Save the probabilities of the training and testing data to a csv file.
        Args:
            model_name: str. The name of the model to save the probabilities for. If 'all', all models will be saved

        Returns:
            None
        """
        model_name = model_name.lower().strip()
        if model_name == "all":
            train_probs = pd.DataFrame(self.train_probas)
            train_probs['_is_test'] = 0
            test_probs = pd.DataFrame(self.test_probas)
            test_probs['_is_test'] = 1
            probs_df = pd.concat([train_probs, test_probs], axis=0)  # vertical concat
            probs_df = probs_df.rename(columns={key: f"{key}_{self.bkgsig_ratio}" for key in probs_df.columns})
            probs_df = probs_df.dropna(how='all', axis=1)  # drop columns with all NaN values
            save_proba('tt', probs_df)
            return
        if model_name not in self.models.keys():
            raise ValueError(f"Model {model_name} not found")
        train_probs = pd.DataFrame(self.train_probas[model_name])
        train_probs[f'_is_test_{self.bkgsig_ratio}'] = 0
        test_probs = pd.DataFrame(self.test_probas[model_name])
        test_probs[f'_is_test_{self.bkgsig_ratio}'] = 1
        probs_df = pd.concat([train_probs, test_probs], axis=0)
        probs_df = probs_df.rename(columns={model_name: f"{model_name}_{self.bkgsig_ratio}"})
        # save_proba(model_name, probs_df)
        save_proba('tt', probs_df)
        utils.log(f"Saved train and test probabilities for model {model_name}_{self.bkgsig_ratio}")

    def save(self) -> None:
        """
        Saves models and the probabilities of the training and testing data to a csv file (`./data/probs/probs.csv`).
        The probabilities will be saved as a DataFrame where the columns are the model names and the rows are the indices of the data.
        Returns:
            None
        """
        self.save_probs("all")
        self.save_model("all")

# TODO: PCA
# TODO: cnn, logreg, gradientboosting, sgd, adaboost, naivebayes

def available_bkgsig_ratios(include_all=False) -> list:
    """
    Returns a list of available bkg/sig ratios
    Returns:
        list. List of available bkg/sig ratios
    """
    ratios = consts.train_bkgsig_ratios
    if not include_all and 'all' in ratios:
        ratios.remove('all')
    mc_files = utils.listdir(config.mc_dir)
    ratios = [item for item in ratios if f'mc_proced_{item}.{config.better_ext}' in mc_files]
    return ratios


def prepare_data(mc_data: pd.DataFrame, test_size=0.05, scale="std") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for training. Split into train and test, and scale the data.
    Args:
        mc_data: pd.DataFrame. Data to prepare
        test_size: float. Size of test data. Default: 0.05
        scale: str. Type of scaling to apply. Options: 'std', 'none'. Default: 'std'

    Returns:
        tuple. Train and test data in the following order: train_x, test_x, train_y, test_y
    """
    # split
    data_y = mc_data['sig'].astype(int).values
    data_x = mc_data.drop(columns=['sig'])
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=test_size, random_state=42)

    # scale
    scaler = Scalers.get_scaler(scaler_type=scale, reset=False)
    train_x = scaler.transform(train_x, fit_extra=None, save_extra=False)
    test_x = scaler.transform(test_x, fit_extra=None, save_extra=False)

    test_y = pd.Series(test_y, index=test_x.index)
    train_y = pd.Series(train_y, index=train_x.index)

    return train_x, test_x, train_y, test_y


def save_proba(proba_type: str, proba: pd.Series | pd.DataFrame) -> None:
    """
    Appends probability vector to probability file under the `probas_dir` specified in the config file.
    Args:
        proba_type: str. Type of probability vector to save
        proba: pd.Series | pd.DataFrame. Probability vector to save

    Returns:
        None
    """
    proba = pd.DataFrame(proba) if isinstance(proba, pd.Series) else proba
    filename = f'{config.probas_dir}/proba_{proba_type}'

    if os.path.exists(filename + ".csv"):
        og_df = pd.read_csv(filename + ".csv")
        og_df = og_df[og_df.columns[~og_df.columns.isin(proba.columns)]]
    elif os.path.exists(filename + ".root"):
        og_df = utils.load(filename + ".root")
        og_df = og_df[og_df.columns[~og_df.columns.isin(proba.columns)]]
    else:
        og_df = pd.DataFrame()
    full_df = pd.concat([proba, og_df], axis=1).sort_index().sort_index(axis=1)

    # full_df.to_csv(filename + ".csv", index=False)
    utils.to_root(full_df, filename + ".root")
    utils.save_sample(config.probas_dir)
    utils.log(f"Saved probabilities for {proba_type} as {filename}")


def main_train() -> None:
    """
    Train models for all available bkg/sig ratios
    Returns:
        None
    """
    for bkgsig_ratio in available_bkgsig_ratios():
        model_container = ModelsContainer(bkgsig_ratio=bkgsig_ratio)
        for model_name in model_container.models.keys():
            model_container.train(model_name)
        model_container.save()
        utils.log(f"Trained and saved models for bkg/sig ratio {bkgsig_ratio}")
    utils.log("Training complete")


def main_test(full=False, tt=True) -> None:
    """
    Load models and save the probabilities of the full data, train-test data, and fuller data
    Returns:
        None
    """
    full_data = utils.load(config.target_core_file) if full else None
    for ratio in available_bkgsig_ratios():
        utils.log(f"Predicting probabilities for bkg/sig ratio {ratio}")
        model_container = ModelsContainer(bkgsig_ratio=ratio)
        model_container.load_model("all")
        if full:
            utils.log("Predicting probabilities for the full data")
            full_probs = model_container.predict("all", full_data, scale=True)
            full_probs = pd.DataFrame(full_probs)
            full_probs = full_probs.rename(columns={key: f"{key}_{ratio}" for key in full_probs.columns})
            save_proba(config.mode, full_probs)
        if tt:
            utils.log("Predicting probabilities for train-test data")
            model_container.calc_tt_probs("all")
            model_container.save_probs("all")
        utils.log(f"Predicted probabilities for bkg/sig ratio {ratio}")
    utils.log("Prediction complete")

    if full:
        utils.log("Joining full probabilities with full data...")
        full_probs = utils.load(config.target_score_file)
        full_data = utils.load(config.target_core_file) if full_data is None else full_data
        full_data = pd.concat([full_data[utils.useful_cols_fit(full_data.columns)],
                               full_probs], axis=1)
        utils.to_root(full_data, config.target_with_score_file)
    utils.save_sample(config.probas_dir)

    # tt files don't need joining as we only use it for train-test probs dists


def main_join() -> None:
    """
    Calculate predictions of composite models (min, max, mean) and save them to a csv file. Works properly only if all models work correctly, so nearly deprecated.
    Returns:
        None
    """
    models = ["torch", "bdt", "randomforest"]
    utils.log("Combining probabilities")
    full_probs_df = pd.read_csv(f"{config.probas_dir}/proba_full.csv")
    tt_probs_df = pd.read_csv(f"{config.probas_dir}/proba_tt.csv")
    for model in models:
        model_cols = [f"{model}_{i}" for i in available_bkgsig_ratios()]
        # model_cols = [model + str(i) for i in available_bkgsig_ratios()]
        full_probs_df[model + "_min"] = full_probs_df[model_cols].min(axis=1)
        full_probs_df[model + '_max'] = full_probs_df[model_cols].max(axis=1)
        full_probs_df[model + '_mean'] = full_probs_df[model_cols].mean(axis=1)
    utils.log('Calculated min, max, mean for each model')

    for i in available_bkgsig_ratios():
        # i_cols = [model + str(i) for model in models]
        i_cols = [f"{model}_{i}" for model in models]
        full_probs_df[f"all_{i}_mean"] = full_probs_df[i_cols].mean(axis=1)
        full_probs_df[f"all_{i}_min"] = full_probs_df[i_cols].min(axis=1)
        full_probs_df[f"all_{i}_max"] = full_probs_df[i_cols].max(axis=1)
        tt_probs_df[f'all_{i}_mean'] = tt_probs_df[i_cols].mean(axis=1)
        tt_probs_df[f'all_{i}_min'] = tt_probs_df[i_cols].min(axis=1)
        tt_probs_df[f'all_{i}_max'] = tt_probs_df[i_cols].max(axis=1)
    utils.log('Calculated min, max, mean for each bkgsig_ratio')

    full_probs_df = full_probs_df.sort_index(axis=1)
    full_probs_df.to_csv(f"{config.probas_dir}/proba_full.csv", index=False)
    utils.to_root(full_probs_df, f"{config.probas_dir}/proba_full.root")
    tt_probs_df = tt_probs_df.sort_index(axis=1)
    tt_probs_df.to_csv(f"{config.probas_dir}/proba_tt.csv", index=False)
    utils.to_root(tt_probs_df, f"{config.probas_dir}/proba_tt.root")
    utils.log("Full probabilities saved")


def main_plot_dist() -> None:
    """
    Plot mass distributions for all available bkg/sig ratios
    Returns:
        None
    """
    ratios = available_bkgsig_ratios()
    ratios.sort(reverse=True)  # all, 10, 5, 2, 1
    m2d_range, dm_range, deltam_range = None, None, None
    for ratio in ratios:
        mc_data = utils.load(config.mc_proced_file(ratio))
        if 'delta_M' not in mc_data.columns:
            mc_data['delta_M'] = mc_data['Dst_M'] - mc_data['D_M']
        sig = mc_data.loc[mc_data['sig'] == 1]
        bkg = mc_data.loc[mc_data['sig'] == 0]
        dists_dir = config.plotdir3(plot_type='dists')
        temp = plotter.plot_2dmass(data=bkg[['D_M', 'delta_M']],
                                   plot_range=m2d_range,
                                   box=False,
                                   filename=f"{dists_dir}/2dm_bkg_{ratio}.png")
        m2d_range = temp if m2d_range is None else m2d_range
        plotter.plot_2dmass(data=sig[['D_M', 'delta_M']],
                            plot_range=m2d_range,
                            box=False,
                            filename=f"{dists_dir}/2dm_sig_{ratio}.png")
        temp = plotter.plot_1dhist(data=bkg['D_M'],
                                   plot_range=dm_range,
                                   title=f"Background D_M Distribution for bkg/sig ratio {ratio}",
                                   filename=f"{dists_dir}/dm_bkg_{ratio}.png")
        dm_range = temp if dm_range is None else dm_range
        plotter.plot_1dhist(data=sig['D_M'],
                            plot_range=dm_range,
                            title=f"Signal D_M Distribution for bkg/sig ratio {ratio}",
                            filename=f"{dists_dir}/dm_sig_{ratio}.png")
        temp = plotter.plot_1dhist(data=bkg['delta_M'],
                                   plot_range=deltam_range,
                                   title=f"Background delta_M Distribution for bkg/sig ratio {ratio}",
                                   filename=f"{dists_dir}/deltam_bkg_{ratio}.png")
        deltam_range = temp if deltam_range is None else deltam_range
        plotter.plot_1dhist(data=sig['delta_M'],
                            plot_range=deltam_range,
                            title=f"Signal delta_M Distribution for bkg/sig ratio {ratio}",
                            filename=f"{dists_dir}/deltam_sig_{ratio}.png")
        utils.log(f"Plotted mass distributions for bkg/sig ratio {ratio}")

    full_data = utils.load(config.target_core_file)
    # box = [[consts.sneha_masscuts['dmmin'], consts.sneha_masscuts['dmmax']],
    #        [consts.sneha_masscuts['deltammin'], consts.sneha_masscuts['deltammax']]]
    # box = [[(box[0][0] + box[0][1])/2, (box[1][0] + box[1][1])/2], [box[0][1] - box[0][0], box[1][1] - box[1][0]]]
    box = [[consts.mc_stats.dmmean, consts.mc_stats.deltammean],
           [consts.mc_stats.dmstd, consts.mc_stats.deltamstd]]
    plotter.plot_2dmass(data=full_data[['D_M', 'delta_M']],
                        plot_range=m2d_range,
                        box=True,
                        box_range=box,
                        filename=f"{config.plotdir3(plot_type='dists')}/2dm_full.png")


if __name__ == "__main__":
    @utils.alert
    def hehe():
        main_train()
        main_test(full=True, tt=True)
        # main_join()
        main_plot_dist()
    hehe()
