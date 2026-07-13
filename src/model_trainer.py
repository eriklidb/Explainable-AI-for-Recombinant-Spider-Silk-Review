import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
import optuna
from dataset import Dataset
from typing import Iterator, Literal, Sequence
import joblib
from typing import Any
from scipy.stats import pearsonr
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

class ModelTrainer():
    def __init__(self,
        n_outer_folds: int=3,
        n_inner_folds: int=5,
        cv_type: Literal['kfold', 'groupkfold']='groupkfold',
        shuffle: bool=True,
        random_state: int | np.random.RandomState | None=None) -> None:
        if cv_type == 'kfold':
            self._outer_cv = KFold(n_splits=n_outer_folds, shuffle=shuffle, random_state=random_state)
            self._inner_cv = KFold(n_splits=n_inner_folds, shuffle=shuffle, random_state=random_state)
        else: 
            self._outer_cv = GroupKFold(n_splits=n_outer_folds, shuffle=shuffle, random_state=random_state) # type: ignore
            self._inner_cv = GroupKFold(n_splits=n_inner_folds, shuffle=shuffle, random_state=random_state) # type: ignore
        self._random_state = random_state

    def split(self, ds: Dataset) -> Iterator:
        X, Y = ds()
        return self._outer_cv.split(X, Y, X['Protein'].to_numpy())

    def hyperparameter_search(self, ds: Dataset,
                          model_type: Literal['rr', 'rfr', 'etr', 'hgbr', 'xgbr', 'cbr', 'knnr'],
                          target: None | str = None,
                          n_trails: int|None=None,
                          timeout: int|None=None,
                          n_jobs: int=1, study_name: str|None=None) -> list[dict[str, Any]]:
        X, Y = ds()
        Y = Y.loc[:, target]
        sample_nums = ds.sample_numbers
        best_params = []
            
        for outer_fold, (train_idx, test_idx) in enumerate(self._outer_cv.split(X, groups=sample_nums.to_numpy())):
            X_train, _ = X.iloc[train_idx], X.iloc[test_idx]
            Y_train, _ = Y.iloc[train_idx], Y.iloc[test_idx]

            iteration_counts = []

            def objective(trial: optuna.trial.Trial) -> float | Sequence[float]:
                match model_type:
                    case 'rr':
                        params = {
                            "alpha": trial.suggest_float("alpha", 1e-4, 100),
                            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                        }
                        model = Ridge(**params, random_state=self._random_state)
                    case 'rfr':
                        params = {
                            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20, 30]),
                            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 1.0, None]),
                            "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
                        }
                        model = RandomForestRegressor(**params, random_state=self._random_state, n_jobs=n_jobs)
                    case 'etr':
                        params = {
                            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20, 30]),
                            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 1.0, None]),
                            "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
                        }
                        model = ExtraTreesRegressor(**params, random_state=self._random_state, n_jobs=n_jobs)
                    case 'hgbr':
                        params = {
                            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 100),
                            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
                            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
                            "max_depth": trial.suggest_categorical("max_depth", [3, 5, 8, None])
                        }
                        # Use early stopping in inner CV
                        model = HistGradientBoostingRegressor(
                            **params,
                            early_stopping=True,
                            validation_fraction=.1,
                            random_state=self._random_state,
                            categorical_features=ds.categorical_columns,
                        )
                    case 'xgbr':
                        params = {
                            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                            "max_depth": trial.suggest_int("max_depth", 3, 100),
                            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                            "gamma": trial.suggest_int("gamma", 0, 10),
                            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100, log=True),
                        }
                        model = XGBRegressor(
                            **params,
                            objective="reg:squarederror",
                            tree_method="hist",
                            random_state=self._random_state,
                            n_jobs=n_jobs
                        )
                    case 'cbr':
                        params = {
                            "iterations": trial.suggest_int("iterations", 200, 800),
                            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                            "depth": trial.suggest_int("depth", 4, 8),
                            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 20),
                            "random_strength": trial.suggest_float("random_strength", 0, 2),
                            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5),
                            #"bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
                            "border_count": trial.suggest_categorical("border_count", [32, 64, 128]),
                        }
                        model = CatBoostRegressor(
                            **params,
                            loss_function="RMSE",
                            random_state=self._random_state,
                            verbose=False,
                            nan_mode='Min',
                            allow_writing_files=False,
                            boosting_type='Plain',
                            leaf_estimation_iterations=1,
                            thread_count=n_jobs,
                            #task_type='GPU',
                            #devices='0'
                        )
                    case 'knnr':
                        params = {
                            "n_neighbors": trial.suggest_int("n_neighbors", 2, 50),
                            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                            "p": trial.suggest_float("p", 1, 2),
                            "leaf_size": trial.suggest_int("leaf_size", 20, 60),
                        }
                        model = KNeighborsRegressor(**params, n_jobs=n_jobs)

                scores = []
                for inner_train_idx, val_idx in self._inner_cv.split(X_train, groups=sample_nums[train_idx].to_numpy()):
                    X_inner_train, X_val = X_train.iloc[inner_train_idx], X_train.iloc[val_idx]
                    Y_inner_train, Y_val = Y_train.iloc[inner_train_idx], Y_train.iloc[val_idx]
                    if model_type == 'cbr':
                        #model.fit(X_inner_train, Y_inner_train, cat_features=ds.categorical_columns)
                        model.fit(X_inner_train, Y_inner_train, cat_features=[11,12,13,14,15,16,17])
                    else:
                        model.fit(X_inner_train, Y_inner_train)

                    preds = model.predict(X_val)
                    score = root_mean_squared_error(Y_val, preds)
                    scores.append(score)

                    # Save iteration counts
                    if model_type == 'hgbr':
                        iteration_counts.append(model.n_iter_)

                return float(np.mean(scores))

            seed = self._random_state if type(self._random_state) == int else None
            sampler = optuna.samplers.TPESampler(seed=seed)
            study_name_ = None if study_name is None else f'{study_name}_{outer_fold}_{target}'
            study = optuna.create_study(direction="minimize", study_name=study_name_, sampler=sampler)
            study.optimize(objective, n_trials=n_trails, timeout=timeout, n_jobs=n_jobs)
            best_params.append(study.best_params)

            # Dynamically determine max_iter
            if model_type == 'hgbr':
                mean_iter = np.mean(iteration_counts)
                std_iter = np.std(iteration_counts)
                max_iter_final = int(mean_iter + std_iter)
                best_params[-1]['max_iter'] = max_iter_final
                print(f"[Fold {outer_fold+1}] Mean Iter: {mean_iter:.2f}, Std: {std_iter:.2f} → Using max_iter={max_iter_final}")

        return best_params

    def train_model(self, 
                    ds: Dataset, 
                    model_type: Literal['lr', 'rfr', 'etr', 'hgbr', 'xgbr', 'cbr'],
                    params: list[dict[str, Any]] | None = None,
                    target: None | str = None,
                    n_jobs: int = -1) -> list[any]:
        X, Y = ds()
        Y = Y.loc[:, target]

        outer_models = []
        for outer_fold, (train_idx, test_idx) in enumerate(self._outer_cv.split(X, groups=ds.sample_numbers.to_numpy())):
            X_train, _ = X.iloc[train_idx], X.iloc[test_idx]
            Y_train, _ = Y.iloc[train_idx], Y.iloc[test_idx]
            # Train final model on full outer training set (no early stopping)
            match(model_type):
                case 'lr':
                    final_model = LinearRegression()
                case 'rr':
                    final_model = Ridge(**params[outer_fold], random_state=self._random_state)
                case 'rfr':
                    final_model = RandomForestRegressor(**params[outer_fold], random_state=self._random_state, n_jobs=n_jobs)
                case 'etr':
                    final_model = ExtraTreesRegressor(**params[outer_fold], random_state=self._random_state, n_jobs=n_jobs)
                case 'hgbr':
                    final_model = HistGradientBoostingRegressor(**params[outer_fold], early_stopping=False, random_state=self._random_state, categorical_features=ds.categorical_columns)
                case 'xgbr':
                    final_model = XGBRegressor(**params[outer_fold], objective="reg:squarederror", tree_method="hist", random_state=self._random_state, n_jobs=n_jobs)
                case 'cbr':
                    final_model = CatBoostRegressor(**params[outer_fold], 
                            loss_function="RMSE",
                            random_state=self._random_state,
                            verbose=False,
                            nan_mode='Min',
                            allow_writing_files=False,
                            boosting_type='Plain',
                            leaf_estimation_iterations=1,
                            thread_count=n_jobs)
                case 'knnr':
                    final_model = KNeighborsRegressor(**params[outer_fold], n_jobs=n_jobs)
            if model_type == 'cbr':
                final_model.fit(X_train, Y_train, cat_features=[11,12,13,14,15,16,17])
            else:
                final_model.fit(X_train, Y_train)
            outer_models.append(final_model)
        return outer_models

    def train_lr_model(self, 
                    ds: Dataset,
                    target: None | str = None) -> list[LinearRegression]:
        multioutput = target is None
        X, Y = ds()
        if not multioutput:
            Y = Y.loc[:, target]

        outer_models = []
        for (train_idx, test_idx) in self._outer_cv.split(X, groups=ds.sample_numbers.to_numpy()):
            X_train, _ = X.iloc[train_idx], X.iloc[test_idx]
            Y_train, _ = Y.iloc[train_idx], Y.iloc[test_idx]
            lr_model = LinearRegression()
            lr_model.fit(X_train, Y_train)
            outer_models.append(lr_model)
        return outer_models

    def train_rf_model(self, 
                    ds: Dataset,
                    params: list[dict[str, Any]],
                    target: None | str = None) -> list[LinearRegression]:
        multioutput = target is None
        X, Y = ds()
        if not multioutput:
            Y = Y.loc[:, target]

        outer_models = []
        for outer_fold, (train_idx, test_idx) in enumerate(self._outer_cv.split(X, groups=ds.sample_numbers.to_numpy())):
            X_train, _ = X.iloc[train_idx], X.iloc[test_idx]
            Y_train, _ = Y.iloc[train_idx], Y.iloc[test_idx]
            rf_model = RandomForestRegressor(**params[outer_fold])
            rf_model.fit(X_train, Y_train)
            outer_models.append(rf_model)
        return outer_models

def save_model(model: Any, fname: str) -> None:
    joblib.dump(model, fname)

def load_model(fname: str) -> Any:
    return joblib.load(fname)

def compute_metrics(y_true, y_pred) -> pd.DataFrame:
    index = ['RMSE', 'MAE', '$R^2$', 'PCC']
    columns = ['Diameter (µm)', 'Strain (mm/mm)', 'Strength (MPa)', 'Youngs Modulus (GPa)', 'Toughness Modulus (MJ m-3)'] 
    data = [root_mean_squared_error(y_true, y_pred, multioutput='raw_values'),
            mean_absolute_error(y_true, y_pred, multioutput='raw_values'),
            r2_score(y_true, y_pred, multioutput='raw_values'),
            pearsonr(y_true, y_pred).statistic] # type: ignore
    return pd.DataFrame(data, index, columns)


if __name__ == '__main__':
    seed = 42
    ds = {}
    ds['A'] = Dataset('spinning_data.csv')
    ds['B'] = Dataset('spinning_data_embeddings.csv')
    #cv_type = 'kfold'
    cv_type = 'groupkfold'
    n_folds = 3
    n_inner_folds = 5
    mt = ModelTrainer(n_outer_folds=n_folds, n_inner_folds=n_inner_folds, cv_type=cv_type, random_state=seed)
    study_params = {
        'n_trails': 50,
        'timeout': 1800,
        'n_jobs': -1
    }
    
    multioutput = False
    if multioutput:
        for k in 'A', 'B':
            model_params = mt.hyperparameter_search(ds[k], study_name=k, **study_params)
            models = mt.train_model(ds[k], model_params)
            for i, model in enumerate(models):
                save_model(model, f'../models/model_{k}_fold_{i}')
    else:
        for k in 'A', 'B':
            for target in 'Diameter (µm)',\
                'Strain (mm/mm)', 'Strength (MPa)',\
                'Youngs Modulus (GPa)', 'Toughness Modulus (MJ m-3)':
                model_params = mt.hyperparameter_search(ds[k], target=target, study_name=k, **study_params)
                models = mt.train_model(ds[k], model_params, target)
                for i, model in enumerate(models):
                    save_model(model, f'../models/model_{k}_fold_{i}_{target.split()[0]}')