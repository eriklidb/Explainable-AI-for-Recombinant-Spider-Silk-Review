import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
import optuna
from dataset import Dataset
from typing import Iterator, Literal, Sequence
import joblib
from typing import Any
from scipy.stats import pearsonr


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
                          target: None | str = None,
                          n_trails: int|None=None,
                          timeout: int|None=None,
                          n_jobs: int=1,
                          study_name: str|None=None) -> list[dict[str, Any]]:
        multioutput = target is None
        X, Y = ds()
        sample_nums = ds.sample_numbers
        if not multioutput:
            Y = Y.loc[:, target]
        best_params = []
            
        for outer_fold, (train_idx, test_idx) in enumerate(self._outer_cv.split(X, groups=sample_nums.to_numpy())):
            X_train, _ = X.iloc[train_idx], X.iloc[test_idx]
            Y_train, _ = Y.iloc[train_idx], Y.iloc[test_idx]

            iteration_counts = []

            def objective(trial: optuna.trial.Trial) -> float | Sequence[float]:
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 50),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                    "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
                }

                # Use early stopping in inner CV
                model = MultiOutputRegressor(
                    HistGradientBoostingRegressor(
                        **params,
                        early_stopping=True,
                        validation_fraction=.1,
                        random_state=self._random_state,
                        categorical_features=ds.categorical_columns
                    )
                ) if multioutput else HistGradientBoostingRegressor(
                        **params,
                        early_stopping=True,
                        validation_fraction=.1,
                        random_state=self._random_state,
                        categorical_features=ds.categorical_columns
                    )

                scores = []
                for inner_train_idx, val_idx in self._inner_cv.split(X_train, groups=sample_nums[train_idx].to_numpy()):
                    X_inner_train, X_val = X_train.iloc[inner_train_idx], X_train.iloc[val_idx]
                    Y_inner_train, Y_val = Y_train.iloc[inner_train_idx], Y_train.iloc[val_idx]
                    model.fit(X_inner_train, Y_inner_train)

                    preds = model.predict(X_val)
                    score = root_mean_squared_error(Y_val, preds)
                    scores.append(score)

                    # Save iteration counts
                    if multioutput:
                        for est in model.estimators_:
                            if isinstance(est, HistGradientBoostingRegressor):
                                iteration_counts.append(est.n_iter_)
                    else:
                        if isinstance(model, HistGradientBoostingRegressor):
                            iteration_counts.append(model.n_iter_)

                return float(np.mean(scores))

            seed = self._random_state if type(self._random_state) == int else None
            sampler = optuna.samplers.TPESampler(seed=seed)
            if multioutput:
                study_name_ = None if study_name is None else f'{study_name}_{outer_fold}'
            else:
                study_name_ = None if study_name is None else f'{study_name}_{outer_fold}_{target}'
            study = optuna.create_study(direction="minimize", study_name=study_name_, sampler=sampler)
            study.optimize(objective, n_trials=n_trails, timeout=timeout, n_jobs=n_jobs)

            # Dynamically determine max_iter
            mean_iter = np.mean(iteration_counts)
            std_iter = np.std(iteration_counts)
            max_iter_final = int(mean_iter + std_iter)

            print(f"[Fold {outer_fold+1}] Mean Iter: {mean_iter:.2f}, Std: {std_iter:.2f} → Using max_iter={max_iter_final}")

            best_params.append(study.best_params)
            best_params[-1]['max_iter'] = max_iter_final
        return best_params

    def train_model(self, 
                    ds: Dataset, 
                    params: list[dict[str, Any]],
                    target: None | str = None) -> list[MultiOutputRegressor] | list[HistGradientBoostingRegressor]:
        multioutput = target is None
        X, Y = ds()
        if not multioutput:
            Y = Y.loc[:, target]

        outer_models = []
        for outer_fold, (train_idx, test_idx) in enumerate(self._outer_cv.split(X, groups=ds.sample_numbers.to_numpy())):
            X_train, _ = X.iloc[train_idx], X.iloc[test_idx]
            Y_train, _ = Y.iloc[train_idx], Y.iloc[test_idx]
            # Train final model on full outer training set (no early stopping)
            params[outer_fold].update({'early_stopping':False,
                        'random_state':self._random_state,
                        'categorical_features':ds.categorical_columns})
            final_model = MultiOutputRegressor(HistGradientBoostingRegressor(**params[outer_fold])) if multioutput else \
                HistGradientBoostingRegressor(**params[outer_fold])
            final_model.fit(X_train, Y_train)

            outer_models.append(final_model)
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