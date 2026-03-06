import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Literal

class Dataset():
    def __init__(self, 
                 source: pd.DataFrame | str = 'spinning_data.csv',
                 scaler: Literal['minmax', 'standard', 'none'] = 'minmax') -> None:
        if isinstance(source, pd.DataFrame):
            self._df = source.reset_index(drop=True)
        else:
            fp = os.path.join(os.pardir, 'data', source)
            if source[-3:] == 'csv':
                self._df = pd.DataFrame(pd.read_csv(fp))
            elif source[-3:] == 'hf5':
                self._df = pd.DataFrame(pd.read_hdf(fp, key='spinning_data', mode='r'))
            else:
                raise ValueError('Filetype must be either CSV or HDF.')
            
        #self._df.loc[:,'Number of baths'] = self._df.loc[:,'Number of baths'].astype(object)
        self._df.loc[:, self.numerical_columns] = self._df.loc[:, self.numerical_columns].astype(float)
        self._df.loc[:, self.categorical_columns] = self._df.loc[:, self.categorical_columns].astype('category')
        self._df.loc[:, self.target_columns] = self._df.loc[:, self.target_columns].astype(float)

        if scaler == 'none':
            return
        elif scaler == 'minmax':
            self._scaler = MinMaxScaler()
        elif scaler == 'standard':
            self._scaler = StandardScaler()
        self._df.iloc[:, -5:] = self._scaler.fit_transform(self._df.iloc[:, -5:])

    
    def __len__(self) -> int:
        return len(self._df)
        
    def __getitem__(self, idx: int | slice) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
        return self.features.iloc[idx], self.targets.iloc[idx]

    def __call__(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        X, Y = self[:] 
        assert(type(X) == pd.DataFrame and type(Y) == pd.DataFrame)
        return X, Y

    @property
    def features(self) -> pd.DataFrame:
        return self._df.iloc[:, 1:-len(self.target_columns)]

    @property
    def targets(self) -> pd.DataFrame:
        return self._df.loc[:, self.target_columns]

    @property
    def categorical_columns(self) -> list[str]:
        return list(set(self._df.columns).intersection({'Protein', 
            'Spinning device', 'Extrusion device', 'Number of baths', 
            'Spinning Buffer', 'Capillery type', 'Continous spinning'}))

    #@property
    #def categorical_features(self) -> list[int]:
    #    return [1,3,4,5,7,11,18]

    @property
    def numerical_columns(self) -> list[str]:
        return list(set(self._df.columns).difference(set(['Sample number'] + self.target_columns + self.categorical_columns)))

    #@property
    #def numerical_features(self) -> list[int]:
    #    return [2,6,8,9,10,12,13,14,15,16,17] + list(range(19, self._df.shape[1] - 5))

    @property
    def target_columns(self) -> list[str]:
        targets = []
        for target in 'Diameter (µm)',\
            'Strain (mm/mm)', 'Strength (MPa)',\
            'Youngs Modulus (GPa)', 'Toughness Modulus (MJ m-3)':
            if target in self._df.columns:
                targets.append(target)
        return targets

    #@property
    #def target_inds(self) -> list[int]:
    #    n = self._df.shape[1]
    #    return list(range(n - 5, n))

    @property
    def sample_numbers(self) -> pd.Series:
        return self._df.iloc[:, 0]

    @property
    def columns(self) -> list[str]:
        return list(self._df.columns)

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    """Uses pandas.DataFrame implementation."""
    def __str__(self) -> str:
        return str(self._df)

    """Uses pandas.DataFrame implementation."""
    def __repr__(self) -> str:
        return repr(self._df)

    """Uses pandas.DataFrame implementation."""
    def to_string(self) -> str:
        return self._df.to_string()

    def group_samples(self) -> pd.DataFrame:
        df = self._df.copy() 
        function_dict = {}
        for col in self._df.iloc[:, :-5]:
            function_dict[col] = 'first'
        for col in self.targets.columns:
            function_dict[col] = list
        return df.groupby(self._df['Sample number'], as_index=False).agg(function_dict)
        