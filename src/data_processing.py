import numpy as np
import os
import pandas as pd
import re
from collections import defaultdict
from tqdm import tqdm
from sys import stderr
from typing import Literal, Iterable, LiteralString

'''Create object to processes the spinning data.

Enables methods reading of spinning condition data and material property targets.
Links toghether the spinnig condition features and targets from different files.
Also enables methods for sanatization and missing value handling.
Methods work inplace to modify the state.
Also provides methods for saving dataset to Excel or HDF format.

Data is stored as a Pandas DataFrame, access using the df property.'''
class DataProcessing:
    """'data_dir' should be path to a folder containing both the 'spinning experiments overview.xlsx', 
    and the corresponding mechanical property Excel sheets."""
    def __init__(self, 
                 data_dir: str | LiteralString = os.path.join(os.path.pardir, 'data')) -> None:
        self._df: pd.DataFrame = pd.DataFrame()
        self._targets_df = pd.DataFrame(columns=['Sample number', 'Diameter (µm)', 'Strain (mm/mm)', 'Strength (MPa)', 'Youngs Modulus (GPa)', 'Toughness Modulus (MJ m-3)'])
        self._data_dir = data_dir
        self._unnamed_sample_count = 0
        self._grouped_samples = False
        self._ind_to_col = {
            0 : "Experiment",
            1 : "Sample number",
            2 : "Protein",
            3 : "Batch",
            4 : "Date Purified",
            5 : "column",
            6 : "Dope prepared",
            7 : "concentration (mg/ml)",
            8 : "Spinning device",
            9 : "Extrusion device",
            10 : "spinning",
            11 : "Bath length (cm)",
            12 : "Temperature SB",
            13 : "Spinning Buffer",
            14 : "SB pH",
            15 : "SB conc. (mM)",
            16 : "NaCl (mM)",
            17 : "Capillery size (um)",
            18 : "Reeling speed (rpm)",
            19 : "Flow rate (ul/min)",
            20 : "pumppressure (bar)",
            21 : "Post spin treatment",
            22 : "Temp C (spinning)",
            23 : "Humidity (spinning)",
            24 : "Continous spinning",
            25 : "Comment",
            26 : "fibers prepared",
            27 : "mechanical testing date",
            28 : "# fibers",
            29 : "Diameter (µm)",
            30 : "Strain (mm/mm)",
            31 : "Strength (MPa)",
            32 : "Youngs Modulus (GPa)",
            33 : "Toughness Modulus (MJ m-3)",
            34 : "Water Soluble (10 min after spin)",
            35 : "Water Soluble (>7 days after spin)",
            36 : "Temp C (tensile test)",
            37 : "Humidity (tensile test)"}
        self._col_to_ind = {v: k for k, v in self._ind_to_col.items()}

        self._sample_column = ''
        self._features_columns = []
        self._targets_columns = []

    """Uses pandas.DataFrame implementation."""
    def __str__(self) -> str:
        return str(self._df)

    """Uses pandas.DataFrame implementation."""
    def __repr__(self) -> str:
        return repr(self._df)

    """Uses pandas.DataFrame implementation."""
    def to_string(self) -> str:
        return self._df.to_string()

    """Get the entire dataframe of the processed dataset."""
    @property
    def df(self) -> pd.DataFrame:
        return self._df

    """Get the spinning conditions of the processed dataset."""
    @property
    def features(self):
        return self._df.iloc[:,1:-5]

    """Get the mechanical property values of the processed dataset."""
    @property
    def targets(self):
        return self._df.iloc[:,-5:]

    """Convert values such as '?', to numpy.nan."""
    def standardize_missing_data(self) -> None:
        # Bypass warning.
        missing_vals = ['?', '???', 'conc unknown', '? Break', 'not collected', 'n.a']
        with pd.option_context("future.no_silent_downcasting", True):
            self._df = self._df.replace(missing_vals, np.nan).infer_objects() 
        

    """Load the 'Spinning experiements overview.xlsx' file. 
    Drops uninteresting columns, such as dates.
    Also converts the 'Sample number' column values (except NaNs) to string format."""
    def load_spinning_experiments_excel(self,
                            fname: str | LiteralString = 'Spinning experiments overview.xlsx',
                            na_targets: bool = True) -> None:
        sample_ind = 1
        feature_inds = [2,7,8,9,11,12,13,14,15,16,17,18,19,20,22,23,24]
        target_inds = [29,30,31,32,33]
        inds = [sample_ind] + feature_inds + target_inds

        fp = os.path.join(self._data_dir, fname)
        self._df = pd.read_excel(fp, usecols=inds, skiprows=2)
        self._df = self._df.rename({
            'strain (mm/mm)': 'Strain (mm/mm)',
            'strength (MPa)': 'Strength (MPa)',
            'Youngs Modulus (Gpa)': 'Youngs Modulus (GPa)',
            'Toughness (MJ m-3)': 'Toughness Modulus (MJ m-3)'}, axis=1)

        self.standardize_missing_data()

        # Convert all numbers (excepts na's) to strings.
        non_na_inds = self._df['Sample number'][self._df['Sample number'].notna()].index
        self._df.loc[non_na_inds, 'Sample number'] = self._df.loc[non_na_inds, 'Sample number'].astype(str)

        self._sample_column = self._df.columns[0]
        self._features_columns = list(self._df.columns[1:-5])
        self._targets_columns = list(self._df.columns[-5:])

        if na_targets:
            self._df.loc[:,self._targets_columns] = np.nan

    """Label unnamed values for the 'Sample number' column to: '<prefix><sep><count>'."""
    def label_unnamed_samples(self, prefix: str ='unnamed', sep: str = '_') -> None:
        for i, isna in enumerate(self._df['Sample number'].isna()):
            if isna:
                self._df.loc[i, 'Sample number'] = f'{prefix}{sep}{self._unnamed_sample_count}'
                self._unnamed_sample_count += 1
        assert(self._df['Sample number'].notna().all())

    """Label duplicate values for the 'Sample number' column to: '<sample><sep><count>."""
    def label_duplicate_samples(self, sep: str = '-') -> None:
        duplicates = defaultdict(lambda: 0) 
        for i, sample in self._df.loc[self._df['Sample number'].duplicated(False), 'Sample number'].items():
            if isinstance(i, int):
                self._df.loc[i, 'Sample number'] = f'{sample}{sep}{duplicates[sample]}'
            else: 
                raise TypeError(f'Index: {i}, is not of type int')
            duplicates[sample] += 1
        assert(self._df['Sample number'].is_unique)

    """Merge data from 'Spinning experiments overview.xlsx' with mechanical property values 
    from the 'all...xlsx' files. Uses a 'left' merging stratergy, keeping only column values for 
    the spinning condition (features) dataset."""
    def merge_targets(self):
        # Add all non-nan targets from the spinning data to the target dataframe.
        t = self._df[[self._sample_column] + list(self._targets_columns)]
        t = t.drop(t[t.isna().any(axis=1)].index)
        self._targets_df = pd.concat((self._targets_df, t), ignore_index=True) 
        assert(self._targets_df.notna().any(axis=None))
        self._df = self._df.drop(self._targets_columns, axis=1) \
            .merge(self._targets_df, 'left')
        assert(self._df['Sample number'].notna().all())
        self._df['Sample number'] = self._df['Sample number'].astype(str)
        #assert(self._df[self._targets_columns].notna().any(axis=None))

    """Drops data points where the target values are NaNs. 
    'all_nans' flag controls if dropping only if all mechanical property value is None or any."""
    def drop_na_targets(self, all_nans: bool = False):
        if all_nans:
            self._df = self._df[self._df[self._targets_columns].notna().any(axis=1)].reset_index(drop=True)
        else:
            self._df = self._df[self._df[self._targets_columns].notna().all(axis=1)].reset_index(drop=True)

    '''Collect data points with identical sample number to one data point, stored as a list.'''
    def group_samples(self) -> None:
        if self._grouped_samples:
            return
        function_dict = {}
        for col in self._df[self._features_columns]:
            function_dict[col] = 'first'
        for col in self._df[self._targets_columns]:
            function_dict[col] = list
        self._df = self._df.groupby(self._df['Sample number'], as_index=False).agg(function_dict)
        self._grouped_samples = True

    '''Divide target lists data points with sample number to multiple data points.'''
    def ungroup_samples(self) -> None:
        if not self._grouped_samples:
            return
        df = pd.DataFrame(columns = self._df.columns)
        for _, row in self._df.iterrows():
            features = row[:-5]
            targets = row[-5:]
            rows = pd.DataFrame(columns = self._df.columns)
            for i in range(len(targets.iat[0])):
                targets_i = pd.Series([targets.iat[0][i], targets.iat[1][i], targets.iat[2][i], targets.iat[3][i], targets.iat[4][i]], self._targets_columns)
                row_i = pd.DataFrame(pd.concat((features, targets_i))).T
                rows = row_i.copy() if rows.empty else pd.concat((rows, row_i), ignore_index=True)
            df = rows.copy() if df.empty else pd.concat((df, rows), ignore_index=True)
        self._df = df 
        self._grouped_samples = False

    """Save dataframe to CSV."""
    def to_csv(self,
               fname: str | os.PathLike = 'spinning_data.csv') -> None:
        fp = os.path.join(self._data_dir, fname)
        self._df.to_csv(fp, index=False)

    """Save dataframe to HDF."""
    def to_hdf(self,
               fname: str | os.PathLike | pd.HDFStore = 'spinning_data.hf5') -> None:
        fp = os.path.join(self._data_dir, fname)
        self._df.to_hdf(fp, key='spinning_data', mode='w')

    """Save dataframe to an Excel sheet.
    If 'aggrigate_samples' is 'True', Identical samples are aggrigated and 
    mechanical property values are stored as a comma-separated list."""
    def to_excel(self, 
                  fname: str | LiteralString = 'spinning_data.xlsx') -> None:
        df = self._df.copy()
        if self._grouped_samples:
            for col in self._targets_columns:
                df[col] = df[col].astype(str).apply(lambda a: str(a)[1:-1])
        fp = os.path.join(self._data_dir, fname)
        df.to_excel(fp, sheet_name='data', index=False)

    """Read processed dataset from HDF."""
    def read_hdf(self, 
                 fname: str | os.PathLike | pd.HDFStore = 'spinning_data.hf5') -> None:
        fp = os.path.join(self._data_dir, fname)
        self._df = pd.DataFrame(pd.read_hdf(fp, key='spinning_data', mode='r'))

    """Loads processed HDF file if saved, otherwise reads Excel sheets."""
    def load_targets(self) -> None:
        hdf_fp = os.path.join(self._data_dir, 'targets.csv')
        n = 0
        if os.path.exists(hdf_fp):
            self.targets_read_csv()
            n = len(self._targets_df)
        self.targets_read_excel()
        if len(self._targets_df) > n: 
            self.targets_to_csv()

    """Load mechanical property values from Excel sheets with 
    naming scheme: 'all <sample name>.xlsx'."""
    def targets_read_excel(self) -> None:
        dir = os.path.join(self._data_dir, 'mechanical_properties')
        pbar = tqdm(os.listdir(dir))
        for fname in pbar:
            # Ignore all unsuitable files or file is already processed.
            sample = DataProcessing.rename_target_sample(fname[4:-5].strip()) # Strip 'all ' and '.xlsx' from file name.
            if not (fname.endswith(r'.xlsx') and 'all' in fname.lower()) or sample in self._targets_df['Sample number'].to_numpy():
                continue
            pbar.set_description(f'Processing file "{fname}"')
            fp = os.path.join(dir, fname)

            # Some sheets have naming variations.
            for sheet_name in ['all', 'all ', 'ALL', 'Tabelle11', 'Summary']:
                try:
                    df = pd.read_excel(fp, sheet_name)
                    break
                except ValueError:
                    pass
            else:
                print(f'Invalid sheet name for sample: {sample}', file=stderr)
                continue

            # The diameter column have naming variations in some sheets.
            if sheet_name == 'Summary':
                try:
                    data = df.iloc[2:12,2:7]
                except KeyError:
                    print(f'Could not read spinning conditions for sample: {sample}', file=stderr)
                    continue
            else:
                for diameter_name in ['diameter', 'Diameter', 'diametro', 'diameter ', 'DIAMETER', 'Diameter (um)']:
                    try:
                        i = df.columns.get_loc(diameter_name)
                        break
                    except KeyError:
                        pass
                else:
                    print(f'Could not find diameter for sample: {sample}', file=stderr)
                    continue
                if isinstance(i, int):
                    data = df.iloc[:10, i:i+5]
                else:
                    raise TypeError(f'Index: {i}, is not of type int')

            rows = pd.DataFrame(data.to_numpy(), columns=self._targets_df.columns[1:])
            if re.compile(r'(b|B)\d+-\d+').fullmatch(sample):
                beginning, end = sample.split('-')
                for num in np.arange(int(beginning[1:]), int(end) + 1):
                    sample_i = 'B' + str(num)
                    if sample_i in self._targets_df['Sample number'].to_numpy():
                        continue
                    rows_cpy = rows.copy()
                    rows_cpy.insert(0, 'Sample number', sample_i)
                    self._targets_df = rows.copy() if self._targets_df.empty else pd.concat((self._targets_df, rows_cpy), ignore_index=True)
            else:
                rows.insert(0, 'Sample number', sample)
                self._targets_df = rows.copy() if self._targets_df.empty else pd.concat((self._targets_df, rows), ignore_index=True)
    
    """Renames sample name if necessary based on regular expressions."""
    @staticmethod
    def rename_target_sample(sample: str) -> str:
        if re.compile(r'(b|B)\d+(-|_)Farnaz').fullmatch(sample):
            if '-' in sample:
                return sample.split('-')[0].upper()
            else:
                return sample.split('_')[0].upper()
        # Rename "bxxx..." -> "Bxxx".
        if re.compile(r'(b|B)\d+').match(sample):
            return sample.split()[0].upper()
        # Rename "Benjamin xxx ..." -> "Bxxx"
        if re.compile(r'(b|B)enjamin \d+').match(sample):
            return 'B' + sample.split()[1]
        if re.compile(r'\d+').fullmatch(sample):
            return sample
        raise ValueError(f'Invalid sample: {sample}')

    @staticmethod
    def rename_sample_number(sample: str) -> str:
        if re.compile(r'(b|B)\d+').match(sample):
            return sample.split()[0].upper()
        return sample

    """Save processed targets to CSV."""
    def targets_to_csv(self, fname: str | os.PathLike = 'targets.csv') -> None: 
        fp = os.path.join(self._data_dir, fname)
        self._targets_df.to_csv(fp, index=False)

    """Load processed targets from CSV."""
    def targets_read_csv(self, fname: str | os.PathLike = 'targets.csv') -> None: 
        fp = os.path.join(self._data_dir, fname)
        self._targets_df = pd.read_csv(fp)

    """Save processed targets to HDF."""
    def targets_to_hdf(self, fname: str | os.PathLike | pd.HDFStore = 'targets.hf5', mode: Literal['a', 'w', 'r+']='w') -> None: 
        fp = os.path.join(self._data_dir, fname)
        self._targets_df.to_hdf(fp, key='targets', mode=mode)

    """Load processed targets from HDF."""
    def targets_read_hdf(self, fname: str | os.PathLike | pd.HDFStore = 'targets.hf5') -> None: 
        fp = os.path.join(self._data_dir, fname)
        self._targets_df = pd.read_hdf(fp, key='targets', mode='r')

    """Fill NaNs for a certain column.
    Exactly one of 'value' or 'mode' must be specified.
    If value is specified, replace each NaN with that value.
    Mode can be either 'mean', 'median', 'mode', 'min' or 'max'. 
    If mode is specified, replace NaN by the corresponding statistic/strategy."""
    def fill_na(self, column: str, value: int | float | str | None = None, mode: str | None = None) -> None:
        if (value is None and mode is None) or (value is not None and mode is not None):
            raise ValueError("Exactly one of 'value' or 'mode' must be specified.")
        if mode is not None:
            match mode:
                case 'mean':
                    value = self._df[column].mean()
                case 'median':
                    value = self._df[column].median()
                case 'mode':
                    value = self._df[column].mode()[0]
                case 'max':
                    value = self._df[column].max()
                case 'min':
                    value = self._df[column].min()
                case _:
                    raise NotImplementedError(f"'mode = {mode}' not implemented.")
        self._df[column] = self._df[column].fillna(value)
            
    """Return all samples with sample name 'sample' as a Pandas DataFrame."""
    def get_samples(self, samples: Iterable[str] | pd.Series) -> pd.DataFrame:
        if type(samples) is str:
            samples = [samples]
        return self._df[self._df['Sample number'].isin(samples)]

    """Standardize column values, such as fixing small naming variations (e.g., uppercase/lowercase difference)."""
    def standardize_columns(self,
                            na_unknown_capilleries: bool = True) -> None:
        col = 'Sample number'
        self._df[col] = self._df[col].astype(str).apply(DataProcessing.rename_sample_number)

        col = 'Protein'
        protein_replace = {'A3I-A': 'A3IA',
             ' A3I-A': 'A3IA', 
             'NT2repCT': 'NT2RepCT',
             'Rep2 Resilin': 'Rep2',
             'Rep5 ': 'Rep5',
             '3Rep': 'Rep3',
             'Rep3 Elastin': 'Rep3',
             'Rep3 Elastin short': 'Rep3',
             ' Rep3 Elastin short': 'Rep3',
             'Rep4 Elastin long': 'Rep4',
             'Rep7 Tusp': 'Rep7',
             'pAAAIpA': 'A3IA',
             'fNT-A3IA-MCT': 'fNT A3IA',
             'BrMasp2 300': 'Br_MaSp2_300',
             'BrMasp2 400': 'Br_MaSp2_400',
             'Br_Masp2long': 'Br_MaSp2_long',
             'Br_Masp2short': 'Br_MaSp2_short',
             'Br_Masp2long': 'Br_MaSp2_long',
             'Br_Masp4long': 'Br_MaSp4_long',
             'Br_Masp4short': 'Br_MaSp4_short'}
        self._df[col] = self._df[col]\
            .replace(protein_replace)
            #.replace(['A3I-A', ' A3I-A', 'Rep5 ', ' Rep3 Elastin short'], ['A3IA', 'A3IA', 'Rep5', 'Rep3 Elastin short'])

        col = 'Bath length (cm)'
        self._df[col] = self._df[col].replace('2 bath', '2bath')

        col = 'Continous spinning'
        self._df[col] = self._df[col].replace(r'yes.*', 'yes', regex=True)
        #self._df['Continous spinning'] = self._df['Continous spinning'].replace([r'.*yes.*', r'(.*no.*|discontinous)'], ['yes', 'no'], regex=True)

        # Convert PSI to BAR.    
        col = 'pumppressure (bar)'
        psi_to_bar = .0689475729 
        psis = self._df[col].astype(str).str.contains(r'[pP][sS][iI]')
        self._df.loc[psis, col] = self._df.loc[psis, col]\
            .apply(lambda s: str(psi_to_bar * float(s[:-3])))


        # 1 bath or 2 bath
        col = 'Bath length (cm)'
        self._df.loc[self._df[col].notna(), col] = \
            self._df.loc[self._df[col].notna(), col]\
            .astype(str).replace(r'^\d+$', '1', regex=True)
        with pd.option_context("future.no_silent_downcasting", True):
            self._df[col] = self._df[col].replace(r'2bath', '2').infer_objects()
        self._df.loc[:,col] = self._df.loc[:,col].astype('object')
        new_col = 'Number of baths'
        self._df = self._df.rename(columns={col : new_col})
        self._features_columns[self._features_columns.index(col)] = new_col

        # Extrusion Device
        col = 'Extrusion device'
        self._df[col] = self._df[col]\
            .replace([r'^HPLC pump.*$', r'^Syringe Pump$'], ['HPLC pump', 'Syringe pump'], regex=True)

        col = 'Spinning device'
        self._df[col] = self._df[col].replace(['hulk', 'Hullk'], 'Hulk')

        col = 'Temp C (spinning)'
        self._df[col] = self._df[col].replace(('21..5'), 21.5)

        col = 'Reeling speed (rpm)'
        self._df[col] = self._df[col]\
            .replace(['>200', 'manually', '30&55'], [np.nan, np.nan, (30+55)/2])

        range_m_min = self._df[col].astype(str).str.contains(r'^\d+ \(?m/min\)?$')
        #range_rpm = self._df[col].astype(str).str.contains(r'^\d+$')
        wheel_diameter = .11 
        self._df.loc[range_m_min, col] = self._df.loc[range_m_min, col]\
            .apply(lambda speed: rpm_to_meter_per_minute(float(speed.split()[0]), wheel_diameter))
        # TODO: Include different setup if manual, rpm, or m/min.

        col = 'Flow rate (ul/min)'
        range = self._df[col].astype(str).str.contains(r'\d+-\d+')
        self._df.loc[range, col] = self._df.loc[range, col]\
            .apply(lambda s: np.mean([float(n.split()[0]) for n in s.split('-')]))
        range = self._df[col].astype(str).str.contains(r'\d+ & \d+')
        self._df.loc[range, col] = self._df.loc[range, col]\
            .apply(lambda s: np.mean([float(n) for n in s.split(' & ')]))


        col = 'concentration (mg/ml)'
        range = self._df[col].astype(str).str.contains(r'~ ?\d+')
        self._df.loc[range, col] = self._df.loc[range, col]\
            .apply(lambda s: float(s.split('~')[-1].strip()))
        with pd.option_context("future.no_silent_downcasting", True):
            self._df[col] = self._df[col].replace(r'17% in HFIP', np.nan).infer_objects()


        # Encode capillery as 3 categories and save length as numeric.
        col = 'Capillery size (um)'
        col_type = 'Capillery type'
        loc = self._df.columns.get_loc(col)
        if isinstance(loc, int):
            self._df.insert(loc, col_type, np.nan)
        else:
            raise TypeError(f'Index: {loc}, is not of type int')
        self._df[col_type] = self._df[col_type].astype('object')
        self._features_columns.insert(self._features_columns.index(col), col_type)
        self._df[col] = self._df[col].replace(['45 and 67 um capillary', 'broken cap >100um'], [(45+67)/2, np.nan])

        range_range = self._df[col].astype(str).str.contains(r'\d+-\d+')
        self._df.loc[range_range, col] = self._df.loc[range_range, col]\
            .apply(lambda s: np.mean([float(n.split()[0]) for n in s.split('-')]))

        range_glass = self._df[col].astype(str).str\
            .match(r'(^\d+(\.\d+)?$)|(.*\d+ broken$)|(.*\d+ ?um$)$')
        self._df.loc[range_glass, col_type] = self._df.loc[range_glass, col_type] = 'Glass'


        range = self._df[col].astype(str).str\
            .contains(r'<|>')
        self._df.loc[range, col] = np.nan

        range = self._df[col].astype(str).str\
            .match(r'.*\d+ broken$')
        self._df.loc[range, col] = self._df.loc[range, col]\
            .apply(lambda s: int(s.split()[-2]))

        range = self._df[col].astype(str).str\
            .match(r'.*\d+ ?um$')
        self._df.loc[range, col] = self._df.loc[range, col]\
            .apply(lambda s: int(s.replace('um','').split()[-1]))


        range_tubing = self._df[col].astype(str).str.contains(r'^\d+ um ID PEEK tubing')
        self._df.loc[range_tubing, col_type] = self._df.loc[range_tubing, col_type] = 'PEEK tubing'
        self._df.loc[range_tubing, col] = self._df.loc[range_tubing, col]\
            .apply(lambda s: int(s.split()[0]))

        range_tecdia = self._df[col].astype(str).str.contains(r'Tecdia')
        self._df.loc[range_tecdia, col_type] = self._df.loc[range_tecdia, col_type] = 'Tecdia'
        self._df.loc[range_tecdia, col] = self._df.loc[range_tecdia, col]\
            .apply(lambda s: int(s.split()[-1].split('um')[0]))

        #self._df.astype(str).str.contains('Aspect Biosystems Duo-1')

        if na_unknown_capilleries:
            sets = [set(np.arange(len(self._df))[range]) for range in [range_glass, range_tubing, range_tecdia]]
            inds_invalid = list(set(self._df.index).difference(set().union(*sets)))
            self._df.loc[inds_invalid, [col]] = np.nan

        self._df.loc[:, self.numerical_features] = self._df.loc[:, self.numerical_features].astype(float)
        self._df.loc[:, self.categorical_features] = self._df.loc[:, self.categorical_features].astype('category')
        self._df.loc[:, self._targets_columns] = self._df.loc[:, self._targets_columns].astype(float)

    def sort(self) -> None:
        self._df.sort_values('Sample number', axis='index', ignore_index=True)

    @property
    def categorical_features(self) -> list[str]:
        return list(self._df.columns[[1,3,4,5,7,11,18]])

    @property
    def numerical_features(self) -> list[str]:
        return list(self._df.columns[[2,6,8,9,10,12,13,14,15,16,17]])

    def append_sequence_embeddings(self, csv_fname: str | os.PathLike='protein_embeddings_pca.csv') -> None:
        csv_path = os.path.join(os.pardir, 'data', csv_fname)
        df_pca = pd.read_csv(csv_path)
        num_embeddings = df_pca.shape[-1] - 1
        self._df = dp.df.merge(df_pca, how='inner', on='Protein')
        n = self._df.shape[-1]
        self._df = self._df.iloc[:, \
            list(range(n - 5 - num_embeddings)) + \
            list(range(n -num_embeddings, n)) + \
            list(range(n - 5 - num_embeddings, n - num_embeddings))]
        

def meter_per_minute_to_rpm(speed_m_min: float, wheel_diameter_m: float) -> float:
    return speed_m_min/(np.pi*wheel_diameter_m)

def rpm_to_meter_per_minute(speed_rpm: float, wheel_diameter_m: float) -> float:
    return np.pi*wheel_diameter_m*speed_rpm
        
        

if __name__ == '__main__':
    dp = DataProcessing()
    dp.load_spinning_experiments_excel(na_targets=False)
    dp.label_unnamed_samples()
    dp.label_duplicate_samples()
    dp.standardize_columns()
    dp.load_targets()
    dp.label_duplicate_samples()
    dp.merge_targets()
    dp.drop_na_targets()
    dp.group_samples()
    dp.to_excel()
    dp.ungroup_samples()
    dp.sort()
    dp.to_csv()
    dp.to_hdf()
    dp.append_sequence_embeddings()
    dp.to_csv('spinning_data_embeddings_pca.csv')
    dp.to_hdf('spinning_data_embeddings_pca.hf5')
    print(dp._df.shape)
    dp.group_samples()
    dp.to_excel('spinning_data_embeddings_pca.xlsx')
    