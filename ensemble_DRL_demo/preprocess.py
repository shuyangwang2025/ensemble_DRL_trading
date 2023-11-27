import pandas as pd
import numpy as np
from stockstats import StockDataFrame
import os
import warnings
import copy
from dateutil.relativedelta import relativedelta
from datetime import datetime
from utils import Params, set_random_seeds
warnings.filterwarnings('ignore')

raw_data_dir = '../kraken-raw-data'
data_dir = '../processed-data'


class Preprocessor:
    def __init__(self):
        data_params = Params(json_path=f'{data_dir}/data_spec.json')
        self.coin_list = data_params.coin_list
        self.technical_indicators = data_params.technical_indicators
        self.raw_data_path = raw_data_dir
        self.data_path = data_dir

        self.data_start = datetime.strptime(data_params.train_start, '%Y-%m-%d')
        self.train_period = data_params.train_period
        self.validation_period = data_params.validation_period
        self.test_period = data_params.test_period
        self.n_models = data_params.n_models
        self.n_past = data_params.n_past

        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)

        return

    def preprocess_train_data(self, loop):
        data_subdir = os.path.join(self.data_path, f'loop{loop}_data')

        if os.path.isdir(data_subdir) == False:
            os.mkdir(data_subdir)

        train_start = self.data_start + relativedelta(months=self.test_period * (loop - 1))
        train_end = train_start + relativedelta(months=self.train_period)
        print(f'train: {train_start.strftime("%Y-%m-%d")} to {train_end.strftime("%Y-%m-%d")}')
        train_dataset_dict = self.load_data(train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d"))
        state_data, close_change, close_price = self.prepare_unnormalized_dataset(train_dataset_dict)
        norm_stats = self._get_norm_stats_dict(state_data)
        state_data, close_change, close_price = self.normalize_dataset(state_data, close_change, close_price, norm_stats)
        self.save_preprocessed_dataframe(state_data, close_change, close_price, data_subdir)

        return norm_stats

    def preprocess_val_data(self, loop, norm_stats_dict):
        val_data_subdir = os.path.join(self.data_path, f'loop{loop}_data', f'weekly_val_data')
        if not os.path.exists(val_data_subdir):
            os.makedirs(val_data_subdir)
        val_period_length = relativedelta(weeks=self.validation_period)

        train_start = self.data_start + relativedelta(months=self.test_period * (loop - 1))
        train_end = train_start + relativedelta(months=self.train_period)

        for n_val in range(self.n_models):
            n_days = int(np.random.choice(np.arange(1, 180)))
            val_start = train_start + relativedelta(days=n_days)
            val_end = val_start + val_period_length
            while val_end >= train_end:
                n_days = int(np.random.choice(np.arange(1, 180)))
                val_start = train_start + relativedelta(days=n_days)
                val_end = val_start + val_period_length
            print(f'Weekly validation {n_val}: {val_start.strftime("%Y-%m-%d")} to {val_end.strftime("%Y-%m-%d")}')
            state_data, close_change, close_price = self.preprocess_val_test_data_helper(val_start, val_end, norm_stats_dict)
            self.save_preprocessed_dataframe(state_data, close_change, close_price, val_data_subdir, prefix=f'val{n_val}_')

        return

    def preprocess_weekly_test_data(self, loop, norm_stats):
        test_data_subdir = os.path.join(self.data_path, f'loop{loop}_data', f'weekly_test_data')
        if not os.path.exists(test_data_subdir):
            os.mkdir(test_data_subdir)

        train_start = self.data_start + relativedelta(months=self.test_period * (loop - 1))
        train_end = train_start + relativedelta(months=self.train_period)
        test_start = train_end
        test_end = test_start + relativedelta(months=self.test_period)

        weekly_test_starts = []
        weekly_test_ends = []

        weekly_start = test_start
        while weekly_start.isoweekday() != 1:
            weekly_start = weekly_start + relativedelta(days=1)

        while weekly_start < test_end:
            weekly_end = weekly_start + relativedelta(weeks=1)
            if loop == 48 and weekly_end > test_end:
                break
            weekly_test_starts.append(weekly_start)
            weekly_test_ends.append(weekly_end)
            weekly_start = weekly_end

        for week_idx, (weekly_start, weekly_end) in enumerate(zip(weekly_test_starts, weekly_test_ends)):
            print(f'Weekly test {week_idx} {weekly_start.strftime("%Y-%m-%d")} to {weekly_end.strftime("%Y-%m-%d")}')
            state_data, close_change, close_price = self.preprocess_val_test_data_helper(weekly_start, weekly_end, norm_stats)
            self.save_preprocessed_dataframe(state_data, close_change, close_price, test_data_subdir, prefix=f'week{week_idx}_')
        print('\n')

        return

    def preprocess_val_test_data_helper(self, start_date, end_date, norm_stats):
        start_date = start_date - relativedelta(hours=self.n_past)
        dataset_dict = self.load_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        all_state_data, all_close_change, all_close_price = self.prepare_unnormalized_dataset(dataset_dict)
        state_data, close_change, close_price = self.normalize_dataset(all_state_data, all_close_change, all_close_price, norm_stats)
        return state_data, close_change, close_price

    def save_preprocessed_dataframe(self, state_data, close_change, close_price, data_subdir, prefix=''):
        state_data.to_pickle(data_subdir+f'/{prefix}state_data.pkl')
        close_change.to_pickle(data_subdir + f'/{prefix}close_change.pkl')
        close_price.to_pickle(data_subdir + f'/{prefix}close_price.pkl')

        return

    def preprocess_data(self, loop):
        print(f'Loop {loop}')
        norm_stats_dict = self.preprocess_train_data(loop)
        self.preprocess_val_data(loop, norm_stats_dict)
        self.preprocess_weekly_test_data(loop, norm_stats_dict)
        return

    def load_data(self, start_date, end_date):
        dataset_dict = {}
        for coin in self.coin_list:
            raw_data = pd.read_csv(os.path.join(self.raw_data_path, f'{coin}USD_60.csv'), header=None)
            dataset = self._select_data(raw_data, start_date=start_date, end_date=end_date)
            dataset_dict[coin] = dataset
        common = dataset_dict[self.coin_list[0]].index
        for coin in self.coin_list:
            common = common.intersection(dataset_dict[coin].index)
        for coin in self.coin_list:
            dataset_dict[coin] = dataset_dict[coin].loc[common]

        return dataset_dict

    def _select_data(self, raw_data, start_date=None, end_date=None, start_time=" 00:00:00", end_time=" 00:00:00"):
        # Slice and keep only the relevant data
        raw_data = raw_data[raw_data.columns[:-1]]
        raw_data.columns = ["date", "open", "high", "low", "close", "volume"]
        raw_data.set_index('date', inplace=True)
        raw_data.index = pd.to_datetime(raw_data.index, origin='unix', unit='s')
        dataset = StockDataFrame.retype(raw_data)
        if start_date is None or end_date is None:
            dataset = dataset[::-1]
            return dataset
        start_idx = datetime.strptime(start_date + start_time, '%Y-%m-%d %H:%M:%S')
        end_idx = datetime.strptime(end_date + end_time, '%Y-%m-%d %H:%M:%S') - relativedelta(hours=1)
        dataset = dataset.loc[start_idx:end_idx]
        return dataset

    def _get_norm_stats_dict(self, state_data):
        all_returns = state_data[['open', 'high', 'low', 'close', 'volume']]
        all_sma = state_data[['close_30_sma', 'close_60_sma']]
        returns_mean = all_returns.values.mean()
        returns_std = all_returns.values.std(ddof=1)
        macd_mean = state_data[['macd']].values.mean()
        atr_mean = state_data[['atr']].values.mean()
        cci_mean = state_data[['cci']].values.mean()
        macd_std = state_data[['macd']].values.std(ddof=1)
        atr_std = state_data[['atr']].values.std(ddof=1)
        cci_std = state_data[['cci']].values.std(ddof=1)
        sma_mean = all_sma.values.mean()
        sma_std = all_sma.values.std(ddof=1)
        norm_stats = [returns_mean, returns_std, macd_mean, macd_std, atr_mean, atr_std, cci_mean, cci_std, sma_mean, sma_std]

        return norm_stats

    def _normalize_state_data(self, state_data, norm_stats):
        normalized_state_data = copy.deepcopy(state_data)
        returns_mean, returns_std, macd_mean, macd_std, atr_mean, atr_std, cci_mean, cci_std, sma_mean, sma_std = norm_stats


        normalized_state_data[['open', 'high', 'low', 'close', 'volume']] = (state_data[['open', 'high', 'low', 'close',
                                                                                         'volume']] - returns_mean) / returns_std
        normalized_state_data[['macd']] = (state_data[['macd']] - macd_mean) / macd_std
        normalized_state_data[['atr']] = (state_data[['atr']] - atr_mean) / atr_std
        normalized_state_data[['cci']] = (state_data[['cci']] - cci_mean) / cci_std

        normalized_state_data[['rsi']] = (state_data[['rsi']] - 50.0) / 100.0
        normalized_state_data[['adx']] = (state_data[['adx']] - 50.0) / 100.0
        normalized_state_data[['close_30_sma', 'close_60_sma']] = (state_data[['close_30_sma',
                                                                               'close_60_sma']] - sma_mean) / sma_std
        return normalized_state_data

    def normalize_dataset(self, state_data, close_change, close_price, norm_stats):
        # Normalize
        normalized_state_data = self._normalize_state_data(state_data, norm_stats)
        normalized_state_data.dropna(inplace=True)
        common = normalized_state_data.index
        close_change = close_change.loc[common]
        close_price = close_price.loc[common]

        return normalized_state_data, close_change, close_price

    def prepare_unnormalized_dataset(self, dataset_dict):
        common = dataset_dict[self.coin_list[0]].index
        state_data_dict = {}
        close_price_dict = {}
        close_change_dict = {}
        for coin in self.coin_list:
            dataset = dataset_dict[coin]
            state_data, close_price, close_change = self._prepare_dataset_helper(dataset)
            common = common.intersection(state_data.index)
            state_data_dict[coin] = state_data
            close_price_dict[coin] = close_price
            close_change_dict[coin] = close_change

        for coin in self.coin_list:
            state_data_dict[coin] = state_data_dict[coin].loc[common]
            close_price_dict[coin] = close_price_dict[coin].loc[common]
            close_change_dict[coin] = close_change_dict[coin].loc[common]

        all_state_data = pd.concat(state_data_dict.values(), axis=1)
        all_close_change = pd.concat(close_change_dict.values(), axis=1)
        all_close_change.columns = list(close_change_dict.keys())
        all_close_price = pd.concat(close_price_dict.values(), axis=1)
        all_close_price.columns = list(close_price_dict.keys())

        return all_state_data, all_close_change, all_close_price

    def _prepare_dataset_helper(self, data):
        data.dropna(inplace=True)
        data = data[data['volume'] > 1e-10]
        df = StockDataFrame.retype(data)

        indicators1 = df[self.technical_indicators]
        indicators2 = df[["open", "high", "low", "close", "volume"]]

        indicators2 = indicators2.pct_change()
        indicators2.dropna(inplace=True)
        state_data = pd.merge(indicators1, indicators2, on='date')
        state_data.dropna(inplace=True)

        # Remove outliers in volume change
        q_low = state_data["volume"].quantile(0.001)
        q_hi = state_data["volume"].quantile(0.999)
        state_data = state_data[(state_data["volume"] < q_hi) & (state_data["volume"] > q_low)]

        # Save the raw close price and close change
        close_change = state_data["close"]  # Percentage change of the close price
        close_change.rename('close_change', inplace=True)
        close_price = df.loc[state_data.index, "close"]
        close_price.rename('close_price', inplace=True)

        return state_data, close_price, close_change


def load_preprocessed_train_data(loop, data_dir=data_dir):
    data_subdir = os.path.join(data_dir, f'loop{loop}_data')
    state_data = pd.read_pickle(data_subdir + f'/state_data.pkl')
    close_change = pd.read_pickle(data_subdir + f'/close_change.pkl')
    close_price = pd.read_pickle(data_subdir + f'/close_price.pkl')

    train_data = state_data, close_change, close_price

    return train_data


def load_preprocessed_val_data(loop, data_dir=data_dir, n_models=9):
    val_data_subdir = os.path.join(data_dir, f'loop{loop}_data', f'weekly_val_data')
    val_data = []
    for n_val in range(n_models):
        val_state_data = pd.read_pickle(val_data_subdir + f'/val{n_val}_state_data.pkl')
        val_close_change = pd.read_pickle(val_data_subdir + f'/val{n_val}_close_change.pkl')
        val_close_price = pd.read_pickle(val_data_subdir + f'/val{n_val}_close_price.pkl')

        val = [val_state_data, val_close_change, val_close_price]
        val_data.append(val)

    return val_data


def load_preprocessed_weekly_test_data(loop, data_dir=data_dir):
    test_data_subdir = os.path.join(data_dir, f'loop{loop}_data', f'weekly_test_data')
    test_data = []
    week_idx = 0
    path = os.path.join(test_data_subdir, f'week{week_idx}_state_data.pkl')
    while os.path.exists(path):
        test_state_data = pd.read_pickle(test_data_subdir + f'/week{week_idx}_state_data.pkl')
        test_close_change = pd.read_pickle(test_data_subdir + f'/week{week_idx}_close_change.pkl')
        test_close_price = pd.read_pickle(test_data_subdir + f'/week{week_idx}_close_price.pkl')

        test = test_state_data, test_close_change, test_close_price
        test_data.append(test)

        week_idx += 1
        path = os.path.join(test_data_subdir, f'week{week_idx}_state_data.pkl')

    return test_data


def load_preprocessed_data(loop, data_dir=data_dir):
    train_data = load_preprocessed_train_data(loop, data_dir)
    val_data = load_preprocessed_val_data(loop, data_dir)
    test_data = load_preprocessed_weekly_test_data(loop, data_dir)

    return train_data, val_data, test_data


if __name__ == '__main__':
    set_random_seeds(42)
    p = Preprocessor()
    for i in range(1, 49):
        p.preprocess_data(i)