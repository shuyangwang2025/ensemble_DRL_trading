import copy
import pickle
import os
import argparse
import numpy as np
import torch
import logging
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from GymEnv import TradingEnv
from Network import IndependentNormalModel
from utils import get_hparams, set_logger, set_random_seeds
from preprocess import load_preprocessed_data
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_results_folder = 'training_results'


class ModelTest:
    def __init__(self, hparams, loop, dataset):
        self.loop = loop
        self._unpack_datasets(dataset)
        self._set_hparams(hparams)
        self._prepare_envs_models()

        return

    def _prepare_envs_models(self):
        hparams = copy.deepcopy(self.hparams)

        # Create the environments
        all_games = []
        for week in range(self.n_weeks):
            week_hparams = copy.deepcopy(hparams)
            week_hparams.folder += '/test_plots'
            if not os.path.isdir(week_hparams.folder):
                os.mkdir(week_hparams.folder)
            week_hparams.folder += f'/week{week}'

            test_game = TradingEnv(self.loop, -2, self.concat_test_state_data_list[week],
                                   self.test_close_price_dict_list[week],
                                   self.test_close_change_dict_list[week], week_hparams)
            test_game.reset()
            all_games.append(test_game)

        self.all_games = all_games

        base_model = IndependentNormalModel(self.hparams).to(device)

        all_models = []
        for n_val in range(self.n_models):
            model = copy.deepcopy(base_model)
            best_network = self._get_network_path(self.loop, n_val)
            checkpoint = torch.load(best_network, map_location=torch.device(device))
            model.load_state_dict(checkpoint['model_state_dict'])
            # Compare with model at final epoch
            # latest_network = self._get_network_path(self.loop, n_val, 'latest')
            # final_checkpoint = torch.load(latest_network, map_location=torch.device(device))

            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            all_models.append(model)

        self.all_models = all_models

        return

    def _unpack_datasets(self, dataset):
        train_data, val_data, test_data = dataset

        self.concat_state_data, self.close_change_dict, self.close_price_dict = train_data

        self.concat_val_state_data_list, self.val_close_change_dict_list, self.val_close_price_dict_list = [], [], []
        for val in val_data:
            concat_val_state_data, val_close_change_dict, val_close_price_dict = val
            self.concat_val_state_data_list.append(concat_val_state_data)
            self.val_close_change_dict_list.append(val_close_change_dict)
            self.val_close_price_dict_list.append(val_close_price_dict)

        self.concat_test_state_data_list, self.test_close_change_dict_list, self.test_close_price_dict_list = [], [], []
        for test in test_data:
            concat_state_data, close_change_dict, close_price_dict = test
            self.concat_test_state_data_list.append(concat_state_data)
            self.test_close_change_dict_list.append(close_change_dict)
            self.test_close_price_dict_list.append(close_price_dict)

        self.coin_list = list(self.close_change_dict.keys())
        self.n_models = args.n_models
        self.n_weeks = len(test_data)

    def _set_hparams(self, hparams):
        self.hparams = hparams
        hmax_dict = {}
        baseline_price = self.close_price_dict[self.coin_list[0]].mean()
        hmax_dict[self.coin_list[0]] = 1
        for c_idx in range(1, len(self.coin_list)):
            if hparams.norm_hmax:
                hmax_dict[self.coin_list[c_idx]] = hmax_dict[self.coin_list[0]] * baseline_price / self.close_price_dict[self.coin_list[c_idx]].mean()
            else:
                hmax_dict[self.coin_list[c_idx]] = hparams.hmax
        self.hparams.hmax_dict = copy.deepcopy(hmax_dict)

        return

    def get_test_results(self):
        logging.info(f'Loop {self.loop}')
        test_profits = []
        for week in range(self.n_weeks):
            test_profit = self._evaluate_strategy(week=week)
            test_profits.append(test_profit)

        return test_profits

    def _evaluate_strategy(self, week=0):
        set_random_seeds(args.seed)
        models = self.all_models
        game = self.all_games[week]

        # Roll out the entire episode
        with torch.no_grad():
            done = False
            next_obs = game.reset(plot=True)
            while not done:
                actions = self._sample_from_mixture(models, next_obs)
                next_obs, _, done, info = game.step(actions)
            # done, finished one episode
            profit = info['profit']

        return profit

    def _sample_from_mixture(self, models, next_obs):
        weights = torch.ones(len(models), device=device)
        locs = []
        covs = []
        for model in models:
            loc, cov_mat_tril, _ = self._get_distribution_value(model, next_obs)
            locs.append(loc.squeeze())
            covs.append(cov_mat_tril.squeeze())
        all_loc = torch.stack(locs, dim=0)
        all_cov = torch.stack(covs, dim=0)
        mix = Categorical(weights)
        comp = MultivariateNormal(all_loc, scale_tril=all_cov)
        # TanhTransform
        comp = TransformedDistribution(comp, [TanhTransform(cache_size=1), ])
        gmm = MixtureSameFamily(mix, comp)

        actions = gmm.sample().cpu().numpy().squeeze()

        return actions

    def get_buy_hold_results(self):
        test_profits = []
        for week in range(self.n_weeks):
            test_profit = self._buy_hold_helper(self.test_close_price_dict_list[week])
            test_profits.append(test_profit)

        return test_profits

    def _buy_hold_helper(self, price_dict):
        portfolio_weights = np.zeros(len(self.coin_list)) + 1 / len(self.coin_list)
        coin_profits = []
        for c_idx, coin in enumerate(self.coin_list):
            start_price = price_dict[coin].iloc[0]
            end_price = price_dict[coin].iloc[-1]
            del_price = end_price - start_price
            amount_coin = portfolio_weights[c_idx] * self.hparams.init_balance / start_price
            coin_profit = del_price * amount_coin
            coin_profits.append(coin_profit)

        return sum(coin_profits)

    def _sample_from_individual(self, model, next_obs):
        loc, cov_mat_tril, _ = self._get_distribution_value(model, next_obs)
        # Get independent Normal policies for each cryptocurrency
        pi = MultivariateNormal(loc, scale_tril=cov_mat_tril)
        pi = TransformedDistribution(pi, [TanhTransform(cache_size=1), ])
        actions = pi.sample().cpu().numpy().squeeze()
        return actions

    def _get_distribution_value(self, model, obs):
        loc, scale, v = model(self.obs_to_torch(obs))
        cov_mat_tril = self._build_diag(scale)

        return loc, cov_mat_tril, v

    @staticmethod
    def _build_diag(scale):
        cov_mat_tril = torch.diag_embed(scale)
        return cov_mat_tril

    @staticmethod
    def _get_network_path(loop, n_val, checkpoint_type='best'):
        return os.path.join(train_results_folder, f'loop{loop}', f'{checkpoint_type}_network_val{n_val}.pth')

    @staticmethod
    def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
        return torch.tensor(obs, dtype=torch.float64, device=device)


def main():
    logging.info(f'Evaluating ensemble of models from training results at {train_results_folder}')
    all_weekly_ensemble_profits = []
    all_weekly_buy_hold_profits = []

    for loop in range(1, 49):
        set_random_seeds(args.seed)
        dataset = load_preprocessed_data(loop)
        hparams = get_hparams(args, loop)
        hparams.folder = os.path.join(args.result_folder, f'loop{loop}')
        M = ModelTest(hparams, loop, dataset)

        test_profits = M.get_test_results()
        all_weekly_ensemble_profits += test_profits

        buy_hold_profits = M.get_buy_hold_results()
        all_weekly_buy_hold_profits += buy_hold_profits

    with open(f'{args.result_folder}/weekly_ensemble_profits.pkl', 'wb') as f:
        pickle.dump(all_weekly_ensemble_profits, f)

    with open(f'{args.result_folder}/weekly_buy_hold_profits.pkl', 'wb') as f:
        pickle.dump(all_weekly_buy_hold_profits, f)

    logging.info(f'Average weekly PnL for ensemble policy: {np.mean(all_weekly_ensemble_profits):.6f}')
    logging.info(f'Average weekly PnL for buy and hold strategy: {np.mean(all_weekly_buy_hold_profits):.6f}')

    return


def get_argument():
    # Modify the hyperparameters file directory and name here
    hparams_folder = 'hparams_folder'
    hparams_fname = 'hparams.json'

    if not os.path.exists(hparams_folder):
        raise NotADirectoryError

    if not os.path.exists(train_results_folder):
        raise NotADirectoryError

    result_folder = os.path.join(train_results_folder, 'ensemble_test_results')

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--result_folder', type=str, default=result_folder)
    parser.add_argument('--hparams_folder', type=str, default=hparams_folder)
    parser.add_argument('--hparams_fname', type=str, default=hparams_fname)
    parser.add_argument('--n_models', type=int, default=9)
    args = parser.parse_args()

    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)

    return args

if __name__ == '__main__':
    args = get_argument()
    set_logger(f'{args.result_folder}/ensemble_logs.log')
    logging.info(f'Using random seed {args.seed} for evaluating ensemble policy.')
    main()