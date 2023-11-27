import gym
from gym import spaces
import numpy as np
from collections import deque
import copy
from Plotter import Plotter


class TradingEnv(gym.Env):
    def __init__(self, loop, idx, data, close_price_dict, close_change_dict, hparams):
        super().__init__()
        self.hparams = hparams
        self.plotter = Plotter(hparams)
        # idx indicates if this is for train/validate/test, -1 for validation, -2 for test
        self.idx = idx
        self.loop = loop
        self.n_past = hparams.n_past
        self.n_features = hparams.n_features
        self.n_actions = hparams.n_actions
        self.threshold = 0.01

        self.data = data  # Contains the technical indicators
        self.price_series_dict = close_price_dict
        self.close_change_dict = close_change_dict
        self.coin_list = hparams.coin_list
        self.max_length = self.price_series_dict[self.coin_list[0]].shape[0]  # the length of the current time period

        self.episode_idx = 0

        self.coin_queue = {}
        self.coin_profit = {}
        self.coin_holding = {}
        for coin in self.coin_list:
            self.coin_queue[coin] = deque()
            self.coin_holding[coin] = self.hparams.init_coin[coin]
            self.coin_profit[coin] = 0.0
        self.current_step = self.n_past
        self.step_idx = 0
        self.reset_account()
        self.reset_account_history()
        self.rewards = []
        self.observation_space = spaces.Box(
            low=np.array([[-5.0] * data.shape[1] + [-1] * len(self.coin_list) + [-1]] * self.n_past),
            high=np.array([[5.0] * data.shape[1] + [1] * len(self.coin_list) + [1]] * self.n_past),
            shape=(self.n_past, self.n_features), dtype=np.float64)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.coin_list),), dtype=np.float64)

        return

    def reset_account(self):
        # Reset cash in USD
        self.balance = self.hparams.init_balance
        portfolio_value = self.balance

        # Reset holdings of coins
        self.coin_value_history = np.zeros((len(self.coin_list) + 1, 1))
        self.coin_value_history[-1, 0] = self.hparams.init_balance
        for c_idx, coin in enumerate(self.coin_list):
            # Clear account from previous episodes
            self.coin_queue[coin].clear()
            self.coin_profit[coin] = 0.0
            # Get initial coin holding, initial price to compute initial coin value in USD
            holding = self.hparams.init_coin[coin]
            self.coin_holding[coin] = holding
            p0 = self._price_at_step(coin, self.n_past-1)
            if self.hparams.init_coin[coin] > 0:
                self.coin_queue[coin].append((self.step_idx, p0, holding))
            coin_val = p0 * holding
            portfolio_value += coin_val
            self.coin_value_history[c_idx, 0] = coin_val

        # Reset portfolio value record
        self.portfolio_value = portfolio_value
        self.hparams.init_portfolio_value = portfolio_value
        self.portfolio_history = np.array([self.portfolio_value])

        return

    def reset_account_history(self):
        self.holding_history = np.zeros((len(self.coin_list), self.n_past))
        for c_idx, coin in enumerate(self.coin_list):
            self.holding_history[c_idx, :] = self.hparams.init_coin[coin]
        self.balance_history = np.zeros((1, self.n_past)) + self.hparams.init_balance  # Keep track of the balance

    def reset(self, plot=False):
        self.plot = plot

        # Reset the time step in a new episode
        self.current_step = self.n_past
        self.step_idx = 0

        # Reset the accounting of cash, coins, and value
        self.reset_account()
        self.reset_account_history()

        self.rewards = []  # track rewards throughout this episode
        self._set_max_min_norm()
        # Reset the initial max reward for each episode
        self.max_reward = -self.portfolio_value

        return self._next_obs()

    def step(self, actions: list):
        actions_taken = self._take_action(actions)
        done = False
        # Episode stopping criterion
        if self.current_step >= self.max_length - 1:
            done = True
            profit = self.portfolio_value - self.hparams.init_portfolio_value
            price_hist_dict = {}
            for coin in self.coin_list:
                price_hist = self.price_series_dict[coin].iloc[self.n_past-1:]
                price_hist_dict[coin] = copy.deepcopy(price_hist)

            episode_info = {"holdings": self.holding_history[:, -1],
                            "profit": profit}

            if self.idx == -1: # Validation
                if self.episode_idx % 10 == 0 and self.plot:
                    self.plotter.plot_trade_history(profit, self.loop, self.episode_idx, self.idx, self.holding_history, self.price_series_dict, self.coin_profit)
                    self.plotter.plot_profit_history(self.loop, self.episode_idx, self.idx, self.balance_history, self.coin_value_history, self.portfolio_value, self.price_series_dict)
                next_obs = self.reset(plot=False)
            elif self.idx == -2: # Test
                if self.plot:
                    self.plotter.plot_trade_history(profit, self.loop, self.episode_idx, self.idx, self.holding_history, self.price_series_dict, self.coin_profit)
                    self.plotter.plot_profit_history(self.loop, self.episode_idx, self.idx, self.balance_history, self.coin_value_history, self.portfolio_value, self.price_series_dict)
                next_obs = self.reset(plot=False)
            else: # For training agents
                self.episode_idx += 1
                next_obs = self.reset()
        else:
            self.current_step += 1
            self.step_idx += 1
            next_obs = self._next_obs()
            episode_info = {"holdings": self.holding_history[:, -1]}

        return next_obs, self._normalize_reward(self.reward), done, episode_info

    def _extend_holding_history(self):
        new_col = []
        for coin in self.coin_list:
            coin_h = self.coin_holding[coin]
            new_col.append(coin_h)
        self.holding_history = np.hstack((self.holding_history, np.array(new_col)[:,np.newaxis]))

    def _set_max_min_norm(self):
        self.max_holding = []
        self.min_holding = []
        for coin in self.coin_list:
            max_h = self.hparams.init_balance / self._current_price(coin)
            self.max_holding.append(max_h)
            self.min_holding.append(0.0)
        self.max_holding = np.array(self.max_holding)
        self.min_holding = np.array(self.min_holding)
        self.max_balance = 1.5 * self.hparams.init_balance
        self.min_balance = 1.0

    def _normalize_reward(self, reward):
        self.max_reward = np.maximum(abs(reward), self.max_reward)
        normalized_reward = reward / self.max_reward

        return normalized_reward

    def _min_max_zero_centering(self, arr, arr_min, arr_max): # To be in range (-1, 1)
        arr = ((arr - arr_min) / (arr_max - arr_min) - 0.5) * 2
        return arr

    def _next_obs(self):
        # obs shape is (n_past, n_features)
        start = self.current_step - self.n_past
        end = self.current_step

        obs = self.data.iloc[start:end, :].values
        holding_obs = self.holding_history[:, -self.n_past:].T
        balance_obs = self.balance_history[-self.n_past:].reshape((self.n_past, 1))

        # Normalization the state vector
        holding_obs = self._min_max_zero_centering(holding_obs, self.min_holding, self.max_holding)
        balance_obs = self._min_max_zero_centering(balance_obs, self.min_balance, self.max_balance)

        obs = np.concatenate((obs, holding_obs, balance_obs), axis=1)
        return obs

    def _next_price(self, coin):
        return self.price_series_dict[coin].iloc[self.current_step + 1]

    def _current_price(self, coin):
        return self.price_series_dict[coin].iloc[self.current_step]

    def _price_at_step(self, coin, step):
        return self.price_series_dict[coin].iloc[step]

    def _take_action(self, actions):
        for idx, action in enumerate(actions):
            coin = self.coin_list[idx]
            p = self._current_price(coin)
            available_coin_to_buy = np.floor(self.balance / p)
            available_coin_to_sell = self.coin_holding[coin]

            hmax = self.hparams.hmax_dict[coin]
            amount = abs(action * hmax)
            if action < 0 and amount > available_coin_to_sell: # When selling
                amount = available_coin_to_sell
                action = max(-1 * amount / hmax, -0.999)

            if action > 0 and amount > available_coin_to_buy: # When buying
                amount = available_coin_to_buy
                action = min(amount / hmax, 0.999)

            if action < -1 * self.threshold: # Selling
                left_to_sell = amount
                reward = 0.0
                while left_to_sell > 0.00001:
                    buy_idx, buy_price, buy_amount = self.coin_queue[coin].popleft()
                    if buy_amount <= left_to_sell:
                        # Sell all this portion
                        sell_amount = buy_amount
                        left_to_sell -= sell_amount
                    else:
                        # Sell only a fraction of the bought portion
                        sell_amount = left_to_sell
                        left_to_sell = 0
                        buy_amount_left = buy_amount - sell_amount
                        self.coin_queue[coin].appendleft((buy_idx, buy_price, buy_amount_left))

                    buy_cost = buy_price * sell_amount
                    sell_gain = p * sell_amount
                    reward += (sell_gain - buy_cost)
                    self.coin_holding[coin] -= sell_amount
                    self.balance += sell_gain

            elif action > 1 * self.threshold: # Buying
                self.coin_queue[coin].append((self.step_idx, p, amount))
                self.coin_holding[coin] += amount
                self.balance -= p * amount

        self._extend_holding_history()
        self.balance_history = np.append(self.balance_history, self.balance)
        prev_portfolio_value = self.portfolio_value
        self._compute_portfolio_value()
        current_portfolio_value = self.portfolio_value
        total_reward = current_portfolio_value - prev_portfolio_value
        self.reward = total_reward  # The rewards at this step
        self.rewards.append(total_reward)  # Track the rewards along this episode

        return

    def _compute_portfolio_value(self):
        portfolio_value = self.balance
        coin_values = []
        for coin in self.coin_list:
            holding = self.coin_holding[coin]
            price = self._current_price(coin)
            coinval = holding * price
            portfolio_value += coinval
            coin_values.append(coinval)
        coin_values.append(self.balance)

        self.coin_value_history = np.hstack((self.coin_value_history, np.array(coin_values)[:,np.newaxis]))
        self.portfolio_history = np.append(self.portfolio_history, portfolio_value)
        self.portfolio_value = portfolio_value

        return portfolio_value