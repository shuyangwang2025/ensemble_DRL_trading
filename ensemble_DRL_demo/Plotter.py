import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import copy
import pandas as pd
style.use('seaborn-colorblind')


class Plotter:

    def __init__(self, hparams):
        self.coin_list = hparams.coin_list
        self.hparams = hparams

        return

    def plot_profit_history(self, loop, episode_idx, idx, balance_history, coin_value_history, portfolio_value, price_series_dict):
        fig, axes = plt.subplots(3, sharex=True, figsize=(12,12))

        ax = axes[0]
        ax.grid(True)
        ax.plot(balance_history[self.hparams.n_past-1:], label='balance')
        ax.set_title('Balance')
        ax.set_ylabel('USD')
        ax.legend()
        xval = ax.lines[0].get_xdata()

        ax = axes[1]
        ax.set_title(f'Portfolio value breakdown, end={portfolio_value:.2f}')
        ax.stackplot(xval, coin_value_history, labels=np.append(np.array(self.coin_list), 'USD'))
        ax.set_ylabel('USD')
        ax.legend()

        ax = axes[2]
        ax.set_title('Portfolio value history')
        ax.grid(True)
        profit_hist = np.sum(coin_value_history, axis=0)
        ax.plot(profit_hist, label='portfolio value')
        ax.set_ylabel('Portfolio value in USD')

        price_ratios = []
        for coin in self.coin_list:
            price_hist = price_series_dict[coin].iloc[self.hparams.n_past-1:].values
            p0 = price_hist[0]
            price_ratio = price_hist/p0
            price_ratios.append(copy.deepcopy(price_ratio))
        price_ratios = np.array(price_ratios)
        buy_hold_portfolio = 1/len(self.coin_list)*self.hparams.init_portfolio_value*np.sum(price_ratios, axis=0)
        ax.plot(buy_hold_portfolio, label='buy-hold')
        ax.legend(loc='upper left')

        fig.tight_layout()
        if not os.path.isdir(self.hparams.folder):
            os.mkdir(self.hparams.folder)

        plt.savefig(self.hparams.folder + f'/profit_history{idx}_{episode_idx}')
        plt.close()

        return

    def plot_trade_history(self, total_gain, loop, episode_idx, idx, holding_history, price_series_dict, coin_profit):
        # Plot only the trading history for each coin
        fig, axes = plt.subplots(len(self.coin_list), sharex=True, figsize=(12,3*len(self.coin_list)))
        fig.subplots_adjust(top=0.8)
        coin_hold_profits = []
        for coin_idx in range(len(self.coin_list)):
            ax = axes[coin_idx]
            ax.grid(True)
            coin = self.coin_list[coin_idx]
            twin_x = ax.twinx()

            holding_hist = holding_history[coin_idx, self.hparams.n_past-1:]
            ax.plot(holding_hist, label=coin + ' held')
            ax.set_ylabel(coin +' held')

            price_hist = price_series_dict[coin].iloc[self.hparams.n_past-1:]

            start_price = price_hist.iloc[0]
            end_price = price_hist.iloc[-1]
            change_price = end_price/start_price - 1
            coin_hold_profit = 1/len(self.coin_list) * self.hparams.init_portfolio_value * change_price
            coin_hold_profits.append(coin_hold_profit)

            ax.set_title(coin+' profit: {:.2f}, buy-hold profit: {:.2f}'.format(coin_profit[coin], coin_hold_profit), loc='right')

            price_hist.index = pd.DatetimeIndex(price_hist.index).strftime('%Y-%m-%d-%H')
            price_hist.plot(ax=twin_x, label=coin+' price', color='red', legend=True, xlabel='hours').legend(loc='upper right')
            twin_x.set_ylabel(coin+' price in USD')

            ax.legend(loc='upper left')
        buy_hold_profit = sum(coin_hold_profits)
        fig.suptitle(f'Episode={episode_idx}, \nprofit={total_gain:.2f}, buy-hold={buy_hold_profit:.2f}', y=0.99, fontsize=8)
        fig.tight_layout()

        if not os.path.isdir(self.hparams.folder):
            os.mkdir(self.hparams.folder)

        plt.savefig(self.hparams.folder + f'/trading_history{idx}_{episode_idx}')
        plt.close()

        return
