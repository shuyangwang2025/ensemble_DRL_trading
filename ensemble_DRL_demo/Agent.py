import os
import logging
import warnings
from datetime import datetime
import copy
import numpy as np
from collections import deque
import torch
from torch import optim
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from torch.distributions.transforms import TanhTransform
from torch.utils.tensorboard import SummaryWriter
from torch.nn import MSELoss
from utils import set_random_seeds
from GymEnv import TradingEnv
from Network import IndependentNormalModel
warnings.filterwarnings('ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Agent:
    def __init__(self, loop, dataset):
        self.model = None
        self.optimizer = None
        self.mse_loss = MSELoss()
        self.loop = loop
        self._upack_datasets(dataset)

    def _upack_datasets(self, dataset):
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

        self.n_weeks = len(test_data)
        self.n_models = len(val_data)
        self.coin_list = list(self.close_change_dict.keys())

        return

    def set_hparams(self, hparams, Env=TradingEnv):
        self.model = IndependentNormalModel(hparams).to(device)

        # optimizer
        if not hasattr(hparams, 'optim'):
            self.optimizer = optim.SGD(self.model.parameters(), lr=hparams.lr_init)
        elif hparams.optim == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=hparams.lr_init)
        elif hparams.optim == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=hparams.lr_init)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=hparams.lr_init)

        self.learning_rate = hparams.lr_init

        # Get the writer for this loop and this configuration
        run_timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        tb_path = os.path.join(hparams.folder, f'runs/{run_timestamp}')
        self.writer = SummaryWriter(os.path.join(tb_path, f'loop{self.loop}'))
        self.hparams = hparams
        self.n_past = hparams.n_past  # The look-back window size
        self.n_features = hparams.n_features

        self.gamma = hparams.gamma
        self.lam = hparams.lam

        # number of epochs to train the model with sampled data for each step
        self.epochs = 10

        # number of steps to run on each process for a single update
        self.sample_length = hparams.sample_length
        self.n_workers = hparams.n_workers
        # total number of samples for a single update
        self.mini_batch_size = self.n_workers * self.sample_length

        self.train_steps = 0  # Track the number of training updates in total so far
        self.episodes = 0  # Record the episodes trained for current loop
        # total episodes to train for each worker
        self.n_episodes = hparams.n_episodes

        # Data path for the best validation performance network and most recent network
        self.best_network_paths = [os.path.join(hparams.folder, f'best_network_val{n_val}.pth') for n_val in range(self.n_models)]
        self.latest_network_path = os.path.join(hparams.folder, 'latest_network.pth')
        self.init_network_path = os.path.join(hparams.folder, 'init_network.pth')

        # Initialize the first observation
        self.obs = np.zeros((self.n_workers, self.n_past, self.n_features), dtype=np.float64)

        # Compute hmax_dict
        hmax_dict = {}
        for c_idx in range(len(self.coin_list)):
            if hparams.norm_hmax:
                hmax_dict[self.coin_list[c_idx]] = hparams.init_balance / self.close_price_dict[self.coin_list[c_idx]].mean()
            else:
                hmax_dict[self.coin_list[c_idx]] = hparams.hmax
        self.hparams.hmax_dict = copy.deepcopy(hmax_dict)

        # Create game object for each worker for training for this current loop
        self.games = []
        for g in range(self.n_workers):
            game = Env(self.loop, g, self.concat_state_data, self.close_price_dict, self.close_change_dict, self.hparams)
            self.obs[g] = game.reset()
            self.games.append(game)

        # Create the games object for validation
        self.val_games = []
        for n_val in range(self.n_models):
            val_hparams = copy.deepcopy(hparams)
            val_hparams.folder += '/validate_plots'
            if not os.path.isdir(val_hparams.folder):
                os.mkdir(val_hparams.folder)
            val_hparams.folder += f'/val{n_val}'
            val_game = Env(self.loop, -1, self.concat_val_state_data_list[n_val],
                           self.val_close_price_dict_list[n_val],
                           self.val_close_change_dict_list[n_val], val_hparams)
            val_game.reset()
            self.val_games.append(val_game)

        # Create the game object for testing
        self.test_games = []
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
            self.test_games.append(test_game)

        return

    def run(self):
        logging.info(f"Train/validate/test loop {self.loop}:")
        logging.info(f"Use {self.n_workers} workers to roll out {self.n_episodes * self.n_workers} training episodes.")
        # episodes trained so far in a single loop
        self.episodes = self.games[0].episode_idx

        # Initialize the latest and best network checkpoint
        self._save_latest_checkpoint(self.model, self.latest_network_path)
        for n_val in range(self.n_models):
            self._save_checkpoint(self.model, n_val, -np.infty, self.best_network_paths)

        # Record the best validation performance for each validation period
        recent_val_profits = [deque(maxlen=5) for _ in range(self.n_models)]
        best_profits = [-np.infty for _ in range(self.n_models)]

        # Start this current loop of train-validate-test
        has_validated_in_this_episode = False
        while self.episodes < self.n_episodes:
            progress = self.episodes / self.n_episodes

            if self.episodes == 1 and not has_validated_in_this_episode:
                self._save_latest_checkpoint(self.model, self.init_network_path)

            # Validate current policy every 2 episodes
            if self.episodes % 2 == 0 and not has_validated_in_this_episode:
                has_validated_in_this_episode = True
                # Save the latest network in the current loop
                self._save_latest_checkpoint(self.model, self.latest_network_path)

                # Validation
                for n_val in range(self.n_models):
                    val_profit = self.validate_policy(n_val)
                    recent_val_profits[n_val].append(val_profit)
                    recent_mean = np.mean(recent_val_profits[n_val])
                    logging.info(f"Loop{self.loop}, progress: {self.n_workers * self.episodes} / {self.n_workers * self.n_episodes}, validation{n_val} profit: {val_profit:.2f}, smoothed profit: {recent_mean:.2f}")

                    # Save the best (smoothed) validation performance network during this loop
                    if self.episodes == 4 or (len(recent_val_profits[n_val]) >= 3 and recent_mean > best_profits[n_val]):
                        best_profits[n_val] = recent_mean
                        logging.info(f'Saving checkpoint at episode {self.episodes} at {self.best_network_paths[n_val]}')
                        # Save the model
                        self._save_checkpoint(self.model, n_val, val_profit, self.best_network_paths)

            # decreasing clip_range for PPO
            clip_range = 0.1 * (1 - 0.1 * progress)

            # Roll out episodes
            samples = self.sample()

            # Increment episode count
            if self.games[0].episode_idx > self.episodes:
                self.episodes = self.games[0].episode_idx
                has_validated_in_this_episode = False

            # Train
            self.train(samples, clip_range)
            self.train_steps += 1

        # Test at the end of training for this loop
        test_profits = self.test_policy()

        return test_profits

    def sample(self):
        # Sample data with current policy
        self.model.eval()
        rewards = np.zeros((self.n_workers, self.sample_length), dtype=np.float64)
        actions = np.zeros((self.n_workers, self.sample_length, len(self.coin_list)), dtype=np.float64)
        holdings = np.zeros((self.n_workers, self.sample_length, len(self.coin_list)), dtype=np.float64)
        done = np.zeros((self.n_workers, self.sample_length), dtype=bool)
        obs = np.zeros((self.n_workers, self.sample_length, self.n_past, self.n_features), dtype=np.float64)
        log_pis = np.zeros((self.n_workers, self.sample_length), dtype=np.float64)
        values = np.zeros((self.n_workers, self.sample_length), dtype=np.float64)
        behavior_log_pis = np.zeros((self.n_workers, self.sample_length), dtype=np.float64)

        # sample `worker_steps` from each worker
        with torch.no_grad():
            for g in range(self.n_workers):
                for t in range(self.sample_length):
                    obs[g, t] = self.obs[g]
                    loc, cov_mat_tril, v = self._get_distribution_value(self.model,self.obs[g])
                    pi = MultivariateNormal(loc, scale_tril=cov_mat_tril)
                    pi = TransformedDistribution(pi, [TanhTransform(cache_size=1), ])

                    # Record the value at this state
                    values[g, t] = v.cpu().numpy()

                    sample_act_val = pi.sample()
                    log_pis[g, t] = pi.log_prob(sample_act_val).cpu().numpy()
                    behavior_log_pis[g, t] = pi.log_prob(sample_act_val).cpu().numpy()
                    actions_arr = sample_act_val.cpu().numpy().squeeze()

                    # Take a step
                    self.obs[g], rewards[g, t], done[g, t], info = self.games[g].step(actions_arr)
                    actions[g, t, :] = actions_arr.reshape((-1,))
                    holdings[g, t, :] = info['holdings']

                    # At the end of one episode, i.e. done sampling the entire training period
                    if done[g, t]:
                        # Save the training metrics, using only the first worker
                        n_epi = self.games[g].episode_idx
                        if g == 0:
                            self.writer.add_scalar('train/profit', info['profit'], n_epi)

        # calculate advantages
        advantages = self._calc_advantages(done, rewards, values)
        samples = {
            'obs': self.obs_to_torch(obs),
            'actions': torch.tensor(actions, device=device),
            'values': torch.tensor(values, device=device),
            'log_pis': torch.tensor(log_pis, device=device),
            'behavior_log_pis': torch.tensor(behavior_log_pis, device=device),
            'advantages': torch.tensor(advantages, device=device),
            'holdings': torch.tensor(holdings, device=device)
        }

        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            if k == 'obs':
                samples_flat[k] = self.obs_to_torch(v)
            else:
                samples_flat[k] = torch.tensor(v, device=device)

        return samples_flat

    def train(self, samples, clip_range):
        self.model.train()

        total_loss = torch.tensor(0.0, device=device)
        for e in range(self.epochs):
            # shuffle for each epoch
            indexes = torch.randperm(self.mini_batch_size)
            mini_batch_indexes = indexes[:]
            mini_batch = {}
            for k, v in samples.items():
                mini_batch[k] = v[mini_batch_indexes]

            loss, entropy_bonus, policy_reward, vf_loss = self._calc_loss(samples=mini_batch, clip_range=clip_range)
            total_loss += loss

            for pg in self.optimizer.param_groups:
                pg['lr'] = self.learning_rate

            self.optimizer.zero_grad()
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        param.grad = torch.nan_to_num(param.grad, nan=0.0)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10000.0)

            # Update weights
            self.optimizer.step()

            progress = self.episodes / self.n_episodes

            self.learning_rate = self.hparams.lr_init * (1 - progress)
            current_lr = self.optimizer.param_groups[0]['lr']

        # Record the training loss in tensorboard
        self.writer.add_scalar('train_info/loss', total_loss.item()/self.epochs, self.train_steps)
        self.writer.add_scalar('train_info/learning_rate', current_lr, self.train_steps)

        return

    def validate_policy(self, n_val):
        # Record the validation at the current trained episodes in current loop
        val_game = self.val_games[n_val]
        val_game.episode_idx = self.episodes

        # Start running validation episodes
        self.model.eval()

        with torch.no_grad():
            done = False
            # Generate validation trading history plot
            next_obs = val_game.reset(plot=True)

            # Run one validation episode
            while not done:
                loc, cov_mat_tril, v = self._get_distribution_value(self.model, next_obs)
                pi = MultivariateNormal(loc, scale_tril=cov_mat_tril)
                pi = TransformedDistribution(pi, [TanhTransform(cache_size=1), ])
                actions = pi.sample().cpu().numpy().squeeze()
                next_obs, reward, done, info = val_game.step(actions)

        # done is True, record metrics for this one validation episode
        profit = self._write_to_tb(f'validate{n_val}', info)

        # Return the validation metrics
        return profit

    def test_policy(self):
        logging.info('Start testing for loop{}'.format(self.loop))
        # Load the model for each validation period
        models = []
        for n_val in range(self.n_models):
            val_model = copy.deepcopy(self.model)
            checkpoint = torch.load(self.best_network_paths[n_val], map_location=torch.device(device))
            val_model.load_state_dict(checkpoint['model_state_dict'])
            best_epi = checkpoint['n_epi']
            logging.info(f'Loading best validation {n_val} performance network from episode {best_epi}...')

            models.append(val_model)

        # Set each model to evaluation mode
        for model in models:
            model.eval()

        test_profits = []
        for week in range(self.n_weeks):
            set_random_seeds(self.hparams.seed)
            test_game = self.test_games[week]
            # Mark the test at the current trained episodes in current loop
            test_game.episode_idx = self.episodes

            # Run the test episode, sample from a Gaussian Mixture Model
            with torch.no_grad():
                done = False
                next_obs = test_game.reset(plot=True)
                while not done:
                    actions = self._sample_from_mixture(models, next_obs)
                    next_obs, reward, done, info = test_game.step(actions)

            # One episode is done
            test_profit = self._write_to_tb(f'test{week}', info)
            test_profits.append(test_profit)

        return test_profits

    def log_results(self):
        self.writer.add_hparams(
            {"loop": self.loop,},
            {"tuning/test profit": self.test_profit})
        self.writer.close()
        return

    def _get_distribution_value(self, model, obs):
        loc, scale, v = model(self.obs_to_torch(obs))
        cov_mat_tril = self._build_diag(scale)

        return loc, cov_mat_tril, v

    def _calc_loss(self, samples, clip_range):
        sampled_actions = samples['actions']

        loc, cov_mat_tril, value = self._get_distribution_value(self.model, samples['obs'])

        pi = MultivariateNormal(loc, scale_tril=cov_mat_tril)
        pi = TransformedDistribution(pi, [TanhTransform(cache_size=1), ])
        entropy_bonus = torch.log(torch.linalg.det(cov_mat_tril)) + len(self.coin_list) / 2 * torch.log(
            2 * torch.pi * torch.exp(torch.tensor(1.0, device=device)))
        entropy_bonus = entropy_bonus.mean()

        log_pi = pi.log_prob(sampled_actions)

        # Compute ratio and clipped ratio
        sample_log_pi = samples['log_pis']
        ratio = torch.exp(log_pi - sample_log_pi)
        clipped_ratio = ratio.clamp(min=1.0 - clip_range, max=1.0 + clip_range)

        # Compute importance weights
        behavior_log_pi = samples['behavior_log_pis']
        importance_weights = torch.exp(sample_log_pi - behavior_log_pi)

        # Compute normalized sample advantages
        sampled_normalized_advantage = self._normalize(samples['advantages'])
        # Get policy reward
        policy_reward = torch.min(importance_weights * ratio * sampled_normalized_advantage,
                                  importance_weights * clipped_ratio * sampled_normalized_advantage)
        policy_reward = policy_reward.nanmean()

        # Clip values and compute value function loss
        sampled_return = samples['values'] + samples['advantages']
        clipped_value = samples['values'] + (value - samples['values']).clamp(min=-clip_range, max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = 0.5 * vf_loss.nanmean()
        loss = -self.hparams.c1 * policy_reward - self.hparams.c2 * vf_loss - self.hparams.c3 * entropy_bonus

        # Return the loss to take a gradient step, the components for monitor purpose
        return loss, entropy_bonus, policy_reward, vf_loss

    def _calc_advantages(self, done: np.ndarray, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
        # advantages table
        advantages = np.zeros((self.n_workers, self.sample_length), dtype=np.float64)

        for g in range(self.n_workers):
            last_advantage = 0
            _, _, last_value = self._get_distribution_value(self.model, self.obs[g])
            last_value = last_value.cpu().data.numpy()

            for t in reversed(range(self.sample_length)):
                # mask if episode completed after step t
                mask = 1.0 - done[g, t]
                last_value = last_value * mask
                last_advantage = last_advantage * mask
                delta = rewards[g, t] + self.gamma * last_value - values[g, t]
                last_advantage = delta + self.gamma * self.lam * last_advantage

                advantages[g, t] = last_advantage

                last_value = values[g, t]

        return advantages

    def _sample_from_mixture(self, models, next_obs):
        locs = []
        covs = []
        for model in models:
            loc, cov_mat_tril, _ = self._get_distribution_value(model, next_obs)
            locs.append(loc.squeeze())
            covs.append(cov_mat_tril.squeeze())
        all_loc = torch.stack(locs, dim=0)
        all_cov = torch.stack(covs, dim=0)
        mix = Categorical(torch.ones(self.n_models, device=device))
        comp = MultivariateNormal(all_loc, scale_tril=all_cov)
        comp = TransformedDistribution(comp, [TanhTransform(cache_size=1), ])
        gmm = MixtureSameFamily(mix, comp)

        actions = gmm.sample().cpu().numpy().squeeze()
        return actions

    def _save_checkpoint(self, model, n_val, profit, paths):
        torch.save({
            'loop': self.loop,
            'n_epi': self.episodes,
            'n_val': n_val,
            'profit': profit,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, paths[n_val])

        return

    def _save_latest_checkpoint(self, model, path):
        torch.save({
            'loop': self.loop,
            'n_epi': self.episodes,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

        return

    def _write_to_tb(self, period, info):
        if 'test' in period:
            step_idx = self.loop
        else:
            step_idx = self.episodes

        # Record the test profit in tensorboard
        self.writer.add_scalar(f'{period}/profit', info['profit'], step_idx)

        if 'test' in period:
            logging.info(
                f'loop{self.loop}, {period} profit: {info["profit"]:.2f}')
            self.test_profit = info['profit']

        return info['profit']

    def _build_tril(self, scale, cov):
        cov_mat_tril = torch.diag_embed(scale)
        cov_idx = 0
        for row in range(len(self.coin_list)):
            for col in range(row):
                cov_mat_tril[:, row, col] = cov[:, cov_idx]
                cov_idx += 1
        return cov_mat_tril

    def _build_diag(self, scale):
        cov_mat_tril = torch.diag_embed(scale)

        return cov_mat_tril

    @staticmethod
    def _normalize(adv: torch.Tensor):
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    @staticmethod
    def smoothed_profits(scalars, weight):
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)  # Save it
            last = smoothed_val  # Anchor the last smoothed value

        return last

    @staticmethod
    def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
        return torch.tensor(obs, dtype=torch.float64, device=device)
