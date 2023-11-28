# An Ensemble Method of DRL for Automated Cryptocurrency Trading

The code repository for the manuscript "An Ensemble Method of Deep Reinforcement Learning for Automated Cryptocurrency Trading", including the implementation of the proposed ensemble method using a mixture distribution.

The structure of the code files is as follows.
```angular2html
├── README.md
├── requirements.txt
├── kraken-raw-data
│   ├── BCHUSD_60.csv
│   ├── ETHUSD_60.csv
│   ├── LTCUSD_60.csv
│   ├── XBTUSD_60.csv
│   └── XRPUSD_60.csv
├── processed-data
│   └── data_spec.json
├── ensemble_DRL_demo
│   ├──hparams_folder
│   │   └── hparams.json
│   ├── Agent.py
│   ├── GymEnv.py
│   ├── Network.py
│   ├── Plotter.py
│   ├── utils.py
│   ├── preprocess.py
│   ├── rl_agent.py
└── └── evaluate_ensemble.py
```
### 1. Setup Environment
The program uses Python 3.8.16. Install packages specified in `requirements.txt`.

### 2. Data Processing
`raw-kraken-data` contains 4.5-year of hourly OHLCV data for five cryptocurrencies from the Kraken Crypto Exchange. Run `preprocess.py` to process the raw OHLCV data from Kraken Crypto Exchange and add technical indicators according to the specification `processed-data/data_spec.json`. \
The data is divided into 48 rolling windows of train/validate/test data, each consists of 6 months of training data, multiple validation periods, and 1 month of granular testing periods.

### 3. Training
`rl_agent.py` launches the train/validate/test procedure for the 48 rolling windows. \
The program uses command line arguments to specify the hyperparameters file and results directory. By default, the hyperparameters are read from `hparams_folder/hparams.json`, and the results for all rolling windows are stored at `training_results`.
#### Modules used in training
`Agent.py` defines the DRL agent class for rolling out episodes and training the agent by PPO algorithm. \
`GymEnv.py` defines a customized Gym environment for cryptocurrency portfolio trading. \
`Network` defines the deep neural network architecture for PPO. \
`Plotter.py` defines the module for visualizing the changes of portfolio values. \
`utils.py` defines useful functions for logging progress and handling hyperparameters.

### 4. Evaluation
`evaluate_ensemble.py` evaluates the performance of the ensemble of the trained models on all granular test periods. The command line argument specifies the number of models to be used in the ensemble. The results and visualization of the trading behavior are stored at `training_results/ensemble_test_results`.
