import json
import logging
import os
import torch
import numpy as np
import random
from datetime import datetime


def save_args(args):
    filename = os.path.join(args.result_folder, 'saved_arguments.txt')
    try:
        f = open(filename, 'w')
    except IOError:
        f = open(filename, 'x')
    now = datetime.utcnow()
    timestring = now.strftime("%m/%d/%Y %H:%M:%S")
    f.write(f"Experiment starts at UTC {timestring}\n")
    f.write('Arguments used are:\n')
    f.write('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    f.write('\n')
    f.close()
    return


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)


def set_random_seeds(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return


class Params():
    def __init__(self, json_path='params.json'):
        self.coin_list = None
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def get_hparams(args, loop):
    # Set up the result folder for the current loop
    loop_result_folder = os.path.join(args.result_folder, f'loop{loop}')
    if not os.path.isdir(loop_result_folder):
        os.mkdir(loop_result_folder)

    # Get data description
    data_spec_json = '../processed-data/data_spec.json'
    data_params = Params(data_spec_json)
    coin_list = data_params.coin_list
    tech_ind = data_params.technical_indicators

    # Read the hyperparameters from the json file
    hparams = Params(json_path=os.path.join(args.hparams_folder, args.hparams_fname))

    hparams.coin_list = coin_list
    # Features consist of technical indicators + OHLCV + current holding for each coin and cash balance
    hparams.n_features = (len(tech_ind) + 5) * len(coin_list) + len(coin_list) + 1
    hparams.folder = loop_result_folder

    # Create folders to store results of using this set of hyperparameters
    if not os.path.isdir(hparams.folder):
        os.mkdir(hparams.folder)

    json_filename = os.path.join(hparams.folder, 'saved_hparams.json')
    hparams.save(json_filename)

    return hparams