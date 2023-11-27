import os
import logging
import argparse
from utils import save_args, get_hparams, set_logger, set_random_seeds
from preprocess import load_preprocessed_data
from Agent import Agent


def main():
    # Track profits for all granular test periods
    test_profits = []

    # Repeat the training and testing procedure
    for loop in range(1, 49):
        # Load the data for this train/test loop
        dataset = load_preprocessed_data(loop)

        # Set up hyperparameters for this train/test loop
        hparams = get_hparams(args, loop)
        logging.info(f'Using hyperparams from {args.hparams_folder}/{args.hparams_fname}.')

        # Set random seeds specified in the hparams.json file
        set_random_seeds(hparams.seed)

        # Initialize the rl agent
        agent = Agent(loop, dataset)

        # Launch the training
        agent.set_hparams(hparams)
        profits = agent.run()
        test_profits += profits
        
        # Log the metrics to tensorboard
        agent.log_results()

    return

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_folder', type=str, default=f'training_results')
    parser.add_argument('--hparams_folder', type=str, default='hparams_folder')
    parser.add_argument('--hparams_fname', type=str, default='hparams.json')
    parser.add_argument('--comment', type=str, default='')

    args = parser.parse_args()

    if not os.path.isdir(args.result_folder):
        os.mkdir(args.result_folder)

    save_args(args)

    return args


if __name__ == "__main__":
    # Get arguments
    args = get_argument()
    # Set logger
    set_logger(os.path.join(args.result_folder, f'logs.log'))
    # Launch the training
    main()
