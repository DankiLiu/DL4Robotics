import os
import pandas as pd
import numpy as np

from Danki_Tobias.column_names import *

folder_path = '../data/reach_env/'


def load_random_samples(filename):
    df = pd.read_csv(f'{folder_path}{filename}.csv', index_col=0)
    # df = df.reset_index(drop=True)
    states = df[state_columns]
    actions = df[action_columns]
    state_deltas = df[delta_columns]
    return states, actions, state_deltas


def load_rl_samples(collection):
    df = pd.read_csv(f'{folder_path}rl_samples_{collection}.csv', dtype='float64')
    # df = df.reset_index(drop=True)
    states = df[state_columns]
    actions = df[action_columns]
    state_deltas = df[delta_columns]
    return states, actions, state_deltas


def load_normalization_variables(filename):
    return pd.read_csv(f'{folder_path}normalization_variables/{filename}.csv', index_col=0)


def save_normalization_variables(filename='random_samples_2020-12-16_21-18'):
    def compute_normalization_variables(data):
        mean = data.mean()
        std = data.std()
        df = pd.concat([mean, std], axis=1)
        df.columns = ['mean', 'std']
        return df

    states_rand, actions_rand, state_deltas_rand = load_random_samples(filename=filename)

    states_normalized = compute_normalization_variables(states_rand)
    actions_normalized = compute_normalization_variables(actions_rand)
    deltas_normalized = compute_normalization_variables(state_deltas_rand)

    normalization_variables = pd.concat([states_normalized, actions_normalized, deltas_normalized], axis=0)
    normalization_variables.to_csv(f'{folder_path}normalization_variables/{filename}.csv')


def store_in_file(observations, actions, deltas, collection):
    file_name = f'{folder_path}rl_samples_{collection}.csv'
    print(f"Storing data in: {file_name}")

    data = np.concatenate((observations, actions, deltas), axis=1)
    rollout_df = pd.DataFrame(data, columns=state_columns + action_columns + delta_columns)

    if os.path.isfile(file_name):
        rollout_df.to_csv(file_name, mode='a', header=False, index=False)
    else:
        rollout_df.to_csv(file_name, index=False)
