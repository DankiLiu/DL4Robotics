import os
import pandas as pd
import numpy as np

from Danki_Tobias.helper.column_names import *


def load_random_samples(experiment='exp1', validation=False):
    directory = f'../data/random_samples/{experiment}/training/'
    if validation:
        directory = f'../data/random_samples/{experiment}/validation/'

    all_data = []
    for filename in os.listdir(directory):
        if not filename.startswith("normalization"):
            df = pd.read_csv(os.path.join(directory, filename), index_col=0, dtype='float64')
            all_data.append(df)

    df = pd.concat(all_data, axis=0, ignore_index=True)
    df = df.reset_index(drop=True)

    if experiment == 'exp3':
        states = df[state_columns_exp3]
        actions = df[action_columns]
        next_states = df[next_state_columns]
        return states, actions, next_states

    states = df[state_columns]
    actions = df[action_columns]
    state_deltas = df[delta_columns]
    return states, actions, state_deltas


def load_rl_samples(model_id, experiment, model_type='normal'):
    assert model_type == 'normal' or model_type == 'meta' or model_type == 'online_adaptation'
    file_name = f'../data/on_policy/{experiment}/{model_type}/model{model_id}/samples.csv'

    df = pd.DataFrame([], columns=state_columns + action_columns + delta_columns)
    if experiment == 'exp3':
        df = pd.DataFrame([], columns=state_columns_exp3 + action_columns + next_state_columns)

    if os.path.isfile(file_name):
        df = pd.read_csv(file_name, dtype='float64')

    if experiment == 'exp3':
        states = df[state_columns_exp3]
        actions = df[action_columns]
        next_states = df[next_state_columns]
        return states, actions, next_states

    states = df[state_columns]
    actions = df[action_columns]
    state_deltas = df[delta_columns]
    return [states, actions, state_deltas]


def load_normalization_variables(experiment='exp1', validation=False):
    directory = f'../data/random_samples/{experiment}/training/'
    if validation:
        directory = f'../data/random_samples/{experiment}/validation/'
    return pd.read_csv(f'{directory}normalization_variables.csv', index_col=0)


def save_normalization_variables(experiment='exp1', validation=False):
    directory = f'../data/random_samples/{experiment}/training/'
    if validation:
        directory = f'../data/random_samples/{experiment}/validation/'

    def compute_normalization_variables(data):
        mean = data.mean()
        std = data.std()
        df = pd.concat([mean, std], axis=1)
        df.columns = ['mean', 'std']
        return df

    states_rand, actions_rand, state_deltas_rand = load_random_samples(experiment=experiment)

    states_normalized = compute_normalization_variables(states_rand)
    actions_normalized = compute_normalization_variables(actions_rand)
    deltas_normalized = compute_normalization_variables(state_deltas_rand)

    normalization_variables = pd.concat([states_normalized, actions_normalized, deltas_normalized], axis=0)
    normalization_variables.to_csv(f'{directory}normalization_variables.csv')


save_normalization_variables(experiment='exp3')


def store_in_file(observations, actions, deltas, experiment, model_id, model_type='normal'):
    assert model_type == 'normal' or model_type == 'meta' or model_type == 'online_adaptation'
    file_name = f'../data/on_policy/{experiment}/{model_type}/model{model_id}/samples.csv'
    print(f"Storing data in: {file_name}")

    data = np.concatenate((observations, actions, deltas), axis=1)
    if experiment == 'exp3':
        rollout_df = pd.DataFrame(data, columns=state_columns_exp3 + action_columns + next_state_columns)
    else:
        rollout_df = pd.DataFrame(data, columns=state_columns + action_columns + delta_columns)

    if os.path.isfile(file_name):
        rollout_df.to_csv(file_name, mode='a', header=False, index=False)
    else:
        rollout_df.to_csv(file_name, index=False)
