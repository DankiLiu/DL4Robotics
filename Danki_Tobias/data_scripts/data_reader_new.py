import os
import pandas as pd
import numpy as np

from Danki_Tobias.column_names import *


class DataReader():
    def __init__(self, data_type, train_on):
        assert train_on == 'non_crippled' or train_on == 'multiple_envs'
        self.data_type = data_type
        self.train_on = train_on

        self.state_columns, self.action_columns, self.label_columns = self.get_column_names()
        self.random_data_directory = f'../data/{self.data_type}/random_samples/{self.train_on}/'
        self.on_policy_data_directory = f'../data/{self.data_type}/on_policy/trained_on_{self.train_on}/'

    def get_column_names(self):
        if self.data_type == 'position' or self.data_type == 'position_deltas':
            return state_columns_position_only, action_columns, label_columns_position_only
        elif self.data_type == 'position_and_velocity' or self.data_type == 'position_and_velocity_deltas':
            return state_columns, action_columns, label_columns
        else:
            print("Data Type not valid")

    def get_random_data_directory(self, validation):
        directory = self.random_data_directory + 'training/'
        if validation:
            directory = self.random_data_directory + 'validation/'
        return directory

    def load_random_samples(self, validation=False):
        directory = self.get_random_data_directory(validation=validation)
        all_data = []
        for filename in os.listdir(directory):
            if not filename.startswith("normalization"):
                df = pd.read_csv(os.path.join(directory, filename), index_col=0, dtype='float64')
                all_data.append(df)

        df = pd.concat(all_data, axis=0, ignore_index=True)
        df = df.reset_index(drop=True)

        states = df[self.state_columns]
        actions = df[self.action_columns]
        labels = df[self.label_columns]
        return states, actions, labels

    def load_rl_samples(self, algorithm):
        df = pd.DataFrame([], columns=self.state_columns + self.action_columns + self.label_columns)

        file_name = f'{self.on_policy_data_directory}{algorithm}/samples.csv'
        if os.path.isfile(file_name):
            df = pd.read_csv(file_name, dtype='float64')

        states = df[self.state_columns]
        actions = df[self.action_columns]
        labels = df[self.label_columns]
        return [states, actions, labels]

    def load_normalization_variables(self, validation=False):
        directory = self.get_random_data_directory(validation=validation)
        return pd.read_csv(f'{directory}normalization_variables.csv', index_col=0)

    def save_normalization_variables(self, validation=False):
        def compute_normalization_variables(data):
            mean = data.mean()
            std = data.std()
            df = pd.concat([mean, std], axis=1)
            df.columns = ['mean', 'std']
            return df

        directory = self.get_random_data_directory(validation=validation)
        states_rand, actions_rand, labels_rand = self.load_random_samples(validation=False)
        states_normalized = compute_normalization_variables(states_rand)
        actions_normalized = compute_normalization_variables(actions_rand)
        labels_normalized = compute_normalization_variables(labels_rand)

        normalization_variables = pd.concat([states_normalized, actions_normalized, labels_normalized], axis=0)
        normalization_variables.to_csv(f'{directory}normalization_variables.csv')

    def store_in_file(self, states, actions, labels, algorithm):
        file_name = f'{self.on_policy_data_directory}{algorithm}/samples.csv'
        print(f"Storing data in: {file_name}")

        data = np.concatenate((states, actions, labels), axis=1)
        rollout_df = pd.DataFrame(data, columns=self.state_columns + self.action_columns + self.label_columns)

        if os.path.isfile(file_name):
            rollout_df.to_csv(file_name, mode='a', header=False, index=False)
        else:
            rollout_df.to_csv(file_name, index=False)


data_type_options = ['position', 'position_deltas', 'position_and_velocity', 'position_and_velocity_deltas']
train_on_possibilities = ['non_crippled', 'multiple_envs']

for dt in data_type_options:
    for tr in train_on_possibilities:
        dr = DataReader(data_type=dt, train_on=tr)
        dr.save_normalization_variables(validation=True)
        dr.save_normalization_variables(validation=False)


