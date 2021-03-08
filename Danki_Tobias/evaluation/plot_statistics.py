import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_type_options = ['position', 'position_deltas', 'position_and_velocity', 'position_and_velocity_deltas']
train_on_options = ['non_crippled', 'multiple_envs']
algorithms = ['normal', 'meta', 'online_adaptation']
names = ['training_0', 'training_1', 'training_2', 'training_3', 'training_4', 'training_5',
         'eval_0', 'eval_1', 'eval_2', 'eval_3', 'eval_4', 'eval_5']

cripple_options_training = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                                     [1, 1, 0, 1, 1, 1, 1, 1],
                                     [1, 1, 1, 1, 0, 1, 1, 1],
                                     [0.5, 1, 1, 1, 0, 0.3, 1, 1],
                                     [0.8, 0.9, 0.6, 0.8, 0.5, 1, 0.7, 1],
                                     [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1]])

cripple_options_evaluation = np.array([[1, 1, 1, 0, 1, 1, 1, 1],
                                       [1, 1, 1, 1, 1, 1, 0, 1],
                                       [1, 0.2, 1, 1, 1, 1, 1, 1],
                                       [1, 1, 1, 1, 0.4, 1, 1, 1],
                                       [1, 0.8, 0.1, 1, 0.5, 0.2, 1, 1],
                                       [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])


def read_average_reward(data_type, train_on, algorithm, name):
    file_name = f'../data/{data_type}/on_policy/trained_on_{train_on}/{algorithm}/evaluation_{name}.txt'
    with open(file_name, "r") as file:
        file.readline()
        file.readline()
        text = file.readline()
    _, average_reward = text.split("=")
    return float(average_reward)


def read_all_rewards():
    multi_index = pd.MultiIndex.from_product([data_type_options, train_on_options, algorithms, names],
                                             names=["data_type", "train_on", "algorithm", "name"])
    rewards = []
    for data_type in data_type_options:
        for train_on in train_on_options:
            for algorithm in algorithms:
                for name in names:
                    rewards.append(read_average_reward(data_type, train_on, algorithm, name))

    all_rewards = pd.Series(rewards, index=multi_index)


x = read_all_rewards()
print(x)