import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os

from Danki_Tobias.data_scripts.data_reader import *
from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl, ReachEnvJointTorqueCtrl
from Danki_Tobias.rl_agents.dynamicsModel import *
from Danki_Tobias.rl_agents.metaRLDynamicsModel import *
from Danki_Tobias.rl_agents.controller import MPCcontroller, sample

data_type_options = ['position', 'position_deltas', 'position_and_velocity', 'position_and_velocity_deltas']
train_on_options = ['non_crippled', 'multiple_envs']

data_type = data_type_options[0]
train_on = train_on_options[0]
algorithm = 'meta'  # 'normal' # 'online_adaptation'

meta = algorithm == 'meta'
with_adaptation = algorithm == 'online_adaptation'
states_only = (data_type == 'position' or data_type == 'position_deltas')

if data_type == 'position' or data_type == 'position_and_velocity':
    predicts_state = True
    if meta:
        dynamicsModel = MetaRLDynamicsModel
    else:
        dynamicsModel = NNDynamicsModel
elif data_type == 'position_deltas' or data_type == 'position_and_velocity_deltas':
    predicts_state = False
    if meta:
        dynamicsModel = MetaRLDynamicsModelDeltaPrediction
    else:
        dynamicsModel = NNDynamicsModelDeltaPrediction
else:
    print("Data Type not valid")

data_reader = DataReader(data_type=data_type, train_on=train_on)
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


def load_model(env, model_checkpoint=50):
    normalization = data_reader.load_normalization_variables()

    model_path = f'../models/{data_type}/trained_on_{train_on}/{algorithm}/iteration_{model_checkpoint}.hdf5'
    if with_adaptation:
        model_path = f'../models/{data_type}/trained_on_{train_on}/normal/iteration_{model_checkpoint}.hdf5'

    model = keras.models.load_model(filepath=model_path)
    dyn_model = dynamicsModel(env=env, normalization=normalization, model=model, states_only=states_only)
    return dyn_model


def calculate_errors(validation=True):
    env = ReachEnvJointVelCtrl(render=False, nsubsteps=10, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))
    states_rand, actions_rand, labels_rand = data_reader.load_random_samples(validation=validation)

    for i in range(0, 50):
        dyn_model = load_model(env, model_checkpoint=i)
        predictions = dyn_model.predict(states_rand.copy(), actions_rand.copy())

        # calculate difference of predictions and actual values of the next state
        difference = predictions - labels_rand
        difference = difference.abs()

        mean_absolute_error = difference.mean().mean()

        # calculate mean squared error
        difference_squared = difference.pow(2)
        mean_squared_error = difference_squared.mean().mean()

        print(f'Metrics with model after {i} training Epochs')
        print(f'Mean absolute Error: {mean_absolute_error}')
        print(f'Mean squared Error: {mean_squared_error}')


def visualize_paths(num_paths, path_length, model_checkpoint=50, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1])):
    controller_env = ReachEnvJointVelCtrl(render=False, )
    env = ReachEnvJointVelCtrl(render=True, nsubsteps=10, crippled=crippled)
    dyn_model = load_model(env, model_checkpoint=model_checkpoint)

    # init the mpc controller
    mpc_controller = MPCcontroller(env=controller_env, dyn_model=dyn_model, horizon=1, num_simulated_paths=100,
                                   states_only=states_only)
    sample(env, mpc_controller, horizon=path_length, num_paths=num_paths, finish_when_done=True,
           with_adaptation=(meta or with_adaptation), predicts_state=predicts_state, states_only=states_only)


def average_reward(num_paths, path_length, name, model_checkpoint=50, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1])):
    controller_env = ReachEnvJointVelCtrl(render=False, )
    env = ReachEnvJointVelCtrl(render=False, nsubsteps=10, crippled=crippled)
    dyn_model = load_model(env, model_checkpoint=model_checkpoint)
    # init the mpc controller
    # init the mpc controller
    mpc_controller = MPCcontroller(env=controller_env, dyn_model=dyn_model, horizon=1, num_simulated_paths=100,
                                   states_only=states_only)

    paths, rewards, costs = sample(env, mpc_controller, horizon=path_length, num_paths=num_paths, finish_when_done=True,
                                   with_adaptation=(meta or with_adaptation), predicts_state=predicts_state,
                                   states_only=states_only)
    average_reward = sum(rewards) / num_paths

    file_name = f'../data/{data_type}/on_policy/trained_on_{train_on}/{algorithm}/evaluation_{name}.txt'
    with open(file_name, "a+") as file:
        file.write(f"Rewards with crippled = {crippled}\n")
        file.write(f"{rewards}\n")
        file.write(f"Average Reward = {average_reward}\n")

    return average_reward


def average_reward_training_envs(num_paths):
    for i, c in enumerate(cripple_options_training):
        average_reward(num_paths=num_paths, path_length=500, model_checkpoint=50, crippled=c,
                       name=f"training_{i}")


def average_reward_test_envs(num_paths):
    for i, c in enumerate(cripple_options_evaluation):
        if i < 3:
            continue
        average_reward(num_paths=num_paths, path_length=500, model_checkpoint=50, crippled=c,
                       name=f"eval_{i}")


if __name__ == "__main__":
    # visualize_paths(num_paths=3, path_length=1000, model_checkpoint=50)
    # calculate_errors()
    # average_reward(num_paths=100, path_length=1000, model_checkpoint=50, meta=meta)
    # for e in ['exp1', 'exp2']:
    #    experiment = e

    #average_reward_training_envs(20)
    average_reward_test_envs(20)

