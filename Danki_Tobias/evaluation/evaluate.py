import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os

from Danki_Tobias.data_scripts.data_reader import *
from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl, ReachEnvJointTorqueCtrl
from Danki_Tobias.rl_agents.dynamicsModel import NNDynamicsModel
from Danki_Tobias.rl_agents.metaRLDynamicsModel import MetaRLDynamicsModel
from Danki_Tobias.rl_agents.controller import MPCcontroller, sample

experiment = 'exp2'
model_id = 0

"""
dynamicsModel = NNDynamicsModel
meta = False
model_type = 'normal'
"""
dynamicsModel = MetaRLDynamicsModel
meta = True
model_type = 'meta'

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


def load_model(env, model_checkpoint=99, meta=False):
    normalization = load_normalization_variables(experiment=experiment)

    model_path = f'../models/{experiment}/{model_type}/model_{model_id}/iteration_{model_checkpoint}.hdf5'
    model = keras.models.load_model(filepath=model_path)
    if meta:
        dyn_model = MetaRLDynamicsModel(env=env, normalization=normalization, model=model)
    else:
        dyn_model = NNDynamicsModel(env=env, normalization=normalization, model=model)
    return dyn_model


def calculate_errors():  # TODO: adapt to new data reader functions
    env = ReachEnvJointVelCtrl(render=False, nsubsteps=10, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))

    random_data_file = 'random_samples_2020-12-16_21-18'  # small datafile for testing purpose
    # random_data_file = 'random_samples_2021-1-6_11-49'  # big file
    states_rand, actions_rand, state_deltas_rand = load_random_samples(random_data_file)

    state_deltas_rand.columns = state_columns
    next_states = states_rand + state_deltas_rand

    for i in range(0, 47):
        dyn_model = load_model(env, model_checkpoint=i)
        predictions = dyn_model.predict(states_rand.copy(), actions_rand.copy())

        # calculate difference of predictions and actual values of the next state
        difference = predictions - next_states
        difference = difference.abs()

        mean_absolute_error = difference.mean().mean()

        # calculate mean squared error
        difference_squared = difference.pow(2)
        mean_squared_error = difference_squared.mean().mean()

        print(f'Metrics with model after {i} training Epochs')
        print(f'Mean absolute Error: {mean_absolute_error}')
        print(f'Mean squared Error: {mean_squared_error}')


def visualize_paths(num_paths, path_length, model_checkpoint, meta):
    controller_env = ReachEnvJointVelCtrl(render=False, )
    env = ReachEnvJointVelCtrl(render=True, nsubsteps=10, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))
    dyn_model = load_model(env, model_checkpoint=model_checkpoint, meta=meta)

    # init the mpc controller
    mpc_controller = MPCcontroller(env=controller_env, dyn_model=dyn_model, horizon=1, num_simulated_paths=20)
    sample(env, mpc_controller, horizon=path_length, num_paths=num_paths, finish_when_done=True, with_adaptaion=meta)


def average_reward(num_paths, path_length, model_checkpoint, meta, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]),
                   name=""):
    controller_env = ReachEnvJointVelCtrl(render=False, )
    env = ReachEnvJointVelCtrl(render=False, nsubsteps=10, crippled=crippled)
    dyn_model = load_model(env, model_checkpoint=model_checkpoint, meta=meta)
    # init the mpc controller
    mpc_controller = MPCcontroller(env=controller_env, dyn_model=dyn_model, horizon=1, num_simulated_paths=20)

    paths, rewards, costs = sample(env, mpc_controller, horizon=path_length, num_paths=num_paths, finish_when_done=True,
                                   with_adaptaion=meta)
    average_reward = sum(rewards) / num_paths

    file_name = f'../data/on_policy/{experiment}/{model_type}/model{model_id}/evaluation_{name}.txt'
    # file_name = f'../data/on_policy/{experiment}/online_adaptation/model{model_id}/evaluation_{name}.txt'
    with open(file_name, "a+") as file:
        file.write(f"Rewards with crippled = {crippled}\n")
        file.write(f"{rewards}\n")
        file.write(f"Average Reward = {average_reward}\n")

    return average_reward


def average_reward_training_envs():
    for i, c in enumerate(cripple_options_training):
        average_reward(num_paths=100, path_length=1000, model_checkpoint=50, meta=meta, crippled=c,
                       name=f"training_{i}")


def average_reward_test_envs():
    for i, c in enumerate(cripple_options_evaluation):
        average_reward(num_paths=100, path_length=1000, model_checkpoint=50, meta=meta, crippled=c, name=f"eval_{i}")


if __name__ == "__main__":
    visualize_paths(num_paths=3, path_length=1000, model_checkpoint=1, meta=meta)
    # calculate_errors()
    # average_reward(num_paths=100, path_length=1000, model_checkpoint=50, meta=meta)
    # for e in ['exp1', 'exp2']:
    #    experiment = e
    #    average_reward_training_envs()
