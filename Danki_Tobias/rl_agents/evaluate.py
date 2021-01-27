import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os

from Danki_Tobias.data_scripts.data_reader import *
from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl
from Danki_Tobias.rl_agents.dynamicsModel import NNDynamicsModel
from Danki_Tobias.rl_agents.controller import MPCcontroller, sample

random_data_file = 'random_samples_2021-1-6_11-49'
model_id = 2


def load_model(env, model_checkpoint=99):
    normalization = load_normalization_variables(random_data_file)

    model = keras.models.load_model(filepath=f'../models/model_{model_id}/iteration_{model_checkpoint}.hdf5')
    dyn_model = NNDynamicsModel(env=env, normalization=normalization, model=model)
    return dyn_model


def calculate_errors():
    env = ReachEnvJointVelCtrl(render=False, nsubsteps=10, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))

    random_data_file = 'random_samples_2020-12-16_21-18'  # small datafile for testing purpose
    #random_data_file = 'random_samples_2021-1-6_11-49'  # big file
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


def visualize_paths(num_paths, path_length, model_checkpoint):
    # TODO: do we need to cripple both environments or keep the controller_env unchanged
    controller_env = ReachEnvJointVelCtrl(render=False, nsubsteps=10, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))
    env = ReachEnvJointVelCtrl(render=True, nsubsteps=10, crippled=np.array([0.2, 0.2, 1, 1, 1, 1, 1, 1]))

    dyn_model = load_model(env, model_checkpoint=model_checkpoint)

    # init the mpc controller
    mpc_controller = MPCcontroller(env=controller_env, dyn_model=dyn_model, horizon=1, num_simulated_paths=20)
    sample(env, mpc_controller, horizon=path_length, num_paths=num_paths, finish_when_done=True)


if __name__ == "__main__":
    visualize_paths(num_paths=3, path_length=1000, model_checkpoint=47)
    #calculate_errors()
