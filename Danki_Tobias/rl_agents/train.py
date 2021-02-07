import numpy as np
import pandas as pd
import tensorflow as tf
import os
import csv
from tensorflow import keras

from Danki_Tobias.data_scripts.data_reader import *
from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl
from Danki_Tobias.rl_agents.dynamicsModel import NNDynamicsModel
from Danki_Tobias.rl_agents.controller import MPCcontroller, sample
from Danki_Tobias.helper.get_parameters import *

random_data_file = 'random_samples_2021-1-6_11-49'
# random_data_file = 'random_samples_2020-12-16_21-18' # small datafile for testing purpose


# model_checkpoint = 27
# previous_checkpoint = 0
#
# model_id = 10  # is also the number of the rl_samples file
#
# # if new model = True a new model is created, else set previous_checkpoint to latest finished training iteration to continue training
# new_model = True
# previous_checkpoint = 0
#
# # training parameters
# iterations = 1
# number_of_random_samples = 100000
# training_epochs = 10
# new_paths_per_iteration = 1
# length_of_new_paths = 100
# learning_rate = 1e-3
# batch_size = 512

previous_checkpoint = 0

# Load dynamic model parameters from reach.json
dyn_n_layers, dyn_layer_size, dyn_batch_size, dyn_n_epochs, dyn_learning_rate = dyn_model_params()

# Load mpc_controller parameters from reach.json
num_simulated_paths, horizon, _ = MPCcontroller_params()

# Load parameters for collecting on-policy data
new_paths_per_iteration, length_of_new_paths = on_policy_sampling_params()

# Load parameters for training
# if new model = True a new model is created, else set model_checkpoint to latest finished training iteration to continue training
number_of_random_samples, iterations, training_epochs, new_model = dyn_model_training_params()
model_id = get_model_id()  # is also the number of the rl_samples file


def draw_training_samples():
    states_rand, actions_rand, state_deltas_rand = load_random_samples(random_data_file)

    states_rl, actions_rl, state_deltas_rl = load_rl_samples(collection=model_id)

    all_states = states_rl.append(states_rand)
    all_states = all_states.reset_index(drop=True)
    all_actions = actions_rl.append(actions_rand)
    all_actions = all_actions.reset_index(drop=True)
    all_deltas = state_deltas_rl.append(state_deltas_rand)
    all_deltas = all_deltas.reset_index(drop=True)

    states_sample = all_states.sample(n=number_of_random_samples, replace=False)
    actions_sample = all_actions.iloc[states_sample.index]
    delta_sample = all_deltas.iloc[states_sample.index]
    return states_sample, actions_sample, delta_sample


def save_rewards(rewards):
    average_reward = sum(rewards) / new_paths_per_iteration
    print(f'average_reward{average_reward}')
    with open(f"../data/reach_env/samples_{model_id}_rewards.csv", "a+") as file:
        wr = csv.writer(file)
        wr.writerow(rewards)


# TODO: replace constant values to variables declared in header
if __name__ == "__main__":
    controller_env = ReachEnvJointVelCtrl(render=False, nsubsteps=10, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))
    env = ReachEnvJointVelCtrl(render=False, nsubsteps=10, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))

    normalization = load_normalization_variables(random_data_file)

    if new_model:
        dyn_model = NNDynamicsModel.new_model(env=env,
                                              n_layers=dyn_n_layers,
                                              size=dyn_layer_size,
                                              activation=tf.tanh,
                                              output_activation=None,
                                              normalization=normalization,
                                              batch_size=dyn_batch_size,
                                              learning_rate=dyn_learning_rate)
    else:
        model = keras.models.load_model(
            filepath=f'../models/model_{model_id}/iteration_{previous_checkpoint}.hdf5')
        dyn_model = NNDynamicsModel(env=env, normalization=normalization, model=model)

    # init the mpc controller
    mpc_controller = MPCcontroller(env=controller_env, dyn_model=dyn_model, horizon=horizon, num_simulated_paths=num_simulated_paths)


    # sample new training examples
    # retrain the model
    for iteration in range(iterations):
        print(f'iteration: {iteration}')
        dyn_model.fit(*draw_training_samples(), N_EPOCHS=training_epochs)

        # Generate new trajectories with the MPC controllers
        paths, rewards, costs = sample(env, mpc_controller, horizon=length_of_new_paths,
                                       num_paths=new_paths_per_iteration)

        save_rewards(rewards)

        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        next_observations = np.concatenate([path["next_observations"] for path in paths])
        observation_delta = next_observations - observations

        store_in_file(observations, actions, observation_delta, collection=model_id)
        dyn_model.model.save(
            filepath=f'../models/model_{model_id}/iteration_{iteration + previous_checkpoint + 1}.hdf5')
        print('Model saved')
