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

random_data_file = 'random_samples_2021-1-6_11-49'
# random_data_file = 'random_samples_2020-12-16_21-18' # small datafile for testing purpose

iterations = 100
training_epochs = 50

new_paths_per_iteration = 10
length_of_new_paths = 500

rl_data_collection = 1

# if new model = True a new model is created, else set model_checkpoint to latest finished training iteration to continue training
new_model = False
model_checkpoint = 27


def draw_training_samples(number_of_samples=100000):
    states_rand, actions_rand, state_deltas_rand = load_random_samples(random_data_file)
    states_rl, actions_rl, state_deltas_rl = load_rl_samples(collection=rl_data_collection)

    all_states = states_rl.append(states_rand)
    all_states = all_states.reset_index(drop=True)
    all_actions = actions_rl.append(actions_rand)
    all_actions = all_actions.reset_index(drop=True)
    all_deltas = state_deltas_rl.append(state_deltas_rand)
    all_deltas = all_deltas.reset_index(drop=True)

    states_sample = all_states.sample(n=number_of_samples, replace=False)
    actions_sample = all_actions.iloc[states_sample.index]
    delta_sample = all_deltas.iloc[states_sample.index]
    return states_sample, actions_sample, delta_sample


def save_rewards(rewards):
    average_reward = sum(rewards) / new_paths_per_iteration
    print(f'average_reward{average_reward}')
    with open(f"../data/reach_env/samples_{rl_data_collection}_rewards.csv", "a+") as file:
        wr = csv.writer(file)
        wr.writerow(rewards)


# TODO: replace constant values to variables declared in header
if __name__ == "__main__":
    controller_env = ReachEnvJointVelCtrl(render=False, nsubsteps=10, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))
    env = ReachEnvJointVelCtrl(render=False, nsubsteps=10, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))

    normalization = load_normalization_variables(random_data_file)

    if new_model:
        dyn_model = NNDynamicsModel.new_model(env=env,
                                              n_layers=2,
                                              size=64,
                                              activation=tf.tanh,
                                              output_activation=None,
                                              normalization=normalization,
                                              batch_size=512,
                                              learning_rate=1e-3)
    else:
        model = keras.models.load_model(filepath=f'../models/iteration_{model_checkpoint}.hdf5')
        dyn_model = NNDynamicsModel(env=env, normalization=normalization, model=model)

    # init the mpc controller
    mpc_controller = MPCcontroller(env=controller_env, dyn_model=dyn_model, horizon=1, num_simulated_paths=100)

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

        store_in_file(observations, actions, observation_delta, collection=rl_data_collection)
        dyn_model.model.save(filepath=f'../models/iteration_{iteration}.hdf5')
        print('Model saved')
