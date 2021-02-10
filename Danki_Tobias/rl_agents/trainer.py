import numpy as np
import pandas as pd
import tensorflow as tf
import os
import csv
from tensorflow import keras

from Danki_Tobias.data_scripts.data_reader import *
from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl, ReachEnvJointTorqueCtrl
from Danki_Tobias.rl_agents.dynamicsModel import NNDynamicsModel
from Danki_Tobias.rl_agents.metaRLDynamicsModel import MetaRLDynamicsModel
from Danki_Tobias.rl_agents.controller import MPCcontroller, sample
from Danki_Tobias.helper.get_parameters import *

random_data_file = 'random_samples_2021-1-6_11-49'
# random_data_file = 'random_samples_2020-12-16_21-18' # small datafile for testing purpose

model_id = 2  # is also the number of the rl_samples file
# if new model = True a new model is created,
# else set previous_checkpoint to latest finished training iteration to continue training
new_model = True
previous_checkpoint = 0

dynamicsModel = NNDynamicsModel
meta = False
dynamicsModel = MetaRLDynamicsModel
meta = True

# Load dynamic model parameters from reach.json
dyn_n_layers, dyn_layer_size, dyn_batch_size, dyn_learning_rate, M, K = model_params(meta, model_id)
# Load mpc_controller parameters from reach.json
num_simulated_paths, horizon = MPCcontroller_params(meta, model_id)
# Load parameters for collecting on-policy data
new_paths_per_iteration, length_of_new_paths = on_policy_sampling_params(meta, model_id)
# Load parameters for training
number_of_random_samples, iterations, training_epochs = training_params(meta, model_id)

trajectory_length = M + K


def draw_training_samples():
    """
    draws random trajectories of length M+K
    The first M steps of the trajectory are used for the task specific update step
    the following K steps are used for the Loss calculation for the Meta Update
    In General M > K. Standard values M=32 and K=16 are taken from appendix of the paper
    """
    # TODO: load data of multiple sources with different crippled joints
    states_rand, actions_rand, state_deltas_rand = load_random_samples(random_data_file)
    states_rl, actions_rl, state_deltas_rl = load_rl_samples(collection=model_id, meta=meta)

    all_states = states_rl.append(states_rand)
    all_states = all_states.reset_index(drop=True)
    all_actions = actions_rl.append(actions_rand)
    all_actions = all_actions.reset_index(drop=True)
    all_deltas = state_deltas_rl.append(state_deltas_rand)
    all_deltas = all_deltas.reset_index(drop=True)

    if meta:
        assert (number_of_random_samples / trajectory_length).is_integer()
        num_trajectories = int(number_of_random_samples / trajectory_length)
        random = np.random.randint(len(all_states) - trajectory_length, size=num_trajectories)
        func = lambda v: np.arange(start=v, stop=v + trajectory_length)
        random = (np.array([func(v) for v in random])).flatten()
    else:
        random = np.random.randint(len(all_states), size=number_of_random_samples)

    states_sample = all_states.iloc[random]
    actions_sample = all_actions.iloc[random]
    delta_sample = all_deltas.iloc[random]
    # TODO: Check why index are not same (It should work without reset_index)
    return states_sample.reset_index(drop=True), actions_sample.reset_index(drop=True), delta_sample.reset_index(
        drop=True)


def save_rewards(rewards):
    average_reward = sum(rewards) / new_paths_per_iteration
    print(f'average_reward: {average_reward}')
    file_name = f"../data/reach_env/samples_{model_id}_rewards.csv"
    if meta:
        file_name = f"../data/reach_env/samples_{model_id}_rewards_meta.csv"
    with open(file_name, "a+") as file:
        wr = csv.writer(file)
        wr.writerow(rewards)


# TODO: replace constant values to variables declared in header
if __name__ == "__main__":
    controller_env = ReachEnvJointVelCtrl(render=False, nsubsteps=10, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))
    env = ReachEnvJointVelCtrl(render=False, nsubsteps=10, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))

    normalization = load_normalization_variables(random_data_file)

    if new_model:
        dyn_model = dynamicsModel.new_model(env=env,
                                            n_layers=dyn_n_layers,
                                            size=dyn_layer_size,
                                            activation=tf.tanh,
                                            output_activation=None,
                                            normalization=normalization,
                                            batch_size=dyn_batch_size,
                                            learning_rate=dyn_learning_rate)
    else:
        file_path = f'../models/model_{model_id}/iteration_{previous_checkpoint}.hdf5'
        if meta:
            file_path = f'../meta_models/model_{model_id}/iteration_{previous_checkpoint}.hdf5'
        model = keras.models.load_model(filepath=file_path)
        dyn_model = dynamicsModel(env=env, normalization=normalization, model=model)

    # init the mpc controller
    mpc_controller = MPCcontroller(env=controller_env, dyn_model=dyn_model, horizon=horizon,
                                   num_simulated_paths=num_simulated_paths)

    # sample new training examples
    # retrain the model
    for iteration in range(iterations):
        print(f'iteration: {iteration}')
        dyn_model.fit(*draw_training_samples(), N_EPOCHS=training_epochs)

        # Generate new trajectories with the MPC controllers
        paths, rewards, costs = sample(env, mpc_controller, horizon=length_of_new_paths,
                                       num_paths=new_paths_per_iteration, finish_when_done=True, with_adaptaion=True)

        save_rewards(rewards)

        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        next_observations = np.concatenate([path["next_observations"] for path in paths])
        observation_delta = next_observations - observations

        store_in_file(observations, actions, observation_delta, collection=model_id, meta=meta)
        file_path = f'../models/model_{model_id}/iteration_{iteration + previous_checkpoint + 1}.hdf5'
        if meta:
            file_path = f'../meta_models/model_{model_id}/iteration_{iteration + previous_checkpoint + 1}.hdf5'
        dyn_model.model.save(filepath=file_path)
        print('Model saved')
