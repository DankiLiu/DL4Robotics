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

experiment = 'exp3'
if experiment == 'exp3':
    from Danki_Tobias.rl_agents.dynamicsModelState import NNDynamicsModel
    print("TEST")

cripple_options = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 0, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 0, 1, 1, 1],
                            [0.5, 1, 1, 1, 0, 0.3, 1, 1],
                            [0.8, 0.9, 0.6, 0.8, 0.5, 1, 0.7, 1],
                            [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1]])

model_id = 0  # is also the number of the rl_samples file
# if new model = True a new model is created,
# else set previous_checkpoint to latest finished training iteration to continue training
new_model = True
previous_checkpoint = 0

dynamicsModel = NNDynamicsModel
meta = False
model_type = 'normal'
"""
dynamicsModel = MetaRLDynamicsModel
meta = True
model_type = 'meta'
"""

# Load dynamic model parameters from reach.json
dyn_n_layers, dyn_layer_size, dyn_batch_size, dyn_learning_rate, M, K = model_params(experiment, model_type, model_id)
# Load mpc_controller parameters from reach.json
num_simulated_paths, horizon = MPCcontroller_params(experiment, model_type, model_id)
# Load parameters for collecting on-policy data
new_paths_per_iteration, length_of_new_paths = on_policy_sampling_params(experiment, model_type, model_id)
# Load parameters for training
number_of_random_samples, iterations, training_epochs = training_params(experiment, model_type, model_id)

trajectory_length = M + K


def draw_training_samples():
    """
    draws random trajectories of length M+K
    The first M steps of the trajectory are used for the task specific update step
    the following K steps are used for the Loss calculation for the Meta Update
    In General M > K. Standard values M=32 and K=16 are taken from appendix of the paper
    """
    # TODO: load data of multiple sources with different crippled joints
    states_rand, actions_rand, state_deltas_rand = load_random_samples(experiment=experiment)
    states_rl, actions_rl, state_deltas_rl = load_rl_samples(model_id=model_id, model_type=model_type,
                                                             experiment=experiment)

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


def save_rewards(rewards, model_type):
    average_reward = sum(rewards) / new_paths_per_iteration
    print(f'average_reward: {average_reward}')

    file_name = f'../data/on_policy/{experiment}/{model_type}/model{model_id}/rewards.csv'
    with open(file_name, "a+") as file:
        wr = csv.writer(file)
        wr.writerow(rewards)


# TODO: replace constant values to variables declared in header
if __name__ == "__main__":
    controller_env = ReachEnvJointVelCtrl(render=False, nsubsteps=10)
    env = ReachEnvJointVelCtrl(render=False, nsubsteps=10)

    normalization = load_normalization_variables(experiment=experiment)

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
        model_path = f'../models/{experiment}/{model_type}/model_{model_id}/iteration_{previous_checkpoint}.hdf5'
        model = keras.models.load_model(filepath=model_path)
        dyn_model = dynamicsModel(env=env, normalization=normalization, model=model)

    # init the mpc controller
    mpc_controller = MPCcontroller(env=controller_env, dyn_model=dyn_model, horizon=horizon,
                                   num_simulated_paths=num_simulated_paths, exp3=True)

    # sample new training examples
    # retrain the model
    for iteration in range(iterations):
        print(f'iteration: {iteration}')
        dyn_model.fit(*draw_training_samples(), N_EPOCHS=training_epochs)
        if experiment == 'exp2':
            random_env_index = np.random.randint(6)
            env.set_crippled(cripple_options[random_env_index])
            print("Change dynamic of environment")
            print(cripple_options[random_env_index])

        # Generate new trajectories with the MPC controllers
        paths, rewards, costs = sample(env, mpc_controller, horizon=length_of_new_paths,
                                       num_paths=new_paths_per_iteration, finish_when_done=True, with_adaptaion=meta)

        save_rewards(rewards, model_type=model_type)
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        next_observations = np.concatenate([path["next_observations"] for path in paths])
        observation_delta = next_observations - observations
        # store on policy data
        store_in_file(observations, actions, observation_delta, experiment=experiment, model_id=model_id,
                      model_type=model_type)

        if experiment == 'exp2' and not meta:
            # Generate trajectories with adaptation
            paths, rewards, costs = sample(env, mpc_controller, horizon=length_of_new_paths,
                                           num_paths=new_paths_per_iteration, finish_when_done=True,
                                           with_adaptaion=True)

            save_rewards(rewards, model_type='online_adaptation')
            observations = np.concatenate([path["observations"] for path in paths])
            actions = np.concatenate([path["actions"] for path in paths])
            next_observations = np.concatenate([path["next_observations"] for path in paths])
            observation_delta = next_observations - observations
            # store on policy data
            store_in_file(observations, actions, observation_delta, experiment=experiment, model_id=model_id,
                          model_type='online_adaptation')

        # save the latest model
        model_path = f'../models/{experiment}/{model_type}/model_{model_id}/iteration_{iteration + previous_checkpoint + 1}.hdf5'
        dyn_model.model.save(filepath=model_path)
        print('Model saved')
