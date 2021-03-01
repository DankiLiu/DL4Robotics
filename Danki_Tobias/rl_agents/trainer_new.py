import numpy as np
import pandas as pd
import tensorflow as tf
import os
import csv
from tensorflow import keras

from Danki_Tobias.data_scripts.data_reader_new import *
from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl, ReachEnvJointTorqueCtrl
from Danki_Tobias.rl_agents.dynamicsModelNew import *
from Danki_Tobias.rl_agents.metaRLDynamicsModelNew import *
from Danki_Tobias.rl_agents.controller import MPCcontroller, sample
from Danki_Tobias.helper.get_parameters_new import *

data_type_options = ['position', 'position_deltas', 'position_and_velocity', 'position_and_velocity_deltas']
train_on_options = ['non_crippled', 'multiple_envs']

data_type = data_type_options[3]
train_on = train_on_options[0]

algorithm = 'meta'  # normal
meta = algorithm == 'meta'

# if new model = True a new model is created,
# else set previous_checkpoint to latest finished training iteration to continue training
new_model = True
previous_checkpoint = 0

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

cripple_options = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 0, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 0, 1, 1, 1],
                            [0.5, 1, 1, 1, 0, 0.3, 1, 1],
                            [0.8, 0.9, 0.6, 0.8, 0.5, 1, 0.7, 1],
                            [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1]])

# Load dynamic model parameters from reach.json
dyn_n_layers, dyn_layer_size, dyn_batch_size, dyn_learning_rate, M, K = model_params(data_type, train_on, algorithm)
# Load mpc_controller parameters from reach.json
num_simulated_paths, horizon = MPCcontroller_params(data_type, train_on, algorithm)
# Load parameters for collecting on-policy data
new_paths_per_iteration, length_of_new_paths = on_policy_sampling_params(data_type, train_on, algorithm)
# Load parameters for training
number_of_random_samples, iterations, training_epochs = training_params(data_type, train_on, algorithm)

trajectory_length = M + K


def draw_training_samples():
    """
    draws random trajectories of length M+K
    The first M steps of the trajectory are used for the task specific update step
    the following K steps are used for the Loss calculation for the Meta Update
    In General M > K. Standard values M=32 and K=16 are taken from appendix of the paper
    """
    # TODO: load data of multiple sources with different crippled joints
    states_rand, actions_rand, labels_rand = data_reader.load_random_samples()
    states_rl, actions_rl, labels_rl = data_reader.load_rl_samples(algorithm=algorithm)

    all_states = states_rl.append(states_rand)
    all_states = all_states.reset_index(drop=True)
    all_actions = actions_rl.append(actions_rand)
    all_actions = all_actions.reset_index(drop=True)
    all_labels = labels_rl.append(labels_rand)
    all_labels = all_labels.reset_index(drop=True)

    if meta:  # draw trajectories
        assert (number_of_random_samples / trajectory_length).is_integer()
        num_trajectories = int(number_of_random_samples / trajectory_length)
        random = np.random.randint(len(all_states) - trajectory_length, size=num_trajectories)
        func = lambda v: np.arange(start=v, stop=v + trajectory_length)
        random = (np.array([func(v) for v in random])).flatten()
    else:  # draw single state changes
        random = np.random.randint(len(all_states), size=number_of_random_samples)

    states_sample = all_states.iloc[random]
    actions_sample = all_actions.iloc[random]
    all_labels_sample = all_labels.iloc[random]
    # TODO: Check why index are not same (It should work without reset_index)
    return states_sample.reset_index(drop=True), actions_sample.reset_index(drop=True), all_labels_sample.reset_index(
        drop=True)


# TODO: replace constant values to variables declared in header
if __name__ == "__main__":
    controller_env = ReachEnvJointVelCtrl(render=False, nsubsteps=10)
    env = ReachEnvJointVelCtrl(render=False, nsubsteps=10)
    data_reader = DataReader(data_type=data_type, train_on=train_on)

    normalization = data_reader.load_normalization_variables()

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
        model_path = f'../models/{data_type}/trained_on_{train_on}/{algorithm}/iteration_{previous_checkpoint}.hdf5'
        model = keras.models.load_model(filepath=model_path)
        dyn_model = dynamicsModel(env=env, normalization=normalization, model=model)

    # init the mpc controller
    mpc_controller = MPCcontroller(env=controller_env, dyn_model=dyn_model, horizon=horizon,
                                   num_simulated_paths=num_simulated_paths)

    # sample new training examples
    # retrain the model
    for iteration in range(iterations):
        print(f'iteration: {iteration}')
        dyn_model.fit(*draw_training_samples(), N_EPOCHS=training_epochs)
        if train_on == 'multiple_envs':
            random_env_index = np.random.randint(6)
            env.set_crippled(cripple_options[random_env_index])
            print("Change dynamic of environment")
            print(cripple_options[random_env_index])

        # Generate new trajectories with the MPC controllers
        paths, rewards, costs = sample(env, mpc_controller, horizon=length_of_new_paths,
                                       num_paths=new_paths_per_iteration, finish_when_done=True, with_adaptaion=meta)
        data_reader.save_rewards(rewards, algorithm, new_paths_per_iteration)
        states = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        labels = np.concatenate([path["next_observations"] for path in paths])
        if not predicts_state:
            labels = labels - states
        # store on policy data
        data_reader.store_in_file(states, actions, labels, algorithm)

        if train_on == 'multiple_envs' and not meta:
            # Generate trajectories with adaptation
            paths, rewards, costs = sample(env, mpc_controller, horizon=length_of_new_paths,
                                           num_paths=new_paths_per_iteration, finish_when_done=True,
                                           with_adaptaion=True)
            data_reader.save_rewards(rewards, 'online_adaptation', new_paths_per_iteration)
            states = np.concatenate([path["observations"] for path in paths])
            actions = np.concatenate([path["actions"] for path in paths])
            labels = np.concatenate([path["next_observations"] for path in paths])
            if not predicts_state:
                labels = labels - states
            # store on policy data
            data_reader.store_in_file(states, actions, labels, 'online_adaptation')

        # save the latest model
        model_path = f'../models/{data_type}/trained_on_{train_on}/{algorithm}/iteration_{iteration + previous_checkpoint + 1}.hdf5'
        dyn_model.model.save(filepath=model_path)
        print('Model saved')
