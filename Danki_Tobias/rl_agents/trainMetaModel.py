import numpy as np
import pandas as pd
import tensorflow as tf
import os

from Danki_Tobias.data_scripts.data_reader import *
from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl
from metaRLDynamicsModel import MetaRLDynamicsModel
from controller import MPCcontroller, sample

random_data_file = 'random_samples_2021-1-6_11-49'
# random_data_file = 'random_samples_2020-12-16_21-18' # small datafile for testing purpose

iterations = 100
training_epochs = 20

new_paths_per_iteration = 10
length_of_new_paths = 500


def draw_training_samples(number_of_samples=2, trajectory_length=32 + 16):
    """
    draws random trajectories of length M+K
    The first M steps of the trajectory are used for the task specific update step
    the following K steps are used for the Loss calculation for the Meta Update
    In General M > K. Standard values M=32 and K=16 are taken from appendix of the paper
    """
    # TODO: load data of multiple sources with different crippled joints
    states_rand, actions_rand, state_deltas_rand = load_random_samples(random_data_file)
    states_rl, actions_rl, state_deltas_rl = load_rl_samples()

    all_states = states_rl.append(states_rand)
    all_states = all_states.reset_index(drop=True)
    all_actions = actions_rl.append(actions_rand)
    all_actions = all_actions.reset_index(drop=True)
    all_deltas = state_deltas_rl.append(state_deltas_rand)
    all_deltas = all_deltas.reset_index(drop=True)

    #
    random = np.random.randint(len(all_states) - trajectory_length, size=number_of_samples)
    func = lambda v: np.arange(start=v, stop=v + trajectory_length)
    random = (np.array([func(v) for v in random])).flatten()

    states_sample = all_states.iloc[random]
    actions_sample = all_actions.iloc[random]
    delta_sample = all_deltas.iloc[random]

    return states_sample, actions_sample, delta_sample


# TODO: replace constant values to variables declared in header
if __name__ == "__main__":
    controller_env = ReachEnvJointVelCtrl(render=False, nsubsteps=10, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))
    env = ReachEnvJointVelCtrl(render=False, nsubsteps=10, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))

    normalization = load_normalization_variables(random_data_file)
    dyn_model = MetaRLDynamicsModel.new_model(env=env,
                                              n_layers=2,
                                              size=64,
                                              activation=tf.tanh,
                                              output_activation=None,
                                              normalization=normalization,
                                              batch_size=512,
                                              learning_rate=1e-3)

    # init the mpc controller
    mpc_controller = MPCcontroller(env=controller_env, dyn_model=dyn_model, horizon=1, num_simulated_paths=500)

    # sample new training examples
    # retrain the model
    for iteration in range(iterations):
        print(f'iteration: {iteration}')
        dyn_model.fit(*draw_training_samples(), N_EPOCHS=training_epochs)

        dyn_model.model.save(filepath=f'../models/iteration_{iteration}.hdf5')

        if False:
            # Generate trajectories from MPC controllers
            paths, rewards, costs = sample(env, mpc_controller, horizon=length_of_new_paths,
                                           num_paths=new_paths_per_iteration)

            observations = np.concatenate([path["observations"] for path in paths])
            actions = np.concatenate([path["actions"] for path in paths])
            next_observations = np.concatenate([path["next_observations"] for path in paths])
            observation_delta = next_observations - observations

            store_in_file(observations, actions, observation_delta)
