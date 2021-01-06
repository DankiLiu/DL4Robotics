import numpy as np
import pandas as pd
import tensorflow as tf
import os

from Danki_Tobias.data_scripts.data_reader import *
from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl
from dynamicsModel import NNDynamicsModel
from controller import MPCcontroller, sample

random_data_file = 'random_samples_2020-12-16_21-18'


def draw_training_samples(number_of_samples=100):
    all_states = states_rl.append(states_rand)
    all_actions = actions_rl.append(actions_rand)
    all_deltas = state_deltas_rl.append(state_deltas_rand)

    states_sample = all_states.sample(n=number_of_samples, replace=True)
    actions_sample = all_actions.loc[states_sample.index]
    delta_sample = all_deltas.loc[states_sample.index]
    return states_sample, actions_sample, delta_sample


# TODO: replace constant values to variables declared in header
if __name__ == "__main__":
    controller_env = ReachEnvJointVelCtrl(render=False, nsubsteps=10, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))
    env = ReachEnvJointVelCtrl(render=False, nsubsteps=10, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))

    # Load D_rand
    states_rand, actions_rand, state_deltas_rand = load_random_samples(random_data_file)
    normalization = load_normalization_variables(random_data_file)

    # initialize empty DataFrames representing D_rl
    states_rl = pd.DataFrame([], columns=state_columns)
    actions_rl = pd.DataFrame([], columns=action_columns)
    state_deltas_rl = pd.DataFrame([], columns=delta_columns)

    dyn_model = NNDynamicsModel.new_model(env=env,
                                          n_layers=2,
                                          size=500,
                                          activation=tf.tanh,
                                          output_activation=None,
                                          normalization=normalization,
                                          batch_size=32,  # 512,
                                          learning_rate=1e-3)

    # init the mpc controller
    mpc_controller = MPCcontroller(env=controller_env, dyn_model=dyn_model, )

    # sample new training examples
    # retrain the model
    for iteration in range(3):
        print(f'iteration: {iteration}')
        dyn_model.fit(*draw_training_samples(5))

        dyn_model.model.save(filepath=f'../models/iteration_{iteration}.hdf5')

        # Generate trajectories from MPC controllers
        paths, rewards, costs = sample(env, mpc_controller, horizon=2, num_paths=2)

        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        next_observations = np.concatenate([path["next_observations"] for path in paths])
        observation_delta = next_observations - observations

        store_in_file(observations, actions, observation_delta)

        states_rl = states_rl.append(pd.DataFrame(observations, columns=state_columns))
        actions_rl = actions_rl.append(pd.DataFrame(actions, columns=action_columns))
        state_deltas_rl = state_deltas_rl.append(pd.DataFrame(observation_delta, columns=delta_columns))
