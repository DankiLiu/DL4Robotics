import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os

from Danki_Tobias.data_scripts.data_reader import *
from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl
from dynamicsModel import NNDynamicsModel
from controller import MPCcontroller

random_data_file = 'random_samples_2020-12-16_21-18'


def load_model(env):
    normalization = load_normalization_variables(random_data_file)

    model = keras.models.load_model(filepath=f'../models/iteration_{2}.hdf5')
    dyn_model = NNDynamicsModel(env=env, normalization=normalization, model=model)
    return dyn_model


def sample(env,
           controller,
           num_paths=10,
           horizon=1000):
    """
        Write a sampler function which takes in an environment, a controller (either random or the MPC controller),
        and returns rollouts by running on the env.
        Each path can have elements for observations, next_observations, rewards, returns, actions, etc.
    """
    paths = []
    rewards = []
    costs = []
    print("num_sum_path", num_paths)
    for i in range(num_paths):
        print("path :", i)
        states = list()
        actions = list()
        next_states = list()
        states.append(env.reset()[0:14])
        # print(np.array(states).shape)
        total_reward = 0
        total_cost = 0
        for j in range(horizon):
            if j % 100 == 0:
                print(j)
            act, c = controller.get_action(states[j], env.sim.get_state())
            actions.append(act)

            obs, r, done, _ = env.step(np.append(actions[j], 0.4))  # append value for gripper

            # extract relevant state information
            next_states.append(obs[0:14])
            if j != horizon - 1:
                states.append(next_states[j])
            total_reward += r
            total_cost += c

        path = {'observations': np.array(states),
                'actions': np.array(actions),
                'next_observations': np.array(next_states)
                }
        paths.append(path)
        rewards.append(total_reward)
        costs.append(total_cost)

    return paths, rewards, costs


if __name__ == "__main__":
    controller_env = ReachEnvJointVelCtrl(render=False, nsubsteps=10, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))
    env = ReachEnvJointVelCtrl(render=True, nsubsteps=10, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))

    dyn_model = load_model(env)

    # init the mpc controller
    mpc_controller = MPCcontroller(env=controller_env, dyn_model=dyn_model, )

    sample(env, mpc_controller, horizon=33, num_paths=2)
