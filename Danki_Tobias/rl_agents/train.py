import numpy as np
import pandas as pd
import tensorflow as tf

from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl
from dynamicsModel import NNDynamicsModel
from controller import MPCcontroller

sess = tf.Session()


def compute_normalization(data):
    """
    Write a function to take in a dataset and compute the means, and stds.
    Return 6 elements: mean of s_t, std of s_t, mean of (s_t+1 - s_t), std of (s_t+1 - s_t), mean of actions, std of actions
    """

    """ YOUR CODE HERE """
    mean_obs = np.mean(data['observations'], axis=0)
    std_obs = np.std(data['observations'], axis=0)
    mean_deltas = np.mean(data['delta'], axis=0)
    std_deltas = np.std(data['delta'], axis=0)
    mean_actions = np.mean(data['actions'], axis=0)
    std_actions = np.std(data['actions'], axis=0)
    return mean_obs, std_obs, mean_deltas, std_deltas, mean_actions, std_actions


if __name__ == "__main__":
    env = ReachEnvJointVelCtrl(render=True, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))

    data = pd.DataFrame()  # load D_rand


    mean_obs, std_obs, mean_deltas, std_deltas, mean_actions, std_actions = compute_normalization(data)
    normalization = dict()
    normalization['observations'] = [mean_obs, std_obs]
    normalization['actions'] = [mean_actions, std_actions]
    normalization['delta'] = [mean_deltas, std_deltas]

    dyn_model = NNDynamicsModel(env=env,
                                n_layers=2,
                                size=500,
                                activation=tf.tanh,
                                output_activation=None,
                                normalization=normalization,
                                batch_size=512,
                                iterations=150,
                                learning_rate=1e-3,
                                sess=sess)

    mpc_controller = MPCcontroller(env=env,
                                   dyn_model=dyn_model,
                                   horizon=15,
                                   cost_fn=cost_fn,
                                   num_simulated_paths=num_simulated_paths)
