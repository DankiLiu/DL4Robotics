import numpy as np
import pandas as pd
import tensorflow as tf

from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl
from dynamicsModel import NNDynamicsModel


# from controller import MPCcontroller


def load_random_samples():
    df = pd.read_csv('../data/reach_env/random_samples_2020-12-16_21-18.csv', index_col=0)
    states = df[[
        'state_position_0', 'state_position_1', 'state_position_2', 'state_position_3', 'state_position_4',
        'state_position_5', 'state_position_6', 'state_velocity_0', 'state_velocity_1', 'state_velocity_2',
        'state_velocity_3', 'state_velocity_4', 'state_velocity_5', 'state_velocity_6']]
    actions = df[[
        'action_0', 'action_1', 'action_2', 'action_3', 'action_4', 'action_5', 'action_6']]
    state_deltas = df[[
        'state_delta_position_0', 'state_delta_position_1', 'state_delta_position_2', 'state_delta_position_3',
        'state_delta_position_4', 'state_delta_position_5', 'state_delta_position_6',
        'state_delta_velocity_0', 'state_delta_velocity_1', 'state_delta_velocity_2', 'state_delta_velocity_3',
        'state_delta_velocity_4', 'state_delta_velocity_5', 'state_delta_velocity_6']]
    return states, actions, state_deltas


def compute_normalization_variables(data):
    mean = data.mean()
    std = data.std()
    return [mean, std]


if __name__ == "__main__":
    env = ReachEnvJointVelCtrl(render=True, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))

    states, actions, state_deltas = load_random_samples()

    normalization = dict()
    normalization['observations'] = compute_normalization_variables(states)
    normalization['actions'] = compute_normalization_variables(actions)
    normalization['delta'] = compute_normalization_variables(state_deltas)

    dyn_model = NNDynamicsModel(env=env,
                                n_layers=2,
                                size=500,
                                activation=tf.tanh,
                                output_activation=None,
                                normalization=normalization,
                                batch_size=32,  # 512,
                                iterations=150,
                                learning_rate=1e-3)

    dyn_model.fit(states, actions, state_deltas)
    dyn_model.predict(states, actions)

    """
    mpc_controller = MPCcontroller(env=env,
                                   dyn_model=dyn_model,
                                   horizon=15,
                                   cost_fn=cost_fn,
                                   num_simulated_paths=num_simulated_paths)
    """
