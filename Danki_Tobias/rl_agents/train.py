import numpy as np
import pandas as pd
import tensorflow as tf

from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl
from dynamicsModel import NNDynamicsModel
from controller import MPCcontroller


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


def sample(env,
           controller,
           num_paths=10,
           horizon=1000,
           render=False,
           verbose=False):
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
            if (j % 100 == 0):
                print(j)
            act, c = controller.get_action(states[j])
            actions.append(act)
            obs, r, done, _ = env.step(actions[j])
            next_states.append(obs)
            if j != horizon - 1:
                states.append(next_states[j])
            total_reward += r
            total_cost += c
        # print(np.array(next_states).shape)
        # print(np.array(states).shape)
        path = {'observations': np.array(states),
                'actions': np.array(actions),
                'next_observations': np.array(next_states)
                }
        paths.append(path)
        rewards.append(total_reward)
        costs.append(total_cost)

    return paths, rewards, costs


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


    # init the mpc controller
    mpc_controller = MPCcontroller(env=env,
                                   dyn_model=dyn_model,)

    # fit with initial training data (random sampled)
    # dyn_model.fit(states, actions, state_deltas)


    # sample new training examples
    # retrain the model
    while True:
       #  states, actions, state_deltas = load_random_samples()

        # Generate trajectories from MPC controllers

        paths, rewards, costs = sample(env, mpc_controller)
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        next_observations = np.concatenate([path["next_observations"] for path in paths])
        observation_delta = next_observations - observations

        """
        data = {'observations': np.concatenate((data['observations'],
                                                obs)), 'actions': np.concatenate((data['actions'],
                                                                                  acs)),
                'delta': np.concatenate((data['delta'],
                                         delta))}"""

        dyn_model.predict(states, actions, state_deltas)

    """
    mpc_controller = MPCcontroller(env=env,
                                   dyn_model=dyn_model,
                                   horizon=15,
                                   cost_fn=cost_fn,
                                   num_simulated_paths=num_simulated_paths)
    """
