import numpy as np
import pandas as pd
import tensorflow as tf

from Danki_Tobias.column_names import *
from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl
from dynamicsModel import NNDynamicsModel
from controller import MPCcontroller


# from controller import MPCcontroller


def load_random_samples():  # TODO: pick dataset to load
    df = pd.read_csv('../data/reach_env/random_samples_2020-12-16_21-18.csv', index_col=0)
    states = df[state_columns]
    actions = df[action_columns]
    state_deltas = df[delta_columns]
    return states, actions, state_deltas


def compute_normalization_variables(data):
    mean = data.mean()
    std = data.std()
    return [mean, std]


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
            act, c = controller.get_action(states[j])
            actions.append(act)

            obs, r, done, _ = env.step(np.append(actions[j], 0.4))  # append value for gripper

            # extract relevant state information
            next_states.append(obs[0:14])
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
    env = ReachEnvJointVelCtrl(render=False, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))

    # Load D_rand
    states_rand, actions_rand, state_deltas_rand = load_random_samples()

    # initialize empty DataFrames representing D_rl
    states_rl = pd.DataFrame([], columns=state_columns)
    actions_rl = pd.DataFrame([], columns=action_columns)
    state_deltas_rl = pd.DataFrame([], columns=delta_columns)

    normalization = dict()
    normalization['observations'] = compute_normalization_variables(states_rand)
    normalization['actions'] = compute_normalization_variables(actions_rand)
    normalization['delta'] = compute_normalization_variables(state_deltas_rand)

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
                                   dyn_model=dyn_model, )

    # sample new training examples
    # retrain the model
    for iteration in range(3):
        print(f'iteration: {iteration}')
        dyn_model.fit(*draw_training_samples(5))

        # Generate trajectories from MPC controllers
        paths, rewards, costs = sample(env, mpc_controller, horizon=2, num_paths=2)

        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        next_observations = np.concatenate([path["next_observations"] for path in paths])
        observation_delta = next_observations - observations

        states_rl = states_rl.append(pd.DataFrame(observations, columns=state_columns))
        actions_rl = actions_rl.append(pd.DataFrame(actions, columns=action_columns))
        state_deltas_rl = state_deltas_rl.append(pd.DataFrame(observation_delta, columns=delta_columns))
