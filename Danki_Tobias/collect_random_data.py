import datetime as dt
import numpy as np
import pandas as pd
import os
from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl

folder_path = './data/reach_env/'
dateTimeObj = dt.datetime.now()

timestamp = str(dateTimeObj.year) + '-' + str(dateTimeObj.month) + '-' + str(dateTimeObj.day) + '_' + \
            str(dateTimeObj.hour) + '-' + str(dateTimeObj.minute)
'''
Generate #steps_per_rollout samples for every rollout.
'''

def random_rollout(steps_per_rollout):
    # Init environment
    env = ReachEnvJointVelCtrl(render=False, nsubsteps=10)

    state_positions = []
    state_velocities = []
    actions = []
    state_delta_positions = []
    state_delta_velocities = []
    """
    Todo: case without velocity
    """
    for step in range(steps_per_rollout):
        old_state_position = env.agent.state[:7]  # 7 joint positions
        old_state_velocity = env.agent.state[7:14]  # 7 joint velocities
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        new_state_positions = env.agent.state[:7]  # 7 new joint positions
        new_state_velocities = env.agent.state[7:14]  # 7 new joint velocities
        # Calculate delta using new and old states
        state_delta_position = new_state_positions - old_state_position
        state_delta_velocity = new_state_velocities - old_state_velocity

        state_positions.append(old_state_position)
        state_velocities.append(old_state_velocity)
        actions.append(action)
        state_delta_positions.append(state_delta_position)
        state_delta_velocities.append(state_delta_velocity)

    df = samples_array_to_df(rollout_num, state_positions, state_velocities, state_delta_positions, state_delta_velocities, actions, is_val=is_val)
    return df


def samples_array_to_df(rollout_num, state_positions, state_velocities, state_delta_positions, state_delta_velocities, actions, is_val):
    # 7, 7
    state_positions = np.array(state_positions)
    state_velocities = np.array(state_velocities)
    state_delta_positions = np.array(state_delta_positions)
    state_delta_velocities = np.array(state_delta_velocities)
    actions = np.array(actions)
    # All should have same length
    assert state_positions.shape[0] == state_velocities.shape[0] == \
           state_delta_positions.shape[0] == state_delta_velocities.shape[0] \
           == actions.shape[0]

    dict = {}
    for i in range(7):
        dict["rollout_num"] = rollout_num * 7
        dict["state_position_"+ str(i)] = state_positions[:, i]
        dict["state_velocities_" + str(i)] = state_velocities[:, i]
        dict["actions_" + str(i)] = actions[:, i]
        dict["state_delta_positions_" + str(i)] = state_delta_positions[:, i]
        dict["state_delta_velocities_" + str(i)] = state_delta_velocities[:, i]

    df = pd.DataFrame(dict)
    print(df.head)
    return df
"""
Todo: store data correctly
look at the step in the paper
"""


def collect_random_samples(number_rollouts, steps_per_rollout, is_val=False):
    all_rollouts: pd.DataFrame = pd.DataFrame()
    print("Start collecting samples ... ")
    for rollout in range(number_rollouts):
        print("-----rollout no.", rollout, "-------")
        df = random_rollout(rollout, steps_per_rollout, is_val=is_val)
        store_in_file(df, is_val=is_val)

def store_in_file(rollout_df: pd.DataFrame, is_val):
    print("Storing data ... ")

    def store(file_name, rollout_df=rollout_df):
        if os.path.isfile(file_name):
            df_old = pd.read_csv(file_name)
            df = pd.concat([df_old, rollout_df])
            df.reset_index(drop=True, inplace=True)
            print("append new data to file")
            df.to_csv(file_name, index=False)
        else:
            rollout_df.to_csv(file_name, index=False)
    if is_val:
        file_name = folder_path + 'random_samples_val_' + str(timestamp) + '.csv'
        print("in data path ", file_name)
        store(file_name, rollout_df)
    else:
        file_name = folder_path + 'random_samples_' + str(timestamp) + '.csv'
        print("in data path ", file_name)
        store(file_name, rollout_df)

num_rollouts_train = 7
num_rollouts_val = 7

steps_per_rollout_train = 10
steps_per_rollout_val = 10

collect_random_samples(number_rollouts=num_rollouts_train, steps_per_rollout=steps_per_rollout_train)
#collect_random_samples(number_rollouts=num_rollouts_val, steps_per_rollout=steps_per_rollout_val, is_val=True)

