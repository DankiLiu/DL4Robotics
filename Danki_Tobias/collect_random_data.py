import datetime as dt
import numpy as np
import pandas as pd
from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl

folder_path = './data/reach_env/'
dateTimeObj = dt.datetime.now()
timestamp = str(dateTimeObj.year) + '-' + str(dateTimeObj.month) + '-' + str(dateTimeObj.day) + '_' + str(
    dateTimeObj.hour) + '-' + str(dateTimeObj.minute)

'''
Generate #steps_per_rollout samples for every rollout.
'''


def random_rollout(steps_per_rollout):
    # Init environment
    env = ReachEnvJointVelCtrl(render=True, nsubsteps=10)

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

    df = samples_array_to_df(state_positions, state_velocities, state_delta_positions, state_delta_velocities, actions)
    return df


def samples_array_to_df(state_positions, state_velocities, state_delta_positions, state_delta_velocities, actions):
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
        dict["state_position_" + str(i)] = state_positions[:, i]
        dict["state_velocity_" + str(i)] = state_velocities[:, i]
        dict["action_" + str(i)] = actions[:, i]
        dict["state_delta_position_" + str(i)] = state_delta_positions[:, i]
        dict["state_delta_velocity_" + str(i)] = state_delta_velocities[:, i]

    df = pd.DataFrame(dict)
    return df


def collect_random_samples(number_rollouts, steps_per_rollout):
    all_rollouts: pd.DataFrame = pd.DataFrame()
    for rollout in range(number_rollouts):
        df = random_rollout(steps_per_rollout)
        df["rollout"] = [rollout] * steps_per_rollout
        all_rollouts = all_rollouts.append(df)

    all_rollouts = all_rollouts.reset_index(drop=True)
    print(all_rollouts.head())

    all_rollouts.to_csv(folder_path + 'random_samples_' + str(timestamp) + '.csv')


collect_random_samples(3000, 50)
