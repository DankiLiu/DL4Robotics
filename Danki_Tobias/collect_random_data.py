import numpy as np
import pandas as pd
from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl

folder_path = './data/reach_env/'


def random_rollout(steps_per_rollout):
    env = ReachEnvJointVelCtrl(render=True, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]), nsubsteps=10)

    states = []
    actions = []
    state_deltas = []
    for step in range(steps_per_rollout):
        old_state = env.agent.state[:14]  # 7 joint position and 7 joint velocity
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        new_state = env.agent.state[:14]
        state_delta = new_state - old_state

        states.append(old_state)
        actions.append(action)
        state_deltas.append(state_delta)

    states = np.array(states)
    actions = np.array(actions)
    state_deltas = np.array(state_deltas)

    df = pd.DataFrame({'state_position_0': states[:, 0],
                       'state_position_1': states[:, 1],
                       'state_position_2': states[:, 2],
                       'state_position_3': states[:, 3],
                       'state_position_4': states[:, 4],
                       'state_position_5': states[:, 5],
                       'state_position_6': states[:, 6],
                       'state_velocity_0': states[:, 7],
                       'state_velocity_1': states[:, 8],
                       'state_velocity_2': states[:, 9],
                       'state_velocity_3': states[:, 10],
                       'state_velocity_4': states[:, 11],
                       'state_velocity_5': states[:, 12],
                       'state_velocity_6': states[:, 13],
                       'action_0': actions[:, 0],
                       'action_1': actions[:, 1],
                       'action_2': actions[:, 2],
                       'action_3': actions[:, 3],
                       'action_4': actions[:, 4],
                       'action_5': actions[:, 5],
                       'action_6': actions[:, 6],
                       'state_delta_position_0': state_deltas[:, 0],
                       'state_delta_position_1': state_deltas[:, 1],
                       'state_delta_position_2': state_deltas[:, 2],
                       'state_delta_position_3': state_deltas[:, 3],
                       'state_delta_position_4': state_deltas[:, 4],
                       'state_delta_position_5': state_deltas[:, 5],
                       'state_delta_position_6': state_deltas[:, 6],
                       'state_delta_velocity_0': state_deltas[:, 7],
                       'state_delta_velocity_1': state_deltas[:, 8],
                       'state_delta_velocity_2': state_deltas[:, 9],
                       'state_delta_velocity_3': state_deltas[:, 10],
                       'state_delta_velocity_4': state_deltas[:, 11],
                       'state_delta_velocity_5': state_deltas[:, 12],
                       'state_delta_velocity_6': state_deltas[:, 13]})
    return df


def collect_random_samples(number_rollouts, steps_per_rollout):
    all_rollouts = pd.DataFrame({'state_position_0': [],
                       'state_position_1': [],
                       'state_position_2': [],
                       'state_position_3': [],
                       'state_position_4': [],
                       'state_position_5': [],
                       'state_position_6': [],
                       'state_velocity_0': [],
                       'state_velocity_1': [],
                       'state_velocity_2': [],
                       'state_velocity_3': [],
                       'state_velocity_4': [],
                       'state_velocity_5': [],
                       'state_velocity_6': [],
                       'action_0': [],
                       'action_1': [],
                       'action_2': [],
                       'action_3': [],
                       'action_4': [],
                       'action_5': [],
                       'action_6': [],
                       'state_delta_position_0': [],
                       'state_delta_position_1': [],
                       'state_delta_position_2': [],
                       'state_delta_position_3': [],
                       'state_delta_position_4': [],
                       'state_delta_position_5': [],
                       'state_delta_position_6': [],
                       'state_delta_velocity_0': [],
                       'state_delta_velocity_1': [],
                       'state_delta_velocity_2': [],
                       'state_delta_velocity_3': [],
                       'state_delta_velocity_4': [],
                       'state_delta_velocity_5': [],
                       'state_delta_velocity_6': []})
    for rollout in range(number_rollouts):
        df = random_rollout(steps_per_rollout)
        df['rollout'] = [rollout] * steps_per_rollout
        all_rollouts = all_rollouts.append(df)

    all_rollouts = all_rollouts.reset_index(drop=True)
    all_rollouts.to_csv(folder_path + 'random_samples.csv')


collect_random_samples(10, 50)
