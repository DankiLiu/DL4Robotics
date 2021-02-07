import datetime as dt
import numpy as np
import pandas as pd
import os
from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl
from Danki_Tobias.helper.get_parameters import data_collection_params

folder_path = '../data/reach_env/'

dateTimeObj = dt.datetime.now()

timestamp = str(dateTimeObj.year) + '-' + str(dateTimeObj.month) + '-' + str(dateTimeObj.day) + '_' + \
            str(dateTimeObj.hour) + '-' + str(dateTimeObj.minute)
'''
Generate #steps_per_rollout samples for every rollout.
'''


class CollectRandomData():
    def __init__(self,
                 env,
                 num_rollouts_train,
                 num_rollouts_val,
                 steps_per_rollout_train,
                 steps_per_rollout_val,
                 dataset_name):
        self.env = env
        self.num_rollouts_train = num_rollouts_train
        self.num_rollouts_val = num_rollouts_val
        self.steps_per_rollout_train = steps_per_rollout_train
        self.steps_per_rollout_val = steps_per_rollout_val
        self.dataset_name = dataset_name

    def random_rollout(self, steps_per_rollout):
        state_positions = []
        state_velocities = []
        actions = []
        state_delta_positions = []
        state_delta_velocities = []
        """
        Todo: case without velocity
        """
        for step in range(steps_per_rollout):
            old_state_position = self.env.agent.state[:7]  # 7 joint positions
            old_state_velocity = self.env.agent.state[7:14]  # 7 joint velocities
            action = self.env.action_space.sample()
            obs, reward, done, _ = self.env.step(action)
            new_state_positions = self.env.agent.state[:7]  # 7 new joint positions
            new_state_velocities = self.env.agent.state[7:14]  # 7 new joint velocities
            # Calculate delta using new and old states
            state_delta_position = new_state_positions - old_state_position
            state_delta_velocity = new_state_velocities - old_state_velocity

            state_positions.append(old_state_position)
            state_velocities.append(old_state_velocity)
            actions.append(action)
            state_delta_positions.append(state_delta_position)
            state_delta_velocities.append(state_delta_velocity)

        return state_positions, state_velocities, state_delta_positions, state_delta_velocities, actions

    def samples_array_to_df(self, rollout_num, state_positions, state_velocities, state_delta_positions,
                            state_delta_velocities, actions):
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
            dict["state_positions_" + str(i)] = state_positions[:, i]
            dict["state_velocities_" + str(i)] = state_velocities[:, i]
            dict["actions_" + str(i)] = actions[:, i]
            dict["state_delta_positions_" + str(i)] = state_delta_positions[:, i]
            dict["state_delta_velocities_" + str(i)] = state_delta_velocities[:, i]
        df = pd.DataFrame(dict)
        print(df.head)
        return df

    def collect_random_samples(self, number_rollouts, steps_per_rollout, is_val=False):
        all_rollouts: pd.DataFrame = pd.DataFrame()
        print("Start collecting samples ... ")
        for rollout in range(number_rollouts):
            print("-----rollout no.", rollout, "-------")
            state_positions, state_velocities, state_delta_positions, state_delta_velocities, actions = self.random_rollout(
                steps_per_rollout)
            df = self.samples_array_to_df(rollout, state_positions, state_velocities, state_delta_positions,
                                          state_delta_velocities, actions)
            self.store_in_file(df, is_val=is_val)

    def store_in_file(self, rollout_df: pd.DataFrame, is_val):
        print("Storing data ... ")

        def store(file_name, rollout_df=rollout_df):
            if os.path.isfile(file_name):
                rollout_df.to_csv(file_name, mode='a', header=False, index=False)
            else:
                rollout_df.to_csv(file_name, index=False)

        if is_val:
            file_name = folder_path + self.dataset_name + '_val_' + str(timestamp) + '.csv'
            print("in data path ", file_name)
            store(file_name, rollout_df)
        else:
            file_name = folder_path + self.dataset_name + '_train_' + str(timestamp) + '.csv'
            print("in data path ", file_name)
            store(file_name, rollout_df)

    def perform_data_collection(self):
        # Collect training data
        print("Start collecting training dataset ... ")
        self.collect_random_samples(number_rollouts=self.num_rollouts_train,
                                    steps_per_rollout=self.steps_per_rollout_train)

        # Collect validation data
        print("Start collecting validation dataset ... ")
        self.collect_random_samples(number_rollouts=self.num_rollouts_val, steps_per_rollout=self.steps_per_rollout_val,
                                    is_val=True)
