import datetime as dt
import numpy as np
import pandas as pd
import os
import pathlib
from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl

# default path to store data
current_path = pathlib.Path().absolute()
reach_env_path = str(current_path.parent) + '/data/reach_env/'

dateTimeObj = dt.datetime.now()

timestamp = str(dateTimeObj.year) + '-' + str(dateTimeObj.month) + '-' + str(dateTimeObj.day) + '_' + \
            str(dateTimeObj.hour) + '-' + str(dateTimeObj.minute)


class CollectRandomData():
    def __init__(self,
                 num_rollouts_train,
                 num_rollouts_val,
                 steps_per_rollout_train,
                 steps_per_rollout_val,
                 dataset_name,
                 env=ReachEnvJointVelCtrl(render=False, nsubsteps=10),  # non_crippled by default
                 path=reach_env_path,
                 state_delta=True
                 ):
        self.env = env
        self.num_rollouts_train = num_rollouts_train
        self.num_rollouts_val = num_rollouts_val
        self.steps_per_rollout_train = steps_per_rollout_train
        self.steps_per_rollout_val = steps_per_rollout_val
        self.dataset_name = dataset_name
        self.path = path
        self.state_delta = state_delta

    def random_rollout(self, steps_per_rollout):
        state_positions = []
        state_velocities = []
        actions = []
        next_state_positions = []
        next_state_velocities = []
        state_delta_positions = []
        state_delta_velocities = []
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
            next_state_positions.append(new_state_positions)
            next_state_velocities.append(new_state_velocities)
            state_delta_positions.append(state_delta_position)
            state_delta_velocities.append(state_delta_velocity)

        if self.state_delta:
            return state_positions, state_velocities, state_delta_positions, state_delta_velocities, actions
        else:
            return state_positions, state_velocities, next_state_positions, next_state_velocities, actions

    def samples_array_to_df(self, rollout_num, state_positions, state_velocities, label_positions,
                            label_velocities, actions):
        # 7, 7
        state_positions = np.array(state_positions)
        state_velocities = np.array(state_velocities)
        label_positions = np.array(label_positions)
        label_velocities = np.array(label_velocities)
        actions = np.array(actions)
        # All should have same length
        assert state_positions.shape[0] == state_velocities.shape[0] == \
               label_positions.shape[0] == label_velocities.shape[0] \
               == actions.shape[0]

        dict = {}
        for i in range(7):
            dict["rollout_num"] = rollout_num * 7
            dict["state_positions_" + str(i)] = state_positions[:, i]
            dict["state_velocities_" + str(i)] = state_velocities[:, i]
            dict["actions_" + str(i)] = actions[:, i]
            dict["label_positions_" + str(i)] = label_positions[:, i]
            dict["label_velocities_" + str(i)] = label_velocities[:, i]
        df = pd.DataFrame(dict)
        print(df.head)
        return df

    def collect_random_samples(self, number_rollouts, steps_per_rollout, is_val=False):
        all_rollouts: pd.DataFrame = pd.DataFrame()
        print("Start collecting samples ... ")
        for rollout in range(number_rollouts):
            print("-----rollout no.", rollout, "-------")
            state_positions, state_velocities, label_positions, label_velocities, actions = self.random_rollout(
                steps_per_rollout)
            df = self.samples_array_to_df(rollout, state_positions, state_velocities, label_positions,
                                          label_velocities, actions)
            self.store_in_file(df, is_val=is_val)

    def store_in_file(self, rollout_df: pd.DataFrame, is_val):
        print("Storing data ... ")
        # If path doesn't exist, create one
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        def store(name, rollout_df=rollout_df):
            if os.path.isfile(name):
                rollout_df.to_csv(name, mode='a', header=False, index=False)
            else:
                rollout_df.to_csv(name, index=False)

        if is_val:
            file_path = self.path + 'validation/' + self.dataset_name + '.csv'
            print("in data path ", file_path)
            store(file_path, rollout_df)
        else:
            file_path = self.path + 'training/' + self.dataset_name + '.csv'
            print("in data path ", file_path)
            store(file_path, rollout_df)

    def perform_data_collection(self):
        # Collect training data
        print("Start collecting training dataset ... ")
        self.collect_random_samples(number_rollouts=self.num_rollouts_train,
                                    steps_per_rollout=self.steps_per_rollout_train)

        # Collect validation data
        print("Start collecting validation dataset ... ")
        if self.num_rollouts_val:
            self.collect_random_samples(number_rollouts=self.num_rollouts_val,
                                        steps_per_rollout=self.steps_per_rollout_val,
                                        is_val=True)
