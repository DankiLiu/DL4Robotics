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
                 path=reach_env_path
                 ):
        self.env = env
        self.num_rollouts_train = num_rollouts_train
        self.num_rollouts_val = num_rollouts_val
        self.steps_per_rollout_train = steps_per_rollout_train
        self.steps_per_rollout_val = steps_per_rollout_val
        self.dataset_name = dataset_name
        self.path = path

    def random_rollout(self, steps_per_rollout):
        states = []
        actions = []
        next_states = []
        for step in range(steps_per_rollout):
            old_state_position = self.env.agent.state[:7]  # 7 joint positions
            action = self.env.action_space.sample()
            obs, reward, done, _ = self.env.step(action)
            new_state_positions = self.env.agent.state[:7]  # 7 new joint positions

            states.append(old_state_position)
            actions.append(action)
            next_states.append(new_state_positions)

        return states, actions, next_states

    def samples_array_to_df(self, rollout_num, states, actions, next_states):
        # 7, 7
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        # All should have same length
        assert states.shape[0] == next_states.shape[0] == actions.shape[0]

        dict = {}
        for i in range(7):
            dict["rollout_num"] = rollout_num * 7
            dict["state_" + str(i)] = states[:, i]
            dict["actions_" + str(i)] = actions[:, i]
            dict["next_state" + str(i)] = next_states[:, i]
        df = pd.DataFrame(dict)
        print(df.head)
        return df

    def collect_random_samples(self, number_rollouts, steps_per_rollout, is_val=False):
        all_rollouts: pd.DataFrame = pd.DataFrame()
        print("Start collecting samples ... ")
        for rollout in range(number_rollouts):
            print("-----rollout no.", rollout, "-------")
            states, actions, next_states = self.random_rollout(steps_per_rollout)
            df = self.samples_array_to_df(rollout, states, actions, next_states)
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
            file_path = self.path + '/' + self.dataset_name + '_val_' + str(timestamp) + '.csv'
            print("in data path ", file_path)
            store(file_path, rollout_df)
        else:
            file_path = self.path + '/' + self.dataset_name + '_train_' + str(timestamp) + '.csv'
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
