import numpy as np
import pathlib

from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl
from Danki_Tobias.helper.get_parameters import data_collection_params

current_path = pathlib.Path().absolute()

# Select the form of the data to collect by choosing on of the following
# data_type = 'position'
# data_type = 'position_deltas'
# data_type = 'position_and_velocity'
data_type = 'position_and_velocity_deltas'

data_path = str(current_path.parent) + f"/data/{data_type}/random_samples"

state_delta = True
if data_type == 'position' or data_type == 'position_and_velocity':
    state_delta = False

if data_type == 'position' or data_type == 'position_deltas':
    from Danki_Tobias.data_scripts.collect_random_data_position_only import CollectRandomData
else:
    from Danki_Tobias.data_scripts.collect_random_data import CollectRandomData


def collect_non_crippled_data():
    env = ReachEnvJointVelCtrl(render=False, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]), nsubsteps=10)
    data_collector = CollectRandomData(num_rollouts_train=50, num_rollouts_val=10, steps_per_rollout_train=1000,
                                       steps_per_rollout_val=1000, dataset_name="samples", env=env,
                                       path=f'{data_path}/non_crippled/', state_delta=state_delta)
    data_collector.perform_data_collection()


def collect_data_from_multiple_envs():
    cripple_options_training = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 0, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 0, 1, 1, 1],
                                         [0.5, 1, 1, 1, 0, 0.3, 1, 1],
                                         [0.8, 0.9, 0.6, 0.8, 0.5, 1, 0.7, 1],
                                         [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1]])
    names = ['non_crippled', 'third_disabled', 'fifth_disabled', '167_partial', '123457_partial', '1234567_partial']
    for c, name in zip(cripple_options_training, names):
        env = ReachEnvJointVelCtrl(render=False, crippled=c, nsubsteps=10)
        data_collector = CollectRandomData(num_rollouts_train=20, num_rollouts_val=2, steps_per_rollout_train=1000,
                                           steps_per_rollout_val=1000, dataset_name=name, env=env,
                                           path=f'{data_path}/multiple_envs/', state_delta=state_delta)
        data_collector.perform_data_collection()


if __name__ == '__main__':
    collect_non_crippled_data()
    collect_data_from_multiple_envs()
