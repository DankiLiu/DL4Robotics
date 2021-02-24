import numpy as np
import pathlib

# from Danki_Tobias.data_scripts.collect_random_data import CollectRandomData
from Danki_Tobias.data_scripts.collect_random_data_2 import CollectRandomData
from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl
from Danki_Tobias.helper.get_parameters import data_collection_params

current_path = pathlib.Path().absolute()
reach_env_data_path = str(current_path.parent) + "/data/reach_env"

# TODO: write params into a separate file

not_crippled = np.ones(8)

disabled_joints_01 = np.array([1, 1, 0, 1, 1, 1, 1, 1])

disabled_joints_02 = np.array([1, 1, 1, 1, 0, 1, 1, 1])

# Test: collect random data with disabled_joints_01
if __name__ == '__main__':
    num_rollouts_train, num_rollouts_val, steps_per_rollout_train, steps_per_rollout_val = data_collection_params()
    print("create agent")
    agent = ReachEnvJointVelCtrl(render=False, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]), nsubsteps=10)
    print("create data collector")
    """data_collector = CollectRandomData(num_rollouts_train,
                                       num_rollouts_val,
                                       steps_per_rollout_train,
                                       steps_per_rollout_val,
                                       "d_03", # disabled third joint
                                       agent,
                                       reach_env_data_path + "/_d_03")"""
    data_collector = CollectRandomData(50,
                                       0,
                                       500,
                                       0,
                                       "test",  # disabled third joint
                                       agent,
                                       reach_env_data_path)
    data_collector.perform_data_collection()
