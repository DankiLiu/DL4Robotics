import numpy as np
import pathlib

from Danki_Tobias.data_scripts.collect_random_data import CollectRandomData
from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl
from Danki_Tobias.helper.get_parameters import data_collection_params

current_path = pathlib.Path().absolute()
reach_env_data_path = str(current_path.parent) + "/data/reach_env"

# TODO: write params into a separate file

class AgentJoints():
    def __init__(self, joints, name):
        self.joints = np.array(joints)
        self.name = name

    def print_agnet_info(self):
        print("joints of agent <{}> are {}".format(self.name, self.joints))


# Test: collect random data with disabled_joints_01
if __name__ == '__main__':
    num_rollouts_train, num_rollouts_val, steps_per_rollout_train, steps_per_rollout_val = data_collection_params()
    """
    agents_joints = [AgentJoints(np.array([1, 1, 1, 1, 1, 1, 1, 1]), "non-crippled"),
                     AgentJoints(np.array([1, 1, 0, 1, 1, 1, 1, 1]), "third_disabled"),
                     AgentJoints(np.array([1, 1, 1, 1, 0, 1, 1, 1]), "fifth_disabled"),
                     AgentJoints(np.array([0.5, 1, 1, 1, 0, 0.3, 1, 1]), "056_partial"),
                     AgentJoints(np.array([0.8, 0.9, 0.6, 0.8, 0.5, 1, 0.7, 1]), "123457_partial"),
                     AgentJoints(np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1]), "1234567_partial")]
    """
    agents_joints = [AgentJoints(np.array([1, 1, 1, 1, 1, 1, 1, 1]), "non_crippled_test"),
                     AgentJoints(np.array([1, 1, 1, 0, 1, 1, 1, 1]), "fourth_disabled_test"),
                     AgentJoints(np.array([1, 1, 1, 1, 1, 1, 0, 1]), "seventh_disabled_test"),
                     AgentJoints(np.array([1, 0.2, 1, 1, 1, 1, 1, 1]), "2_partial_test"),
                     AgentJoints(np.array([1, 1, 1, 1, 0.4, 1, 1, 1]), "5_partial_test"),
                     AgentJoints(np.array([1, 0.8, 0.1, 1, 0.5, 0.2, 1, 1]), "2356_partial_test"),
                     AgentJoints(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), "all_partial_test")]

    for agent_joints in agents_joints:
        agent = ReachEnvJointVelCtrl(render=False, crippled=agent_joints.joints, nsubsteps=10)
        print("create data collector for ", agent_joints.name)
        data_collector = CollectRandomData(num_rollouts_train,
                                           num_rollouts_val,
                                           steps_per_rollout_train,
                                           steps_per_rollout_val,
                                           agent_joints.name, # disabled third joint
                                           agent,
                                           reach_env_data_path + "/" + agent_joints.name)
        data_collector.perform_data_collection()

