import numpy as np
from unittest import TestCase
from gym_framework.mujoco_envs.Danki_Tobias.reach_demo import ReachEnvJointVelCtrl
from gym_framework.mujoco_envs.Danki_Tobias.reach_demo import ReachEnvJointTorqueCtrl
from gym_framework.mujoco_envs.Danki_Tobias.reach_demo import ReachEnvMocapCtrl
from gym_framework.mujoco_envs.Danki_Tobias.reach_demo import ReachEnvJointPosCtrl

class TestReachEnvMocapCtrl(TestCase):
    def setUp(self) -> None:
        self.env = ReachEnvJointVelCtrl(render=True, crippled=np.array([1, 1, 1, 1, 0, 0, 1, 1]))

    def testRndActions(self):
        for i in range(10000):
            action = self.env.action_space.sample()
            obs, reward, done, _ = self.env.step(action)
            print(done)
            if done:
                self.env.reset()
