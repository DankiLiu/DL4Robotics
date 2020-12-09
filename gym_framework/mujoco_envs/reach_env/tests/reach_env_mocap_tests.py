import numpy as np
from unittest import TestCase
from gym_framework.mujoco_envs.reach_env.reach_env import ReachEnvMocapCtrl
from gym_framework.mujoco_envs.reach_env.reach_env import ReachEnvJointVelCtrl
from gym_framework.mujoco_envs.reach_env.reach_env import ReachEnvJointPosCtrl
from gym_framework.mujoco_envs.reach_env.reach_env import ReachEnvJointTorqueCtrl


class TestReachEnvMocapCtrl(TestCase):
    def setUp(self) -> None:
        self.env = ReachEnvMocapCtrl(render=True)
        #self.env = ReachEnvJointPosCtrl(render=True)

    def testRndActions(self):
        for i in range(1000):
            action = self.env.action_space.sample()
            obs, reward, done, _ = self.env.step(action)
            if done:
                self.env.reset()
