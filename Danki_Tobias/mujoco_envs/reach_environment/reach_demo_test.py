import numpy as np
from unittest import TestCase
from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl


class TestReachEnvMocapCtrl(TestCase):
    def setUp(self) -> None:
        self.env = ReachEnvJointVelCtrl(render=True, crippled=np.array([1, 1, 1, 1, 0, 0, 1, 1]))

    def testRndActions(self):
        for i in range(10000):
            action = self.env.action_space.sample()
            obs, reward, done, _ = self.env.step(action)
            if done:
                self.env.reset()
