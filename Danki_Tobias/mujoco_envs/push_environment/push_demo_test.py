import numpy as np
from unittest import TestCase
from Danki_Tobias.mujoco_envs.push_environment.push_demo import PushEnvJointVelCtrl


class TestPushEnvMocapCtrl(TestCase):
    def setUp(self) -> None:
        self.env = PushEnvJointVelCtrl(render=True)

    def testRndActions(self):
        for i in range(10000):
            action = self.env.action_space.sample()
            obs, reward, done, _ = self.env.step(action)
            if done:
                self.env.reset()
