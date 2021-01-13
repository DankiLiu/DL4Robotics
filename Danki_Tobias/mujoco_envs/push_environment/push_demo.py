import numpy as np

from gym_framework.mujoco_envs.mujoco_env import MujocoEnv
from gym_framework.mujoco_objects import env_objects
from Danki_Tobias.panda_ctrl.panda_mujoco_joint_ctrl import PandaJointVelControlCrippled
from gym_framework.utils.helper import obj_distance


class PushEnvBase(MujocoEnv):
    def __init__(self, agent, render=True, max_steps=2000,
                 nsubsteps=1, dt=2e-3, random_env=True, workspace_size='medium', target_min_dist=0.02):
        self._boxes = env_objects.Boxes()

        super().__init__(agent=agent, obj_list=[self._boxes], max_steps=max_steps, render=render,
                         nsubsteps=nsubsteps, dt=dt, random_env=random_env, workspace_size=workspace_size)
        #min initial distance between 2 boxes, currently not used
        self.min_dist = 0.2
        self.target_min_dist = target_min_dist

    @property
    def environment_observations(self):
        red_box_pos = self.sim.data.get_body_xpos(self._boxes.red_ID).copy()
        blue_box_pos = self.sim.data.get_body_xpos(self._boxes.blue_ID).copy()
        print("red_box_pos: " , red_box_pos)
        print("blue_box_pos: ", blue_box_pos)
        box_distance, _ = obj_distance(self.sim, self._boxes.red_ID, self._boxes.blue_ID)
        return np.concatenate([red_box_pos, blue_box_pos, [box_distance]])

    def callback_randomize_env(self):
        '''
        Todo: change the sample function
        '''
        red_box_pos = self.agent.workspace.sample()
        blue_box_pos = self.agent.workspace.sample()

        box_distance, _ = obj_distance(self.sim, self._boxes.red_ID, self._boxes.blue_ID)
        """
        while box_distance <= 0.1:
            red_box_pos = self.agent.workspace.sample()
            blue_box_pos = self.agent.workspace.sample()
        """
        red_q_addr = self.sim.model.get_joint_qpos_addr('red_box:joint')
        blue_q_addr = self.sim.model.get_joint_qpos_addr('blue_box:joint')
        self.qpos[red_q_addr[0]:red_q_addr[0] + 2] = [red_box_pos[0], red_box_pos[1]]
        self.qpos[blue_q_addr[0]:blue_q_addr[0] + 2] = [blue_box_pos[0], blue_box_pos[1]]

    def _reward(self):
        box_distance, _ = obj_distance(self.sim, self._boxes.red_ID, self._boxes.blue_ID)
        reward = np.double(-box_distance)

        if box_distance <= self.target_min_dist:
            reward = np.double(10)
        return reward

    def _termination(self):
        # calculate the distance between two boxes
        box_distance, _ = obj_distance(self.sim, self._boxes.red_ID, self._boxes.blue_ID)

        if box_distance <= self.target_min_dist:
            # terminate if end effector is close enough
            self.terminated = True
        return super()._termination()

    def step(self, action):
        gripper_gain = 1.
        return super().step(np.concatenate([action[:-1], [gripper_gain]]))


class PushEnvJointVelCtrl(PushEnvBase):
    def __init__(self, render=True, crippled=np.ones(8), nsubsteps=1):
        crippled = np.array(crippled)
        agent = PandaJointVelControlCrippled(render, crippled=crippled)
        super().__init__(agent, render, nsubsteps=nsubsteps)