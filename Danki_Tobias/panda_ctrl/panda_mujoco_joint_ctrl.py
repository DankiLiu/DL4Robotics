import gym
import numpy as np

from gym_framework.panda_ctrl.panda_mujoco_base_ctrl import PandaBase


class PandaJointControl(PandaBase):
    """
    Control the panda by specifying joint positions (control=position) / velocities (control=velocity).
    """

    def __init__(self,
                 render=True):
        super().__init__(render=render)

        self._action_dimension = 8  # 7 x actuator joint; 1 gripper width

    def apply_action(self, action):
        assert len(action) == self.action_dimension, ("Error, wrong action dimension. Expected: " +
                                                      str(self.action_dimension) + ". Got:" + str(len(action)))
        gripper_ctrl = action[7]
        joint_action = action[:7]
        self.sim.data.ctrl[:] = np.concatenate(([gripper_ctrl, gripper_ctrl], joint_action))

        # Apply gravity compensation
        self.sim.data.qfrc_applied[self.joint_indices] = self.sim.data.qfrc_bias[self.joint_indices]

        # Forward the simulation
        self.sim.step()

        # Render the scene
        if self.render and self.viewer is not None:
            self.viewer.render()

    @property
    def action_dimension(self):
        return self._action_dimension

    @property
    def state(self):
        current_joint_position = [self.sim.data.get_joint_qpos(j_name) for j_name in self.joint_names]
        current_joint_velocity = [self.sim.data.get_joint_qvel(j_name) for j_name in self.joint_names]

        current_finger_position = [self.sim.data.get_joint_qpos(j_name) for j_name in self.gripper_names]
        current_finger_velocity = [self.sim.data.get_joint_qvel(j_name) for j_name in self.gripper_names]

        tcp_pos = self.sim.data.get_body_xpos('tcp').copy()
        tcp_quat = self.sim.data.get_body_xquat('tcp').copy()
        tcp_velp = self.sim.data.get_body_xvelp('tcp').copy()
        tcp_velr = self.sim.data.get_body_xvelr('tcp').copy()

        return np.concatenate([current_joint_position,
                               current_joint_velocity,
                               current_finger_position,
                               current_finger_velocity,
                               tcp_pos,
                               tcp_quat,
                               tcp_velp,
                               tcp_velr])

    def get_workspace(self):
        workspace_low = np.array([self.sim.data.get_site_xpos('x_constrain_low')[0],
                                  self.sim.data.get_site_xpos('y_constrain_low')[1],
                                  self.sim.data.get_site_xpos('z_constrain_low')[2]])

        workspace_high = np.array([self.sim.data.get_site_xpos('x_constrain_high')[0],
                                   self.sim.data.get_site_xpos('y_constrain_high')[1],
                                   self.sim.data.get_site_xpos('z_constrain_high')[2]])

        return gym.spaces.Box(low=workspace_low, high=workspace_high)


class PandaJointVelControlCrippled(PandaJointControl):
    def __init__(self, render=True, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1])):
        self.crippled = crippled
        super().__init__(render=render)

    def env_setup(self, sim, viewer):
        super().env_setup(sim, viewer)
        self.workspace = self.get_workspace()

    @property
    def action_space(self):
        # upper and lower bound for each joint velocity
        low = [-87, -87, -87, -87, -12, -12, -12, -1]
        high = [87, 87, 87, 87, 12, 12, 12, 1]
        # low = [-100, -100, -100, -100, -100, -100, -100, -1]
        # high = [100, 100, 100, 100, 100, 100, 100, 1]

        low = low * self.crippled
        high = high * self.crippled

        # Setting dtype=np.int enables a better sampling of actions, but the gripper can not be used anymore
        action_space = gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.int)

        return action_space

    @property
    def ctrl_name(self):
        return 'velocity'
