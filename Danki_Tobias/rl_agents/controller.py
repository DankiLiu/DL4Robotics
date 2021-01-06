import numpy as np
import pandas as pd

from Danki_Tobias.column_names import *
# from cost_functions import trajectory_cost_fn
import time


class Controller():
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        self.ac = env.action_space

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Your code should randomly sample an action uniformly from the action space """
        return self.ac.sample(), 0


class MPCcontroller(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """

    def __init__(self,
                 env,
                 dyn_model,
                 horizon=5,
                 cost_fn=None,
                 num_simulated_paths=10,
                 ):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths

    def cost(self, next_state):
        position_state = next_state[:, :7]
        costs = np.zeros(self.num_simulated_paths)
        # set the state of the simulation to the states predicted by the model
        for index, position in enumerate(position_state):
            simulation_state = self.env.sim.get_state()
            simulation_state.qpos[:position.shape[0]] = position
            self.env.sim.set_state(simulation_state)
            self.env.sim.forward()
            # get the reward for this state
            costs[index] = -self.env._reward
        return costs

    def get_action(self, state, simulation_state):
        # sample random actions
        sampled_actions = np.array(
            [[self.env.action_space.sample()[:7] for j in range(self.num_simulated_paths)] for i in
             range(self.horizon)])
        states = [np.array([state] * self.num_simulated_paths)]
        next_states = []

        # predict the next state
        for i in range(self.horizon):
            states_df = pd.DataFrame(states[-1], columns=state_columns)
            actions_df = pd.DataFrame(sampled_actions[i, :], columns=action_columns)
            next_states.append(self.dyn_model.predict(states_df, actions_df).values)

            if i < self.horizon:
                states.append(next_states[-1])

        # calculate cost of each path
        self.env.sim.set_state(simulation_state)
        trajectory_costs = np.zeros(self.num_simulated_paths)
        for i in range(len(sampled_actions)):
            trajectory_costs += self.cost(next_states[i])

        return sampled_actions[0][np.argmin(trajectory_costs)], min(trajectory_costs)
