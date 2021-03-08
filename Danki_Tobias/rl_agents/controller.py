import numpy as np
import pandas as pd

from Danki_Tobias.column_names import *
from Danki_Tobias.helper.get_parameters import *


def sample(env,
           controller,
           num_paths=10,
           horizon=500,
           finish_when_done=True,
           with_adaptation=False,
           predicts_state=True,
           states_only=False):
    """
        Write a sampler function which takes in an environment, a controller (either random or the MPC controller),
        and returns rollouts by running on the env.
        Each path can have elements for observations, next_observations, rewards, returns, actions, etc.
    """
    state_length = 14
    if states_only:
        state_length = 7

    paths = []
    rewards = []
    costs = []
    print("num_sum_path", num_paths)
    for i in range(num_paths):
        print("path :", i)
        states = list()
        actions = list()
        next_states = list()
        states.append(env.reset()[0:state_length])
        total_reward = 0
        total_cost = 0
        for j in range(horizon):
            if j % 100 == 0:
                print(j)

            if with_adaptation and j > 0:
                # adapt meta model with data of last 32 steps trajectory
                if predicts_state:
                    labels = np.array(next_states[max(0, j - 32):j])
                else:  # predicts delta
                    labels = np.array(next_states[max(0, j - 32):j]) - np.array(states[max(0, j - 32):j])
                controller.dyn_model.normalize_and_adapt(states=np.array(states[max(0, j - 32):j]),
                                                         actions=np.array(actions[max(0, j - 32):j]), labels=labels)

            act, cost = controller.get_action(states[j], env.sim.get_state())
            actions.append(act)
            obs, r, done, _ = env.step(np.append(actions[j], 0.4))  # append value for gripper

            # extract relevant state information
            next_states.append(obs[0:state_length])
            total_reward += r
            total_cost += cost

            if done and finish_when_done:
                print('Done')
                break

            if j != horizon - 1:
                states.append(next_states[j])

        path = {'observations': np.array(states),
                'actions': np.array(actions),
                'next_observations': np.array(next_states)
                }
        paths.append(path)
        rewards.append(total_reward)
        costs.append(total_cost)

    return paths, rewards, costs


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
                 horizon=1,
                 cost_fn=None,
                 num_simulated_paths=50,
                 states_only=False):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths
        self.states_only = states_only

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
            if self.states_only:
                states_df = pd.DataFrame(states[-1], columns=state_columns_position_only)
            else:
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
