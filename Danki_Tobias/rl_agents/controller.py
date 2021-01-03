import numpy as np
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

        self.box_position = self.env.environment_observations[0:3]

    def cost(self, next_state):
        print(next_state)
        position_state = next_state[:7]
        print(position_state)
        # set the state of the simulation to the state predicted by the model
        simulation_state = self.env.sim.get_state()
        simulation_state.qpos[:position_state.shape[0]] = position_state
        self.env.sim.set_state(simulation_state)
        self.env.sim.forward()

        # get the reward for this state
        reward = self.env._reward()
        return -reward

    def get_action(self, state):
        """ Note: be careful to batch your simulations through the model for speed """
        print(state)
        print(state.shape)
        sampled_actions = np.array(
            [[self.env.action_space.sample()[:7] for j in range(self.num_simulated_paths)] for i in
             range(self.horizon)])
        print(f'actions shape: {sampled_actions.shape}')
        states = [np.array([state] * self.num_simulated_paths)]
        print(f'State shape: {states[0].shape}')
        next_states = []
        for i in range(self.horizon):
            print("STATE")
            print(states[-1].shape)
            print('ACTION')
            print(sampled_actions[i, :].shape)
            test = self.dyn_model.predict(states[-1], sampled_actions[i, :])
            print(test)

            next_states.append(test)

            print(states[-1].shape)
            print(sampled_actions[i, :].shape)

            next_states.append(self.dyn_model.predict(states[-1], sampled_actions[i, :]))
            if i < self.horizon: states.append(next_states[-1])
            
        trajectory_cost = 0
        for i in range(len(sampled_actions)):
            trajectory_cost += self.cost(next_states[i])

        return sampled_actions[0][np.argmin(trajectory_cost)], min(trajectory_cost)
