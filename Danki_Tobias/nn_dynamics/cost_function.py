import numpy as np

class cost_fn:
    def __init__(self, env, num_simulated_paths = 10, horizon = 5):
        self.env = env
        self.num_simulated_paths = num_simulated_paths
        self.horizon = horizon


    def trajectory_cost(self, states, sampled_actions, nstates):
        # states has shape (horizon, paths, 7), nstates has shape (horizon-1, paths, 7)
        # assert (states.shape == (horizon, num_simulated_paths))
        costs = np.zeros(self.num_simulated_paths)

        envs = [self.env * self.num_simulated_paths]
        current_state = self.env.sim.get_state()
        current_state.qpos[:7] = states[0, 0]
        for env in envs:
            env.sim.setstate(current_state)



