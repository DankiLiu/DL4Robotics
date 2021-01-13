import numpy as np

def cost_func(state, action, next_state):
    """
    Calculate costs for x joints. (with state of position)
    state: current states (positions) of joints. (x, 7) array.
    action:
    next_state:
    """
    costs = np.zeros(state.shape[0])

    for index, state in enumerate(next_state):
        pass
