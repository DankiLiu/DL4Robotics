import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

tf.keras.backend.set_floatx('float64')


# function to build a feedforward neural network
def build_and_compile_model(output_size,
                            n_layers=2,
                            size=500,
                            activation=tf.tanh,
                            output_activation=None
                            ):
    model = keras.Sequential()
    for _ in range(n_layers):
        model.add(layers.Dense(size, activation=activation))
    model.add(layers.Dense(output_size, activation=output_activation))

    model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
    return model


def normalize(data, mean, std):
    return (data - mean) / (std + 1e-10)


def denormalize(data, mean, std):
    return data * (std + 1e-10) + mean


class NNDynamicsModel:
    def __init__(self,
                 env,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 ):
        self.normalization = normalization
        self.iterations = iterations
        self.batch_size = batch_size
        # ob_dim = env.observation_dim.shape[0]  # local variables of init just for convinience
        # ac_dim = env.action_space.shape[0]

        ob_dim = 14
        ac_dim = 7

        self.model = build_and_compile_model(ob_dim, n_layers, size, activation, output_activation)

    def fit(self, states, actions, deltas):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and
        fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        ### normalize
        states_normalized = normalize(states, self.normalization['observations'][0],
                                      self.normalization['observations'][1])
        deltas_normalized = normalize(deltas, self.normalization['delta'][0], self.normalization['delta'][1])
        actions_normalized = normalize(actions, self.normalization['actions'][0], self.normalization['actions'][1])

        # combine state and action to input
        input = states_normalized.join(actions_normalized)

        N_EPOCHS = 50
        self.model.fit(x=input, y=deltas_normalized, batch_size=self.batch_size, epochs=N_EPOCHS)

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions
        and return the (unnormalized) next states as predicted by using the model """
        ### normalize
        states_normalized = np.array([normalize(state, self.normalization['observations'][0],
                                                self.normalization['observations'][1]) for state in states])
        actions_normalized = np.array([normalize(action, self.normalization['actions'][0],
                                                 self.normalization['actions'][1]) for action in actions])

        # combine state and action to input
        input = np.concatenate((states_normalized, actions_normalized), axis=1)

        column_names = ['state_delta_position_0', 'state_delta_position_1', 'state_delta_position_2',
                        'state_delta_position_3',
                        'state_delta_position_4', 'state_delta_position_5', 'state_delta_position_6',
                        'state_delta_velocity_0', 'state_delta_velocity_1', 'state_delta_velocity_2',
                        'state_delta_velocity_3',
                        'state_delta_velocity_4', 'state_delta_velocity_5', 'state_delta_velocity_6']

        predictions = pd.DataFrame(self.model.predict(input), columns=column_names)
        predictions = denormalize(predictions, self.normalization['delta'][0], self.normalization['delta'][1])

        # states.columns = column_names,        TODO if states is a pd Dataframe line makes sense, if it's a numpy array it doesn't
        return predictions + states
