import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

from Danki_Tobias.column_names import *

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


class NNDynamicsModel:
    def __init__(self, env, normalization, model, batch_size=512):
        self.normalization = normalization
        self.batch_size = batch_size
        # ob_dim = env.observation_dim.shape[0]  # local variables of init just for convinience
        # ac_dim = env.action_space.shape[0]

        self.model = model

    @classmethod
    def new_model(cls, env, n_layers, size, activation, output_activation, normalization, batch_size, learning_rate):
        ob_dim = 14
        ac_dim = 7
        model = build_and_compile_model(ob_dim, n_layers, size, activation, output_activation)

        return cls(env, normalization, model, batch_size)

    def normalize(self, data):
        normalization_values = self.normalization.loc[data.columns]
        return (data - normalization_values['mean']) / (normalization_values['std'] + 1e-10)

    def denormalize(self, data):
        normalization_values = self.normalization.loc[data.columns]
        return data * (normalization_values['std'] + 1e-10) + normalization_values['mean']

    def fit(self, states, actions, deltas):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and
        fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        ### normalize
        states_normalized = self.normalize(states)
        deltas_normalized = self.normalize(deltas)
        actions_normalized = self.normalize(actions)

        # combine state and action to input
        input = states_normalized.join(actions_normalized)

        N_EPOCHS = 50
        self.model.fit(x=input, y=deltas_normalized, batch_size=self.batch_size, epochs=N_EPOCHS)

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions
        and return the (unnormalized) next states as predicted by using the model """
        ### normalize
        states_normalized = self.normalize(states)
        actions_normalized = self.normalize(actions)

        # combine state and action to input
        input = np.concatenate((states_normalized, actions_normalized), axis=1)

        predictions = pd.DataFrame(self.model.predict(input), columns=delta_columns)
        predictions = self.denormalize(predictions)

        states.columns = delta_columns
        states.reset_index(drop=True, inplace=True)

        return predictions + states
