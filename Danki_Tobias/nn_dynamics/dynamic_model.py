import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

from Danki_Tobias.column_names import *

def build_mlp(output_size,
              n_layers=2,
              size=200,
              activation=tf.tanh,
              output_activation=None):
    model = keras.Sequential
    for _ in range(n_layers):
        model.add(layers.Dense(size, activation=activation))
    model.add(layers.Dense(output_size, activation=output_activation))

    """
    model.compile(optimizer='rmsprop',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    """
    return model


class NNDynamicsModel:
    def __init__(self,
                 env,
                 n_layers, size, activation, output_activation,
                 normalization_data, batch_size, learning_rate
                 ):
        self.normalization_data = normalization_data
        self.batch_size = batch_size

        ob_dim = 14
        ac_dim = 7
        model = build_mlp(output_size=ob_dim,
                          n_layers=n_layers,
                          size=size,
                          activation=activation,
                          output_activation=output_activation)

        loss = tf.keras.losses.mean_squared_error()
        self.dyna_update_op = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model = model.compile(optimizer=self.dyna_update_op, loss=loss, metrics=['accuracy'])

    """
    ob_dim = env.observation_dim.shape[0] -> positions and velocities of all 7 joints
    ac_dim = env.action_space.shape[0] -> action space of 7 joints
    """

    def normalize(self, data):
        normalization_values = self.normalization_data.loc[data.columns]
        # syntax -> normalization_values['mean']
        return data - normalization_values['mean'] / normalization_values['std'] + 1e-10

    def denormalize(self, data):
        normalization_values = self.normalization_data.loc[data.columns]
        return data * (normalization_values['std'] + 1e-10) + normalization_values['mean']

    def fit(self, states, actions, deltas, N_EPOCHS=50):
        states_normalized = self.normalize(states)
        actions_normalized = self.normalize(actions)
        deltas_normalized = self.normalize(deltas)

        input_data = states_normalized.join(actions_normalized)
        self.model.fit(x=input_data, y=deltas_normalized, batch_size=self.batch_size, epochs=N_EPOCHS)

    def predict(self, states, actions):
        states_normalized = self.normalize(states)
        actions_normalized = self.normalize(actions)

        # combine states and actions to input
        input_data = np.concatenate(states_normalized, actions_normalized)

        predictions = pd.DataFrame(self.model.predict(input_data), columns=state_columns)
        predictions = self.denormalize(predictions)

        states.reset_index(drop=True, inplace=True)

        return predictions + states
