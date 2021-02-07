import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam

from Danki_Tobias.column_names import *
from Danki_Tobias.helper.get_parameters import *
from Danki_Tobias.rl_agents.dynamicsModelBase import *

tf.keras.backend.set_floatx('float64')

n_layers, layer_size, batch_size, n_epochs, M, K = metaRL_dyn_model_params()

# function to build a feedforward neural network
def build_and_compile_model(output_size,
                            n_layers=n_layers,
                            size=layer_size,
                            activation=tf.tanh,
                            output_activation=None
                            ):
    model = keras.Sequential()
    for _ in range(n_layers):
        model.add(layers.Dense(size, activation=activation))
    model.add(layers.Dense(output_size, activation=output_activation))

    model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
    return model

"""
class MetaRLDynamicsModel:
    def __init__(self, env, normalization, model, batch_size=batch_size):
        self.normalization = normalization
        self.batch_size = batch_size
        # ob_dim = env.observation_dim.shape[0]  # local variables of init just for convinience
        # ac_dim = env.action_space.shape[0]

        self.model = model
"""

class MetaRLDynamicsModel(BaseDynamicsModel):
    def __init__(self, env, normalization, model, batch_size=512):
        super().__init__(env, normalization, model, batch_size)


        self.model(np.zeros((1, 21)))
        self.meta_model_weights = self.model.get_weights()
        self.meta_opt = Adam(lr=0.001)


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

    def fit(self, states, actions, deltas, N_EPOCHS=n_epochs, M=M, K=K):
        number_of_trajectories = len(states) / (M + K)
        assert number_of_trajectories.is_integer()
        number_of_trajectories = int(number_of_trajectories)

        ### normalize
        states_normalized = self.normalize(states)
        actions_normalized = self.normalize(actions)
        deltas_normalized = self.normalize(deltas)

        input = states_normalized.join(actions_normalized, how='inner')

        for epoch in range(N_EPOCHS):
            with tf.GradientTape() as tape:
                total_loss = []
                for trajectory in range(number_of_trajectories):
                    index_of_first_elem = trajectory * (M + K)
                    # indices of current trajectory for task specific update
                    m_index = np.arange(index_of_first_elem, index_of_first_elem + M)
                    # indices of current trajectory for task specific evaluation
                    k_index = np.arange(index_of_first_elem + M, index_of_first_elem + (M + K))

                    # adapt meta model with a single trajectory
                    self.adapt(input.iloc[m_index], deltas_normalized.iloc[m_index], M)
                    # calculate Loss of adapted model
                    prediction = self.model(input.iloc[k_index].values)
                    label = deltas_normalized.iloc[k_index].values
                    loss = losses.mean_squared_error(label, prediction)
                    total_loss.append(loss)

                total_loss = tf.math.add_n(total_loss)
                total_loss = tf.math.divide(total_loss, number_of_trajectories)
            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            self.model.set_weights(self.meta_model_weights)
            grads = tape.gradient(total_loss, self.model.trainable_weights)
            self.meta_opt.apply_gradients(zip(grads, self.model.trainable_variables))
            self.meta_model_weights = self.model.get_weights()

    # task specific update step
    def adapt(self, x, y, m):
        # reset model to meta model
        self.model.set_weights(self.meta_model_weights)
        self.model.fit(x, y, batch_size=m, verbose=0)
        return

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
        predictions.columns = state_columns

        states.reset_index(drop=True, inplace=True)
        return predictions + states
