import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

tf.keras.backend.set_floatx('float64')


# Predefined function to build a feedforward neural network
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

    model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy', ''])
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

    def fit(self, state, action, delta):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and
        fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        ### normalize
        state = normalize(state, self.normalization['observations'][0], self.normalization['observations'][1])
        delta = normalize(delta, self.normalization['delta'][0], self.normalization['delta'][1])
        action = normalize(action, self.normalization['actions'][0], self.normalization['actions'][1])

        # combine state and action to input
        input = state + action

        N_EPOCHS = 50
        self.model.fit(x=input, y=delta, batch_size=self.batch_size, epochs=N_EPOCHS)

        train_count = len(state)

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions
        and return the (unnormalized) next states as predicted by using the model """
        obs = normalize(states, self.normalization['observations'][0], self.normalization['observations'][1])
        # delta = normalize(delta,normalization['delta'])
        acs = normalize(actions, self.normalization['actions'][0], self.normalization['actions'][1])
        done = False
        start = 0;
        end = 0
        test_count = len(states)
        # print(test_count)
        prediction = self.sess.run(self.delta_prediction, feed_dict={self.sy_ob: obs, self.sy_ac: acs})

        return denormalize(prediction, self.normalization['delta'][0], self.normalization['delta'][1]) + states
