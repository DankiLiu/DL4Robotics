from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.backend.set_floatx('float64')


# function to build a feedforward neural network
def build_and_compile_model(output_size,
                            n_layers=2,
                            size=500,
                            activation=tf.tanh,
                            output_activation=None,
                            learning_rate=0.001
                            ):
    model = keras.Sequential()
    for _ in range(n_layers):
        model.add(layers.Dense(size, activation=activation))

    model.add(layers.Dense(output_size, activation=output_activation))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    return model


class BaseDynamicsModel(ABC):
    def __init__(self, env, normalization, model, batch_size=512):
        self.normalization = normalization
        self.batch_size = batch_size

        self.model = model

    @classmethod
    def new_model(cls, env, n_layers, size, activation, output_activation, normalization, batch_size, learning_rate):
        ob_dim = 14
        ac_dim = 7
        model = build_and_compile_model(ob_dim, n_layers, size, activation, output_activation, learning_rate)

        return cls(env, normalization, model, batch_size)

    def normalize(self, data):
        normalization_values = self.normalization.loc[data.columns]
        return (data - normalization_values['mean']) / (normalization_values['std'] + 1e-10)

    def denormalize(self, data):
        normalization_values = self.normalization.loc[data.columns]
        return data * (normalization_values['std'] + 1e-10) + normalization_values['mean']

    @abstractmethod
    def fit(self, states, actions, deltas, N_EPOCHS=50):
        raise NotImplementedError

    @abstractmethod
    def predict(self, states, actions):
        raise NotImplementedError
