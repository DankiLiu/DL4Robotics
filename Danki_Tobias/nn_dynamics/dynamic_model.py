import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
                 normalization, batch_size, learning_rate
                 ):
        self.normalization = normalization
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
    def fit(self, states, actions, deltas, N_EPOCHS=50):
        states_normalized = self.normalize(states)
        actions_normalized = self.normalize(actions)
        deltas_normalized = self.normalize(deltas)

        input =  states_normalized.join(actions_normalized)