from Danki_Tobias.rl_agents.dynamicsModelBase import *

tf.keras.backend.set_floatx('float64')


class NNDynamicsModel(BaseDynamicsModel):
    def __init__(self, env, normalization, model, batch_size=512):
        super().__init__(env, normalization, model, batch_size)

    @classmethod
    def new_model(cls, env, n_layers, size, activation, output_activation, normalization, batch_size, learning_rate):
        ob_dim = 14
        ac_dim = 7
        model = build_and_compile_model(ob_dim, n_layers, size, activation, output_activation, learning_rate)

        return cls(env, normalization, model, batch_size)

    def fit(self, states, actions, deltas, N_EPOCHS):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and
        fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        ### normalize
        states_normalized = self.normalize(states)
        actions_normalized = self.normalize(actions)
        deltas_normalized = self.normalize(deltas)

        input = states_normalized.join(actions_normalized, how='inner')
        print(deltas_normalized)
        print(input)

        self.model.fit(x=input, y=deltas_normalized, batch_size=self.batch_size, epochs=N_EPOCHS)

    def normalize_and_adapt(self, states, actions, deltas):
        ### normalize
        states_normalized = self.normalize(pd.DataFrame(states, columns=state_columns))
        actions_normalized = self.normalize(pd.DataFrame(actions, columns=action_columns))
        deltas_normalized = self.normalize(pd.DataFrame(deltas, columns=delta_columns))
        input = states_normalized.join(actions_normalized, how='inner')
        self.model.fit(input, deltas_normalized, batch_size=1, verbose=0)

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