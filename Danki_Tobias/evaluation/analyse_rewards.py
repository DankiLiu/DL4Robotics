import pandas as pd
import matplotlib.pyplot as plt

data_type_options = ['position', 'position_deltas', 'position_and_velocity', 'position_and_velocity_deltas']
train_on_options = ['non_crippled', 'multiple_envs']
algorithms = ['normal', 'meta', 'online_adaptation']

data_type = data_type_options[3]
train_on = train_on_options[0]
algorithm = algorithms[0]

file_name = f'../data/{data_type}/on_policy/trained_on_{train_on}/{algorithm}/rewards.csv'
rewards = pd.read_csv(file_name)
rewards['mean_reward'] = rewards.mean(axis=1)
rewards['mean_reward'].plot()

plt.xlabel("Epochs")
plt.ylabel("Average on policy reward")
plt.savefig("../results/TrainingReward", bbox_inches='tight')
plt.show()
