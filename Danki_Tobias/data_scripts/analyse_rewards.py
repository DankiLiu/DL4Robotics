import pandas as pd
import matplotlib.pyplot as plt



experiment = 'exp2'
model_type = 'meta'
model_id = 0


file_name = f'../data/on_policy/{experiment}/{model_type}/model{model_id}/rewards.csv'
rewards = pd.read_csv(file_name)
rewards['mean_reward'] = rewards.mean(axis=1)
rewards['mean_reward'].plot()
plt.show()