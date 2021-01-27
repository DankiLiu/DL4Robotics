import pandas as pd
import matplotlib.pyplot as plt


def read_reward_file(number):
    df = pd.read_csv(f"../data/reach_env/samples_{number}_rewards.csv", header=None)
    return df


rewards = read_reward_file(1)
rewards['mean_reward'] = rewards.mean(axis=1)
rewards['mean_reward'].plot()
plt.show()