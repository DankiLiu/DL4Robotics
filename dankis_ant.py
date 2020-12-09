import gym
import mujoco_py

from stable_baselines3 import PPO

env = gym.envs.make("Ant-v2")

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=500000)

obs = env.reset()
for i in range(20000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()