import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os

from Danki_Tobias.data_scripts.data_reader import *
from Danki_Tobias.mujoco_envs.reach_environment.reach_demo import ReachEnvJointVelCtrl
from dynamicsModel import NNDynamicsModel
from controller import MPCcontroller, sample

random_data_file = 'random_samples_2020-12-16_21-18'


def load_model(env):
    normalization = load_normalization_variables(random_data_file)

    model = keras.models.load_model(filepath=f'../models/iteration_{2}.hdf5')
    dyn_model = NNDynamicsModel(env=env, normalization=normalization, model=model)
    return dyn_model


if __name__ == "__main__":
    controller_env = ReachEnvJointVelCtrl(render=False, nsubsteps=10, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))
    env = ReachEnvJointVelCtrl(render=True, nsubsteps=10, crippled=np.array([1, 1, 1, 1, 1, 1, 1, 1]))

    dyn_model = load_model(env)

    # init the mpc controller
    mpc_controller = MPCcontroller(env=controller_env, dyn_model=dyn_model, )
    sample(env, mpc_controller, horizon=33, num_paths=2)
