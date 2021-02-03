import json
import logging
#get parameters from json file

with open("json_files/reach.json") as jf:
    data = json.load(jf)

def data_collection_params():
    try:
        num_rollouts_train = data["data_collection"]["num_rollouts_train"]
        num_rollouts_val = data["data_collection"]["num_rollouts_val"]
        steps_per_rollout_train = data["steps"]["steps_per_rollout_train"]
        steps_per_rollout_val = data["steps"]["steps_per_rollout_val"]

        return num_rollouts_train, num_rollouts_val, steps_per_rollout_train, steps_per_rollout_val
    except:
        logging.info("Load data collection parameters error.")

def MPCcontroller_params():
    try:
        num_simulated_paths = data["MPCcontroller"]["num_simulated_paths"]
        horizon = data["MPCcontroller"]["horizon"]
        num_control_samples = data["MPCcontroller"]["num_control_samples"]

        return num_simulated_paths, horizon, num_control_samples
    except:
        logging.info("Load controller parameters error.")


def on_policy_sampling_params():
        try:
            num_paths = data["on_policy_sampling"]["num_paths"]
            on_policy_horizon = data["on_policy_sampling"]["on_policy_horizon"]


            return num_paths, on_policy_horizon
        except:
            logging.info("Load on policy sampling parameters error.")

def dyn_model_params():
    try:
        n_layers = data["dyn_model"]["n_layers"]
        layer_size = data["dyn_model"]["layer_size"]
        batch_size = data["dyn_model"]["batch_size"]
        n_epochs = data["dyn_model"]["n_epochs"]
        learning_rate = data["dyn_model"]["learning_rate"]
        return n_layers, layer_size, batch_size, n_epochs, learning_rate
    except:
        logging.info("Load on policy sampling parameters error.")


def metaRL_dyn_model_params():
    try:
        n_layers = data["metaRL_dyn_model"]["n_layers"]
        layer_size = data["metaRL_dyn_model"]["layer_size"]
        batch_size = data["metaRL_dyn_model"]["batch_size"]
        n_epochs = data["metaRL_dyn_model"]["n_epochs"]
        M = data["metaRL_dyn_model"]["M"]
        K = data["metaRL_dyn_model"]["K"]

        return n_layers, layer_size, batch_size, n_epochs, M, K
    except:
        logging.info("Load on metaRL dynamic model parameters error.")

def data_collection_num():
    try:
        rl_data_collection = data["data_collection_num"]["rl_data_collection"]
        return rl_data_collection
    except:
        logging.info("Load RL collection number failed")

def dyn_model_training_params():
    try:
        iterations = data["dyn_model_training"]["iterations"]
        training_epochs = data["dyn_model_training"]["training_epochs"]
        return iterations, training_epochs
    except:
        logging.info("Load RL collection number failed")