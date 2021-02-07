import json
import logging
import pathlib
# get parameters from json file

current_path = pathlib.Path().absolute()
reach_json_path = str(current_path.parent) + "/json_files/reach.json"

with open(reach_json_path) as jf:
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

def get_model_id():
    try:
        rl_data_collection = data["model_id"]["rl_data_collection"]
        return rl_data_collection
    except:
        logging.info("Load RL collection number failed")

def dyn_model_training_params():
    try:
        number_of_random_samples = ["dyn_model_training"]["number_of_random_samples"]
        iterations = data["dyn_model_training"]["iterations"]
        training_epochs = data["dyn_model_training"]["training_epochs"]
        new_model = data["dyn_model_training"]["new_model"]
        return number_of_random_samples, iterations, training_epochs, new_model
    except:
        logging.info("Load RL collection number failed")