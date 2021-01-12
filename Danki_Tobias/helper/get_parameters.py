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
        logging.info("Load data collection parameters error")

