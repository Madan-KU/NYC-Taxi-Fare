import os
import json
import yaml
import logging
import datetime

from modules.logger_configurator import configure_logger

configure_logger()

def read_yaml_config(file_path):
    """Read and return the configuration from a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def save_metrics(model_name, model_params, f1, accuracy, tn, fp, fn, tp):
    """Save metrics to json file"""
    config = read_yaml_config('parameters.yaml')
    if not config:
        logging.error("Failed to load configuration from parameters.yaml")
        return
    
    params_file_path = config['reports']['params']
    metrics_file_path = config['reports']['metrics']
    metrics_history_path = config['reports']['metrics_history']

    # Saving metrics history
    try:
        with open(metrics_history_path, "a") as file:
            metrics = {
                "timestamp": str(datetime.datetime.now()),
                "model": model_name,
                "metrics": {
                    "f1": f1,
                    "accuracy": accuracy,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp
                },
                "hyperparameters": model_params
            }
            file.write(json.dumps(metrics) + ',' + '\n')
    except Exception as e:
        logging.error(f"Unable to save metrics history to {metrics_history_path}. Error: {e}")

    # Saving metrics
    try:
        if os.path.exists(metrics_file_path):
            with open(metrics_file_path, "r") as file:
                file_contents = file.read()
                data = json.loads(file_contents) if file_contents else {}
        else:
            data = {}

        data[model_name] = {"metrics": {
            "f1": f1,
            "accuracy": accuracy,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        }}

        with open(metrics_file_path, "w") as file:
            json.dump(data, file, indent=4)

    except Exception as e:
        logging.error(f"Unable to save metrics to {metrics_file_path}. Error: {e}")

    # Saving parameters
    try:
        if os.path.exists(params_file_path):
            with open(params_file_path, "r") as file:
                file_contents = file.read()

                params_data = json.loads(file_contents) if file_contents else {}
        else:
            params_data = {}

        params_data[model_name] = {
            "hyperparameters": model_params
        }

        with open(params_file_path, "w") as file:
            json.dump(params_data, file, indent=4)

    except Exception as e:
        logging.error(f"Unable to save parameters to {params_file_path}. Error: {e}")

    return None
