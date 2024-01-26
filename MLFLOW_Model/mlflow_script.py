# NOTE: based as S05_train_and_evaluate.py

import os
import uuid
import pickle
import mlflow
import logging
import argparse
import numpy as np
import pandas as pd 
from mlflow import sklearn 
from urllib.parse import urlparse

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


import logging
import pyarrow as pa
import pyarrow.parquet as pq

import warnings
warnings.filterwarnings("ignore")


class TrainEvaluate:
    def __init__(self, config):
        self.config = config
        self.target_column = self.config['info']['target_column']
        self.test_size = self.config['train_evaluate']['split_data']['test_size']
        self.random_state = self.config['info']['random_state']
        self.experiment_name = self.config['mlflow_configuration']['experiment_name']
        self.X_path = self.config['data']['transformed']['X']
        self.y_path = self.config['data']['transformed']['y']
        self.remote_server_uri = self.config['mlflow_configuration']['remote_server_uri']
        self.models_yaml = self.config['model']
        self.experiment_id = None 

    def read_data(self,directory):
        if not os.path.exists(directory):
            logging.warning(f"Directory {directory} does not exist.")
            return None,'None'
        
        for file in os.listdir(directory):
            if file.endswith(".parquet"):
                try:
                    file_path = os.path.join(directory, file)
                    parquet_table = pq.read_table(file_path)
                    df=parquet_table.to_pandas()
                    logging.info(f"Successfully read {file}, shape {df.shape}")
                    return df, file
                
                except Exception as e:
                    logging.error(f"Error reading {file}: {e}")
                    continue

        logging.warning(f"No data file found in {directory}.")
        return None, 'None'

    def _split_data(self, dfx, dfy):
        """Split the dataframe into train and test sets."""
        X = dfx
        y = dfy
        return train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state
        )
    
    def _get_model(self,model_name, model_params):
        """Return an instance of the model based on the provided name and parameters."""
        model_class = globals().get(model_name)
        if model_class:
            return model_class(**model_params)
        raise ValueError(f"Unknown model class: {model_name}")
    
    def _train_and_evaluate(self, X_train, X_test, y_train, y_test, model_name, model_params):
        """Train the model and evaluate its performance."""
        mlflow.set_tracking_uri(self.remote_server_uri)

        artifact_location = os.path.join(os.getcwd(), "mlflow_artifacts")

        if not os.path.exists(artifact_location):
            os.makedirs(artifact_location)

        run_name = model_name

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=run_name):
            model = self._get_model(model_name, model_params)

            y_test = y_test.squeeze()
            y_train = y_train.squeeze()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            logging.info(f"{model_name} - RMSE: {rmse:.2f} - MAE: {mae:.2f} - R2 Score: {r2:.2f}")

            mlflow.log_params(model_params)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2 Score", r2)

            sklearn.log_model(model, "model")
            model_uri = f"runs:/{mlflow.active_run().info.run_uuid}/model"
            mlflow.register_model(model_uri, model_name)

            tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

            if tracking_url_type_store != "file":
                sklearn.log_model(model, "model")

            return model_name, model

    def exectute_train_evaluate(self):
        dfx, _ = self.read_data(self.X_path)
        dfy, _ = self.read_data(self.y_path)

        X_train, X_test, y_train, y_test = self._split_data(dfx, dfy)

        model_name = 'DecisionTreeRegressor'
        model_params = {
            'max_depth': self.config['model']['DecisionTreeRegressor']['params']['max_depth'],
            'min_samples_leaf': self.config['model']['DecisionTreeRegressor']['params']['min_samples_leaf'],
            'min_samples_split': self.config['model']['DecisionTreeRegressor']['params']['min_samples_split']
        }

        
        for model_name, model_config in self.models_yaml.items():
            model_params = model_config.get('params', {})
            model_name, model = self._train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model_params)




def main():
    args = argparse.ArgumentParser()
    args.add_argument("--max_depth", type=int, default=10)
    args.add_argument("--min_samples_leaf", type=int, default=10)
    args.add_argument("--min_samples_split", type=int, default=8)
    parsed_args = args.parse_args()

    config = {
        'info': {
            'target_column': 'driver_pay',
            'random_state': 50
        },
        'train_evaluate': {
            'split_data': {
                'test_size': 0.3
            }
        },
        'mlflow_configuration': {
            'experiment_name': 'new_experiment',
            'remote_server_uri': 'http://127.0.0.1:5000'
        },
        'data': {
            'transformed': {
                'X': '/mnt/2890FCE090FCB582/DS/NYC MLOPS/data/transformed/X',
                'y': '/mnt/2890FCE090FCB582/DS/NYC MLOPS/data/transformed/y'
            }
        },
        'model': {
            'DecisionTreeRegressor': {
                'params': {
                    'max_depth': parsed_args.max_depth,
                    'min_samples_leaf': parsed_args.min_samples_leaf,
                    'min_samples_split': parsed_args.min_samples_split
                }
            }
        }
         
    }

    train_eval_obj = TrainEvaluate(config=config)
    train_eval_obj.exectute_train_evaluate()


if __name__=="__main__":
    main()