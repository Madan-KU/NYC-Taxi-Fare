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

from dvclive import Live


from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from modules.data_loader import read_data
from modules.read_config import read_config
from modules.logger_configurator import configure_logger
from modules.save_metrics_regression import save_metrics


import warnings
warnings.filterwarnings("ignore")


class TrainEvaluate:
    def __init__(self,config):
        self.config=config
        self.target_column=self.config['info']['target_column']
        self.test_size=self.config['train_evaluate']['split_data']['test_size']
        self.random_state=self.config['info']['random_state']
        self.saved_model_directory=self.config['saved_model_dir']
        self.experiment_name = self.config['mlflow_configuration']['experiment_name']
        self.X_path= self.config['data']['transformed']['X']
        self.y_path= self.config['data']['transformed']['y']
        self.remote_server_uri=self.config['mlflow_configuration']['remote_server_uri']
        self.models_yaml=self.config['model']



    def _get_model(self,model_name, model_params):
        """Return an instance of the model based on the provided name and parameters."""
        model_class = globals().get(model_name)
        if model_class:
            return model_class(**model_params)
        raise ValueError(f"Unknown model class: {model_name}")
    
    def _split_data(self,dfx,dfy):
        """Split the dataframe into train and test sets."""
        X = dfx
        y = dfy
        return train_test_split(
            X, y, test_size=self.test_size,
            random_state= self.random_state
            )


    def _save_model(self,model_name, model):
        """Save the trained model to a directory."""
        if not os.path.exists(self.saved_model_directory):
            os.makedirs(self.saved_model_directory)
        filepath = os.path.join(self.saved_model_directory, model_name+ '.pkl')
        with open(filepath, 'wb') as file:
            pickle.dump(model, file)
            logging.info(f"'{model}' saved to '{filepath}'")

    
    def _train_and_evaluate(self,X_train, X_test, y_train, y_test, model_name, model_params):
        """Train the model and evaluate its performance."""

        mlflow.set_tracking_uri(self.remote_server_uri)

        # Set artifact location to a known directory.
        artifact_location = os.path.join(os.getcwd(), "mlflow", "mlflow_artifacts")

        if not os.path.exists(artifact_location):
            os.makedirs(artifact_location)

        # Dynamic Run names for mlflow
        # unique_id = str(uuid.uuid4())[:4]  # 'n' characters of UUID
        # run_name = f"{model_name}_{unique_id}"
        # experiment_name = config['mlflow_configuration']['experiment_name']

        run_name = model_name
        mlflow.set_experiment(self.experiment_name) #Changes**
        mlflow.create_experiment(name=self.experiment_name, artifact_location=artifact_location)


        with mlflow.start_run(run_name=run_name):  # mlflow*
            model = self._get_model(model_name, model_params)

            # Reshape y_train and y_test to 1D arrays
            y_test = y_test.squeeze()
            y_train = y_train.squeeze()
        
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            logging.info(f"{model_name} - RMSE: {rmse:.2f} - MAE: {mae:.2f} - R2 Score: {r2:.2f}")

            # Log parameters and metrics to mlflow
            mlflow.log_params(model_params) # mlflow*
            mlflow.log_metric("RMSE", rmse) # mlflow*
            mlflow.log_metric("MAE", mae) # mlflow*
            mlflow.log_metric("R2 Score", r2) # mlflow*

            # Save the model to mlflow
            sklearn.log_model(model, "model") # mlflow*
            model_uri = f"runs:/{mlflow.active_run().info.run_uuid}/model"
            mlflow.register_model(model_uri, model_name)

            tracking_url_type_store=urlparse(mlflow.get_artifact_uri()).scheme

            if tracking_url_type_store != "file":
                sklearn.log_model(model, "model")

            # Log parameters and metrics to dvclive
            live.log_params(model_params) # dvclive*
            live.log_metric("RMSE", rmse) # dvclive*
            live.log_metric("MAE", mae) # dvclive*
            live.log_metric("R2 Score", r2) # dvclive*       
            
            save_metrics(model_name, model_params, rmse, mae, r2)

            return model_name, model
        

    def exectute_train_evaluate(self):

        dfx, _ = read_data(self.X_path)
        dfy, _ = read_data(self.y_path)

        X_train, X_test, y_train, y_test = self._split_data(dfx,dfy)

        
        for model_name, model_config in self.models_yaml.items():
            model_params = model_config.get('params', {})

            live.log_params(model_params)

            model_name, model = self._train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model_params)
            self._save_model(model_name, model)

            live.next_step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="parameters.yaml", help="Path to the configuration file")
    args = parser.parse_args()

    configure_logger()
    config = read_config('parameters.yaml')

    with Live() as live:
        train_eval_obj=TrainEvaluate(config)
        train_eval_obj.exectute_train_evaluate()
