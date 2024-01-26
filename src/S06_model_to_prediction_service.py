import os
import json
import shutil
import argparse
import logging

from modules.read_config import read_config
from modules.logger_configurator import configure_logger


class ModelToPredictionService:
    def __init__(self,config):
        self.config=config
        self.report_metrics = self.config['reports']['metrics']
        self.saved_models_dir = self.config['saved_model_dir']
        self.serving_model_dir = self.config['prediction_app']['model']
        self.scaler_dir=self.config['scaler_dir']
        self.serving_scaler_dir = self.config['prediction_app']['scaler']


    def _copy_best_model_to_prediction(self):
        """Copy best model to serving model directory """
        
        # Load metrics from metrics.json
        with open(os.path.join(self.report_metrics), 'r') as file:
            metrics_data = json.load(file)

        # Get r2 values
        scores = {model: metrics_data[model]['metrics']['mae'] for model in metrics_data}

        # Find model with the highest r2/MAE/RMSE value
        best_model = max(scores, key=scores.get)
        best_model_path = os.path.join(self.saved_models_dir, best_model+'.pkl')
        serving_model_path= os.path.join(self.serving_model_dir,'model.pkl')

        if not os.path.exists(self.serving_model_dir):
            os.makedirs(self.serving_model_dir,exist_ok=True)

        # Copy best model to prediction_app directory
        shutil.copy(best_model_path, serving_model_path)
        logging.info(f"Copied '{best_model}' model to '{serving_model_path}'")


    def _copy_scaler_to_prediction(self):
        """Copy all scaler files ending with 'scaler.pkl' into serving folder."""
        
        if not os.path.exists(self.serving_scaler_dir):
            os.makedirs(self.serving_scaler_dir, exist_ok=True)
        
        # Iterate over all files in the source directory
        for file in os.listdir(self.scaler_dir):
            if file.endswith("scaler.pkl"):
                source_scaler_file_path = os.path.join(self.scaler_dir, file)
                destination_scaler_file_path = os.path.join(self.serving_scaler_dir, file)

                # Copy the scaler file to the serving directory
                shutil.copy(source_scaler_file_path, destination_scaler_file_path)
                
                # Log the action
                logging.info(f"Scaler: '{file}' copied from '{source_scaler_file_path}' to '{destination_scaler_file_path}'.")

    def exectute_model_to_prediction_service(self):
            self._copy_best_model_to_prediction()
            self._copy_scaler_to_prediction()


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="parameters.yaml", help="Path to the configuration file")
    args = parser.parse_args()

    configure_logger()
    config = read_config('parameters.yaml')
    data_transformed=ModelToPredictionService(config)
    data_transformed.exectute_model_to_prediction_service()