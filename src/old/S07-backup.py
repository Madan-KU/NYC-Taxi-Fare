import os
import logging
import pandas as pd
import pickle
import argparse
import mlflow
import mlflow.exceptions
import getpass
import logging
from mlflow.tracking import MlflowClient
from modules.read_config import read_config
from modules.logger_configurator import configure_logger


class ModelLogger:
    def __init__(self, config):
        self.config = config
        self.mlflow_config = self.config["mlflow_configuration"]
        self.model_name = self.mlflow_config["registered_model_name"]
        self.remote_server_uri = self.mlflow_config["remote_server_uri"]
        mlflow.set_tracking_uri(self.remote_server_uri)
        self.client = MlflowClient()
        self.user = getpass.getuser()

    def _get_lowest_mae_run_id(self):
        runs = mlflow.search_runs(experiment_ids=[1])
        logging.info("All Run IDs: %s", runs["run_id"].tolist())
        if "metrics.MAE" not in runs.columns:
            logging.error("Column 'metrics.MAE' does not exist in the runs DataFrame.")
            return None
        runs["metrics.MAE"] = pd.to_numeric(runs["metrics.MAE"], errors='coerce')
        lowest = runs["metrics.MAE"].min()
        best_run_id = runs[runs["metrics.MAE"] == lowest]["run_id"].iloc[0]
        logging.info("Best run ID based on lowest MAE: %s", best_run_id)
        return best_run_id

    def create_model_version(self, lowest_run_id):
        try:
            self.client.create_model_version(name=self.model_name, source=f"runs:/{lowest_run_id}/model",
                                             run_id=lowest_run_id,
                                             tags={"CreatedBy": self.user})
            logging.info("Created model version successfully.")
        except mlflow.exceptions.MlflowException as e:
            logging.error("Error creating model version: %s", str(e))

    def transition_model_versions(self, lowest_run_id, model_versions):
        for mv in model_versions:
            mv = dict(mv)
            if mv["run_id"] != lowest_run_id:
                try:
                    self.client.transition_model_version_stage(
                        name=self.model_name,
                        version=mv["version"],
                        stage="Staging"
                    )
                    logging.info("Model version %s transitioned to Staging.", mv['version'])
                except Exception as e:
                    logging.error("Error transitioning model version %s to Staging: %s", mv['version'], str(e))


    def log_production_model(self):
        lowest_run_id = self._get_lowest_mae_run_id()
        if lowest_run_id is None:
            logging.error("No suitable model found, exiting.")
            return

        try:
            loaded_model = mlflow.pyfunc.load_model(f"runs:/{lowest_run_id}/model")
            logging.info("Model loaded successfully from run_id: %s", lowest_run_id)
        except Exception as e:
            logging.error("Error loading model for run_id: %s. Error: %s", lowest_run_id, str(e))
            return

        model_versions = self.client.search_model_versions(f"name='{self.model_name}'")
        existing_run_ids = [dict(mv)["run_id"] for mv in model_versions]

        if lowest_run_id in existing_run_ids:
            logging.info("A model version for run_id: %s already exists. Skipping creation of a new version.", lowest_run_id)
        else:
            self.create_model_version(lowest_run_id)

        self.transition_model_versions(lowest_run_id, model_versions)

        try:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=self.client.get_latest_versions(self.model_name, stages=["Staging"])[0].version,
                stage="Production"
            )
            logging.info("Model version transitioned to Production.")
        except Exception as e:
            logging.error("Error transitioning model version to Production: %s", str(e))

        self.save_model(loaded_model, lowest_run_id, model_versions)


    def save_model(self, loaded_model, lowest_run_id, model_versions):
        production_model_dir = self.config["mlflow_configuration"]["production_model"]
        if not os.path.exists(production_model_dir):
            os.makedirs(production_model_dir)
        model_path = os.path.join(production_model_dir, "mlflow_model.pkl")
        with open(model_path, 'wb') as file:
            pickle.dump(loaded_model, file)
        logging.info("Model saved to: %s", model_path)

    def run(self):
        self.log_production_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="parameters.yaml", help="Path to the configuration file")
    args = parser.parse_args()

    configure_logger()
    config = read_config('parameters.yaml')
    model_logger = ModelLogger(config)
    model_logger.run()
