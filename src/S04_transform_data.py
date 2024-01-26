import os
import pickle
import logging
import argparse
import pandas as pd

from sklearn.preprocessing import StandardScaler

from modules.data_loader import read_data
from modules.read_config import read_config
from modules.logger_configurator import configure_logger


class TransformData:
    def __init__(self,config):
        self.config=config
        self.data_feature_engineered_path=self.config['data']['feature_engineered']
        self.data_X_transformed=self.config['data']['transformed']['X']
        self.data_y_transformed=self.config['data']['transformed']['y']
        self.target_column=self.config['info']['target_column']
        self.scaler_path=self.config['scaler_dir']

    def _read_data(self):
        """Read data from cleansed path"""
        try:
            df, filename = read_data(self.data_feature_engineered_path)
            return df, filename
        except Exception as e:
            logging.error(f"Error reading data: {e}")
            return None, None
        

    def _separate_features_target(self,df):
        """Split data into X, y"""


        try:
            if self.target_column not in df.columns:
                logging.error(f"Target column '{self.target_column}' not found in the dataset.")
                return None, None
            
            X = df.drop(columns=self.target_column, axis=1)
            y = df[self.target_column]
            return X, y

        except Exception as e:
            logging.error(f"Error occurred during splitting data: {e}")
            return None, None

    def _process_features(self, data, prefix):
        try:
            if isinstance(data, pd.Series):
                # If the data is a Series (target variable), just scale it
                scaler = StandardScaler()
                data = scaler.fit_transform(data.values.reshape(-1, 1))
                data = pd.Series(data.ravel())
            else:
                # If the data is a DataFrame (features)
                continuous_cols = data.select_dtypes(include=['float64']).columns
                categorical_cols = data.select_dtypes(exclude=['float64']).columns
                print(categorical_cols,"\n",continuous_cols )
                scaler = StandardScaler()
                scaled = scaler.fit_transform(data[continuous_cols])

                data_scaled = pd.DataFrame(scaled, columns=continuous_cols)
                data_dummies = pd.get_dummies(data[categorical_cols], dtype=int, drop_first=False)
                data = pd.concat([data_scaled, data_dummies], axis=1)

            scaler_filename = prefix + "_" + "scaler.pkl"
            scaler_file_path = os.path.join(self.scaler_path, scaler_filename)
            if not os.path.exists(self.scaler_path):
                os.makedirs(self.scaler_path)
            with open(scaler_file_path, 'wb') as file:
                pickle.dump(scaler, file)
            logging.info(f"Scaler saved at {scaler_file_path}")
            return data
        except Exception as e:
            logging.error(f"Error occurred during processing features: {e}")
            return None
        

    def _save_data(self,filename, output_path, data):
        try:
            if isinstance(data, pd.Series):
                data = data.to_frame(self.target_column)

            data.to_parquet(output_path, index=False)
            logging.info(f"'{filename}' saved to '{output_path}'")
            return True

        except Exception as e:
            logging.error(f"Error occurred while saving '{filename}' to '{output_path}': {e}")


    def execute_transformation(self):
            
        try:
            df, filename=self._read_data()
            X, y=self._separate_features_target(df)

            if X is not None and y is not None:
                processed_X = self._process_features(X,"X")
                processed_y = self._process_features(y,"y")

                if processed_X is not None and processed_y is not None:

                        X_output_path = os.path.join(self.config['data']['transformed']['X'], filename)
                        y_output_path = os.path.join(self.config['data']['transformed']['y'], filename)
                                               
                        if not os.path.exists(config['data']['transformed']['X']):
                            os.makedirs(config['data']['transformed']['X'])

                        if not os.path.exists(config['data']['transformed']['y']):
                            os.makedirs(config['data']['transformed']['y'])

                        save_X_flag=self. _save_data(filename, X_output_path, processed_X)
                        save_y_flag=self._save_data(filename, y_output_path, processed_y)

                        if save_X_flag and save_y_flag:
                            logging.info(f"Data processed and saved to: X -> '{X_output_path}', y -> '{y_output_path}'")
                
                else:
                    logging.warning("Data processing failed")
            else:
                logging.warning("Failed to split data into X and y")
            
        except Exception as e:
            logging.error(f"Error occurred in transformation function: {e}")


        

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="parameters.yaml", help="Path to the configuration file")
    args = parser.parse_args()
    
    configure_logger()
    config = read_config('parameters.yaml')
    data_transformed=TransformData(config)
    data_transformed.execute_transformation()
