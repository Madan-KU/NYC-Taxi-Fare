import os
import pandas as pd
import argparse
import logging

from modules.logger_configurator import configure_logger
from modules.read_config import read_config
from modules.data_loader import read_data

class SampleData:
    def __init__(self, config):
        self.config = config
        self.remote_path = self.config['data']['remote']
        self.sampled_data_path = self.config['data']['remote']

    def _read_data(self):
        """Read data from remote path"""
        try:
            df, filename = read_data(self.remote_path)
            return df, filename
        except FileNotFoundError:
            logging.error("File not found!")
            return None, None
        except PermissionError:
            logging.error("Permission denied when reading the file!")
            return None, None
        except Exception as e:
            logging.error(f"Error reading data: {e}")
            return None, None

    def _save_data(self, df, filename):
        """Save sampled data to a new location"""
        try:
            if not os.path.exists(self.sampled_data_path): 
                os.makedirs(self.sampled_data_path)
            file_path = os.path.join(self.sampled_data_path,filename)
            df.to_parquet(file_path, index=False)
            logging.info(f"Sampled data '{filename}' saved to '{file_path}'")
        except PermissionError:
            logging.error("Permission denied when saving the file!")
        except Exception as e:
            logging.error(f"Error saving data: {e}")

    def sample_data(self, fraction=0.01):  # Default fraction set to 0.01, can be overridden
        """Sample the data and save to a new location"""
        df, filename = self._read_data()
        if df is not None and filename is not None:
            df = df.sample(frac=fraction)
            self._save_data(df, filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="parameters.yaml", help="Path to the configuration file")
    args = parser.parse_args()
    
    configure_logger()
    config = read_config('parameters.yaml')
    sample_data_obj = SampleData(config)
    sample_data_obj.sample_data(0.02)  # For instance, sampling 2% of the data
