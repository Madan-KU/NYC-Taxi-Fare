import os
import sqlite3
import numpy as np
import logging
from modules.data_loader import read_data
from modules.read_config import read_config
from modules.logger_configurator import configure_logger

configure_logger()
config = read_config('parameters.yaml')

class InvalidDataSplitter:
    def __init__(self, config):
        self.config = config
        self.remote_dir = self.config['data']['remote']
        self.db_path = self.config['data']['database']
        self.db_file_path = os.path.join(self.db_path, "data_db.sqlite3")

    def _read_data(self):
        """Read data from remote path"""
        try:
            df, filename = read_data(self.remote_path)
            return df, filename
        except Exception as e:
            logging.error(f"Error reading data: {e}")
            return None, None
    
    def _splitter(self, df):
        is_na = df.isna()
        numeric_df = df.select_dtypes(include=['number'])
        is_negative = (numeric_df < 0)
                
        upper_bound = numeric_df.mean() + 3*numeric_df.std()
        lower_bound = numeric_df.mean() - 3*numeric_df.std()
        is_outlier = ((numeric_df > upper_bound) | (numeric_df < lower_bound))

        invalid_data_condition = is_na | is_negative | is_outlier
        invalid_data_df = df[invalid_data_condition.any(axis=1)]
        valid_data_df = df[~invalid_data_condition.any(axis=1)]

        logging.info(f"Data split into {len(valid_data_df)} valid records and {len(invalid_data_df)} invalid records.")
        return invalid_data_df, valid_data_df

    def _save_to_db(self, valid_data, invalid_data):
        conn = sqlite3.connect(self.db_file_path)

        valid_data.to_sql('valid_data', conn, if_exists='replace', index=False)
        invalid_data.to_sql('invalid_data', conn, if_exists='replace', index=False)
        logging.info(f"Valid and invalid data saved to SQLite database -> {self.db_file_path}")

    def split_valid_invalid_data(self, return_valid_df=False):
        df, _ = self._read_data()
        invalid_data, valid_data = self._splitter(df)
        self._save_to_db(invalid_data, valid_data)

        if return_valid_df:
            return valid_data


if __name__ == '__main__':
    data_splitter_instance = InvalidDataSplitter(config)
    data_splitter_instance.split_valid_invalid_data()
