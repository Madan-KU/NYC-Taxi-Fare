import os
import pandas as pd
import sqlite3
import logging
import argparse

from modules.logger_configurator import configure_logger
from modules.read_config import read_config
from modules.data_loader import read_data


class InvalidDataSplitter:
    def __init__(self, config):
        self.config = config
        self.db_path = self.config['data']['database']
        self.db_file_path = os.path.join(self.db_path, "data_db.sqlite3")

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

    def split_valid_invalid_data(self, df, return_valid_df=False):
        invalid_data, valid_data = self._splitter(df)
        self._save_to_db(invalid_data, valid_data)

        if return_valid_df:
            return valid_data


class LoadData:
    def __init__(self, config):
        self.config = config
        self.raw_data_path = self.config['data']['raw']
        self.remote_path = self.config['data']['remote']

    def _read_data(self):
        """Read data from remote path"""
        try:
            df, filename = read_data(self.remote_path)
            return df, filename
        except Exception as e:
            logging.error(f"Error reading data: {e}")
            return None, None

    def _format_column_name(self, df):
        """Format dataframe column names"""
        try:
            df.columns = df.columns.str.replace(' ', '_')
            return df
        except Exception as e:
            logging.error(f"Error formatting column names: {e}")
            return df

    def _save_data(self, df, filename):
        """Save dataframe to parquet format"""
        try:
            file_path = os.path.join(self.raw_data_path, filename)
            df.to_parquet(file_path, index=False)
            logging.info(f"'{filename}' loaded to '{self.raw_data_path}'")
        except Exception as e:
            logging.error(f"Error saving data: {e}")

    def load_remote_to_raw(self):
        """Load data from remote and save to raw path after formatting column names"""
        df, filename = self._read_data()
        if df is not None and filename is not None:
            df = self._format_column_name(df)
            
            # Splitting data using DataSplitter
            data_splitter = InvalidDataSplitter(self.config)
            valid_data = data_splitter.split_valid_invalid_data(df, return_valid_df=True)
            
            self._save_data(valid_data, filename)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="parameters.yaml", help="Path to the configuration file")
    args = parser.parse_args()

    configure_logger()
    config = read_config('parameters.yaml')
    data_loader = LoadData(config)
    data_loader.load_remote_to_raw()
