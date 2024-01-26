import os
import logging
import argparse
import pandas as pd
from sklearn.impute import SimpleImputer

from modules.data_loader import read_data
from modules.read_config import read_config
from modules.logger_configurator import configure_logger


class CleanData:
    def __init__(self,config):
        self.config=config
        self.target_column=self.config['info']['target_column']
        self.raw_path=self.config['data']['raw']
        self.cleansed_data_path=self.config['data']['cleansed']

    def _read_data(self):
        """Read data from remote path"""
        try:
            df, filename = read_data(self.raw_path)
            return df, filename
        except Exception as e:
            logging.error(f"Error reading data: {e}")
            return None, None
        
    def _save_data(self, df, filename):
        """Save dataframe to parquet format"""
        try:
            file_path = os.path.join(self.cleansed_data_path, filename)
            df.to_parquet(file_path, index=False)
            logging.info(f"'{filename}' loaded to '{file_path}'")
        except Exception as e:
            logging.error(f"Error saving data: {e}")
        
    
    def cleanse_data(self):
        df, filename = self._read_data()
        logging.info("Cleaning data...")

        # Identify continuous and categorical columns
        continuous_cols = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = df.select_dtypes(exclude=['float64', 'int64']).columns

        if df is not None:
            try:
                # Impute continuous columns
                imputer_cont = SimpleImputer(strategy='mean')
                df[continuous_cols] = imputer_cont.fit_transform(df[continuous_cols])

                # Impute categorical columns
                imputer_cat = SimpleImputer(strategy='most_frequent')
                df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

                logging.info("Imputed missing values using Simple Imputer")
                df.dropna(inplace=True)
                logging.info("Dropped rows with NaN values")
                if df.isna().sum().sum() > 0:
                    logging.warning(f"There are still {df.isna().sum().sum()} missing values after cleansing.")


            except Exception as e:
                logging.error(f"Error Imputing missing values. error: {e}")

        self._save_data(df, filename)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="parameters.yaml", help="Path to the configuration file")
    args = parser.parse_args()

    configure_logger()
    config = read_config('parameters.yaml')
    data_cleansed=CleanData(config)
    data_cleansed.cleanse_data()