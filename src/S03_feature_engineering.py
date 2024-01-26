
import os
import logging
import argparse
import pandas as pd

from modules.data_loader import read_data
from modules.read_config import read_config
from modules.logger_configurator import configure_logger
from modules.build_schema import SchemaBuilder


class FeatureEngineer:
    def __init__(self,config):
        self.config=config
        self.data_cleansed_path=self.config['data']['cleansed']
        self.feature_engineered_path=self.config['data']['feature_engineered']


    def _read_data(self):
        """Read data from cleansed path"""
        try:
            df, filename = read_data(self.data_cleansed_path)
            return df, filename
        except Exception as e:
            logging.error(f"Error reading data: {e}")
            return None, None
       
        
    def _feature_engineer(self,df):
        # Datetime Features
        for col in ['request_datetime', 'on_scene_datetime', 'pickup_datetime', 'dropoff_datetime']:
            df[col] = pd.to_datetime(df[col])
            df[col + '_hour'] = df[col].dt.hour.astype('category')
            df[col + '_day'] = df[col].dt.strftime('%A').astype('category') 
            # df[col + '_month'] = df[col].dt.strftime('%B').astype('category')

            df['duration_minutes'] = abs(df['pickup_datetime'] - df['dropoff_datetime']).dt.total_seconds() / 60
            df['wait_time_minutes'] = (df['on_scene_datetime'] - df['request_datetime']).dt.total_seconds() / 60
            df['service_time_minutes'] = (df['dropoff_datetime'] - df['on_scene_datetime']).dt.total_seconds() / 60

        # Trip Features
        valid_speed_rows = df['trip_time'] != 0
        df.loc[valid_speed_rows, 'average_speed'] = df['trip_miles'] / (df['trip_time'] / 60)  # Assuming trip_time is in minutes


        return df

    def _save_data(self,filename, output_path, data):
        try:
            data.to_parquet(output_path, index=False)
            logging.info(f"'{filename}' saved to '{output_path}'")
            return True

        except Exception as e:
            logging.error(f"Error occurred while saving '{filename}' to '{output_path}': {e}")
 

    def _drop_features(self,df):

        columns=['access_a_ride_flag',
            'airport_fee', 'base_passenger_fare', 'bcf', 'congestion_surcharge',
            'dispatching_base_num', 'hvfhs_license_num', 'originating_base_num',
            'on_scene_datetime', 'pickup_datetime', 'request_datetime',
            'dropoff_datetime','sales_tax', 'shared_match_flag', 
            'shared_request_flag','tips', 'tolls', 'wav_match_flag',
            'wav_request_flag', 'DOLocationID', 'PULocationID']
        
        df.drop(columns=columns,inplace=True)

        return df
    
    # def _build_schema(self, file_path):
    #     schema_instance = SchemaBuilder()
    #     schema_instance.generate_and_save_schema(file_path)
    #     return True


    def perform_feature_engineering(self):
        df, filename=self._read_data()
        df_fe=self._feature_engineer(df)
        df=self._drop_features(df_fe)

        df.dropna(inplace=True)
        logging.info("Dropped rows with NaN values")
        
        if df.isna().sum().sum() > 0:
            logging.warning(f"There are still {df.isna().sum().sum()} missing values after cleansing.")
        
        if df is not None:
            output_file_path= os.path.join(self.feature_engineered_path,filename)

            self._save_data(filename, output_file_path, df)

        else:
            logging.warning("DataFrame is empty or None. No feature engineering performed.")


        # try:
        #     if self._build_schema(self.feature_engineered_path):
        #         logging.info(f"Schema successfully built and saved at {self.feature_engineered_path}")
        #     else:
        #         logging.warning(f"Failed to build schema for {self.feature_engineered_path}")
        # except Exception as e:
        #     logging.error(f"Error encountered while building schema: {str(e)}")


        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="parameters.yaml", help="Path to the configuration file")
    args = parser.parse_args()

    configure_logger()
    config = read_config('parameters.yaml')
    feature_engineer_obj=FeatureEngineer(config)
    feature_engineer_obj.perform_feature_engineering()



