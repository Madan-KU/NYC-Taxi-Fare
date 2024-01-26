import os
import yaml
import pickle
import pandas as pd
import logging

import warnings
warnings.simplefilter(action='ignore', category=Warning)


class Files:
    """Utility class for reading and loading files."""

    @staticmethod
    def read_yaml(file_path):
        """Load YAML file."""
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    @staticmethod
    def load_pickle(pickle_path):
        """Load Pickle file."""
        with open(pickle_path, 'rb') as pickle_file:
            return pickle.load(pickle_file)


class ModelPredictor:
    """Class for loading the model and making predictions."""

    def __init__(self, model_path, X_scaler_path, y_scaler_path):
        """Initialize and load the model and scalers."""
        self.model = Files.load_pickle(model_path)
        self.X_scaler = Files.load_pickle(X_scaler_path)
        self.y_scaler = Files.load_pickle(y_scaler_path)

    def predict(self, data):
        """Predict the output for given data."""
        data[continuous_cols] = self.X_scaler.transform(data[continuous_cols])
        prediction = self.model.predict(data)
        inverse_transformed_y = self.y_scaler.inverse_transform(prediction.reshape(-1, 1))
        return inverse_transformed_y


def map_data_to_df(input_data):
    """Map input data to a DataFrame in the expected format."""
    # Initialize DataFrame with zeros
    df_mapped = pd.DataFrame(columns=keys_list, index=[0]).fillna(0)

    # Map values
    for key, value in input_data.items():
        if key in df_mapped.columns:
            df_mapped[key] = value

    # One-hot encode days
    days_columns = [
        'request_datetime_day', 'on_scene_datetime_day',
        'pickup_datetime_day', 'dropoff_datetime_day'
    ]
    for col in days_columns:
        if col in input_data:
            column_name = f"{col}_{input_data[col]}"
            if column_name in df_mapped.columns:
                df_mapped[column_name] = 1

    # Set data types
    for col in continuous_cols:
        if col in df_mapped.columns:
            df_mapped[col] = df_mapped[col].astype(float)
    for col in categorical_cols:
        if col in df_mapped.columns:
            df_mapped[col] = df_mapped[col].astype(str) if "day" in col else df_mapped[col].astype(int)

    return df_mapped



    # @staticmethod        
    # def validate_data(schema, data):
    #     # for key, value in schema.items():
    #     #     # Check if key exists in data
    #     #     if key not in data:
    #     #         logging.warning(f"Key {key} missing in data.")
    #     #         return False

    #     #     # Check data type
    #     #     if value['data_type'] == 'float64' and not isinstance(data[key], float):
    #     #         logging.warning(f"Expected float for {key}, got {type(data[key])}.")
    #     #         return False

    #     #     # Check value range for float and int types
    #     #     if value['data_type'] in ['float64', 'int32']:
    #     #         if not value['min_value'] <= data[key] <= value['max_value']:
    #     #             logging.warning(f"Value for {key} out of range.")
    #     #             return False

    #     #     # Check for valid categories
    #     #     if value['data_type'] == 'category':
    #     #         if data[key] not in value['unique_values']:
    #     #             logging.warning(f"Invalid category for {key}.")
    #     #             return False

    #     return True

    # @staticmethod
    # def read_and_validate(schema_path, data):
    #     return True
    #     # schema = Files.read_json(schema_path)
    #     # return Validator.validate_input(schema, data)




def perform_prediction(data):
    """Main function to load model and make predictions."""
    try:
        config = Files.read_yaml('parameters.yaml')
        model_file_path = os.path.join(config['prediction_app']['model'])
        X_scaler_path = os.path.join(config['prediction_app']['scaler'], "X_scaler.pkl")
        y_scaler_path= os.path.join(config['prediction_app']['scaler'], "y_scaler.pkl")
        # input_schema_path = config['schema']['input']

        # if not  ModelPredictor.validate_data(schema, data):
        #     raise ValueError("Input data validation failed.")

        predictor = ModelPredictor(model_file_path, X_scaler_path, y_scaler_path)
        prediction = predictor.predict(data)
        

        logging.info(f"Prediction result: {prediction}")
        return prediction
    except Exception as e:
        logging.error(f"Error in main_prediction: {e}")
        raise


#> Mapping

keys_list = [
    'trip_miles', 'trip_time', 'duration_minutes', 'wait_time_minutes', 'service_time_minutes', 'average_speed', 
    'request_datetime_hour', 'on_scene_datetime_hour', 'pickup_datetime_hour', 'dropoff_datetime_hour',
    'request_datetime_day_Friday', 'request_datetime_day_Monday', 
    'request_datetime_day_Saturday', 'request_datetime_day_Sunday', 'request_datetime_day_Thursday', 
    'request_datetime_day_Tuesday', 'request_datetime_day_Wednesday', 'on_scene_datetime_day_Friday', 
    'on_scene_datetime_day_Monday', 'on_scene_datetime_day_Saturday', 'on_scene_datetime_day_Sunday', 
    'on_scene_datetime_day_Thursday', 'on_scene_datetime_day_Tuesday', 'on_scene_datetime_day_Wednesday', 
    'pickup_datetime_day_Friday', 'pickup_datetime_day_Monday', 'pickup_datetime_day_Saturday', 
    'pickup_datetime_day_Sunday', 'pickup_datetime_day_Thursday', 'pickup_datetime_day_Tuesday', 
    'pickup_datetime_day_Wednesday', 'dropoff_datetime_day_Friday', 'dropoff_datetime_day_Monday', 
    'dropoff_datetime_day_Saturday', 'dropoff_datetime_day_Sunday', 'dropoff_datetime_day_Thursday', 
    'dropoff_datetime_day_Tuesday', 'dropoff_datetime_day_Wednesday'
]

# Columns
continuous_cols = [
    'trip_miles', 'trip_time', 'duration_minutes',
    'wait_time_minutes', 'service_time_minutes', 'average_speed'
]

categorical_cols = [
    'request_datetime_hour', 'request_datetime_day', 'request_datetime_month',
    'on_scene_datetime_hour', 'on_scene_datetime_day', 'on_scene_datetime_month',
    'pickup_datetime_hour', 'pickup_datetime_day', 'pickup_datetime_month',
    'dropoff_datetime_hour', 'dropoff_datetime_day', 'dropoff_datetime_month'
]


# input_data = {
#     "trip_miles": 10.5,
#     "trip_time": 45.0,
#     "request_datetime_hour": 10,
#     "request_datetime_day": "Monday",
#     "duration_minutes": 25.5,
#     "wait_time_minutes": -5.5,
#     "wait_time_minutes": 15.0,
#     "on_scene_datetime_hour": 11,
#     "on_scene_datetime_day": "Monday",
#     "pickup_datetime_hour": 12,
#     "pickup_datetime_day": "Monday",
#     "dropoff_datetime_hour": 12,
#     "dropoff_datetime_day": "Monday",
#     "average_speed": 0.5
# }



# df_mapped = map_data_to_df(input_data)
# output=perform_prediction(df_mapped)
# print(output)