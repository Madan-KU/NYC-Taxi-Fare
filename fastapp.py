import os
import yaml
import pickle
import mlflow.pyfunc
import pandas as pd
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

# run >> uvicorn fastapp:app --host 0.0.0.0 --port 8000
# http://localhost:8000/docs

# app = FastAPI()
app = FastAPI(template_directory="prediction_app/templates")


mlflow.set_tracking_uri('http://127.0.0.1:1234')

model_name = "GradientBoostingRegressor"
model_stage = "Staging"
model_uri = f"models:/{model_name}/{model_stage}"

mlflow_model = mlflow.pyfunc.load_model(model_uri=model_uri)


class InputData(BaseModel):
    trip_miles: float
    trip_time: float
    access_a_ride_flag: str
    request_datetime_hour: int
    request_datetime_day: str
    request_datetime_month: str
    duration_minutes: float
    wait_time_minutes: float
    service_time_minutes: float
    on_scene_datetime_hour: int
    on_scene_datetime_day: str
    on_scene_datetime_month: str
    pickup_datetime_hour: int
    pickup_datetime_day: str
    pickup_datetime_month: str
    dropoff_datetime_hour: int
    dropoff_datetime_day: str
    dropoff_datetime_month: str
    average_speed: float


@app.post('/predict')
async def predict(input_data: InputData):
    try:
        df_mapped = await run_in_threadpool(map_data_to_df, input_data.dict())

        prediction = await run_in_threadpool(perform_prediction, df_mapped)

        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def batch_predict(input_data_list: List[InputData]):
    try:
        
        df_list = [await run_in_threadpool(map_data_to_df, input_data.dict()) for input_data in input_data_list]
        df_batch = pd.concat(df_list, ignore_index=True)
        predictions = await run_in_threadpool(perform_prediction, df_batch)
        
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def map_data_to_df(input_data):
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

    df_mapped = pd.DataFrame(columns=keys_list, index=[0]).fillna(0)

    for key, value in input_data.items():
        if key in df_mapped.columns:
            df_mapped[key] = value

    days_columns = [
        'request_datetime_day', 'on_scene_datetime_day',
        'pickup_datetime_day', 'dropoff_datetime_day'
    ]
    for col in days_columns:
        if col in input_data:
            column_name = f"{col}_{input_data[col]}"
            if column_name in df_mapped.columns:
                df_mapped[column_name] = 1

    return df_mapped


def perform_prediction(data):

    continuous_cols = [
    'trip_miles', 'trip_time', 'duration_minutes',
    'wait_time_minutes', 'service_time_minutes', 'average_speed'
    ]
    try:
        config_path = 'parameters.yaml'

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        base_dir = os.path.dirname(config_path)
        X_scaler_path = os.path.join(base_dir, config['prediction_app']['scaler'], "X_scaler.pkl")
        y_scaler_path = os.path.join(base_dir, config['prediction_app']['scaler'], "y_scaler.pkl")

        with open(X_scaler_path, 'rb') as f:
            X_scaler = pickle.load(f)
        with open(y_scaler_path, 'rb') as f:
            y_scaler = pickle.load(f)

        data[continuous_cols] = X_scaler.transform(data[continuous_cols])
        prediction = mlflow_model.predict(data)
        inverse_transformed_y = y_scaler.inverse_transform(prediction.reshape(-1, 1))

        return inverse_transformed_y
    except Exception as e:
        raise e

"""
{
  "trip_miles": 1.0,
  "trip_time": 2.0,
  "access_a_ride_flag": "Y",
  "request_datetime_hour": 1,
  "request_datetime_day": "Monday",
  "request_datetime_month": "January",
  "duration_minutes": 3.0,
  "wait_time_minutes": 4.0,
  "service_time_minutes": 5.0,
  "on_scene_datetime_hour": 6,
  "on_scene_datetime_day": "Tuesday",
  "on_scene_datetime_month": "February",
  "pickup_datetime_hour": 7,
  "pickup_datetime_day": "Wednesday",
  "pickup_datetime_month": "March",
  "dropoff_datetime_hour": 8,
  "dropoff_datetime_day": "Thursday",
  "dropoff_datetime_month": "April",
  "average_speed": 9.0
}
"""