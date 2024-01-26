import os
import pandas as pd
from datetime import datetime, timedelta
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage


store = FeatureStore(repo_path="feature_repo")

entity_df = pd.read_parquet(path=r"feature_repo/data/y.parquet")  

storage_file_path="../Dataset/Store/"

if not os.path.exists(storage_file_path):
    os.makedirs(storage_file_path)

# Filter to get only the last n days of data
timestamp_threshold = datetime.utcnow() - timedelta(days=1)
filtered_entity_df = entity_df[entity_df['event_timestamp'] > timestamp_threshold]



hist_df = store.get_historical_features(
    entity_df=filtered_entity_df,
    features=[
        "trip_features_X:trip_miles",
        "trip_features_X:trip_time",
        "trip_features_X:request_datetime_hour",
        "trip_features_X:request_datetime_day",
        "trip_features_X:duration_minutes",
        "trip_features_X:wait_time_minutes",
        "trip_features_X:service_time_minutes",
        "trip_features_X:on_scene_datetime_hour",
        "trip_features_X:on_scene_datetime_day",
        "trip_features_X:pickup_datetime_hour",
        "trip_features_X:pickup_datetime_day",
        "trip_features_X:dropoff_datetime_hour",
        "trip_features_X:dropoff_datetime_day",
        "trip_features_X:average_speed",
        "trip_features_y:driver_pay",
        "trip_features_y:event_timestamp"
    ]
)

data=store.create_saved_dataset(
    from_=hist_df,
    name="nyc_taxi_dataset",
    allow_overwrite=True,
    storage=SavedDatasetFileStorage(os.path.join(storage_file_path,"nyc_feast_dataset.parquet"))
)
