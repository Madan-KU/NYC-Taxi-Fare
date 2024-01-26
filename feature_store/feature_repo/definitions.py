from feast import Entity,Field, FeatureView, FileSource, ValueType
from feast.types import Float32, Float64, Int64,String


# Feast feature and entity definitions
trip_id = Entity(name="trip_id", value_type=ValueType.INT64, description="Trip or Request identifier")


file_source_X = FileSource(
    name="X_training_data",
    path=r"/mnt/hgfs/DS/NYC MLOPS/feature_store/feature_repo/data/X.parquet",
    event_timestamp_column="event_timestamp",
    # created_timestamp_column="created",
)

feature_view_X = FeatureView(
    name="trip_features_X",
    entities=[trip_id],
    ttl=None,
    schema=[
        Field(name="trip_miles", dtype= Float64),
        Field(name="trip_time", dtype= Float64),
        Field(name="request_datetime_hour", dtype= Int64),
        Field(name="request_datetime_day", dtype= String),
        Field(name="duration_minutes", dtype= Float64),
        Field(name="wait_time_minutes", dtype=Float64),
        Field(name="service_time_minutes", dtype= Float64),
        Field(name="on_scene_datetime_hour", dtype= Int64),
        Field(name="on_scene_datetime_day", dtype= String),
        Field(name="pickup_datetime_hour", dtype= Int64),
        Field(name="pickup_datetime_day", dtype= String),
        Field(name="dropoff_datetime_hour", dtype= Int64),
        Field(name="dropoff_datetime_day", dtype=String),
        Field(name="average_speed", dtype= Float64),
        Field(name="event_timestamp", dtype= String)
    ],

    source=file_source_X,
)




file_source_y = FileSource(
    name="y_training_data",
    path=r"/mnt/hgfs/DS/NYC MLOPS/feature_store/feature_repo/data/y.parquet",
    event_timestamp_column="event_timestamp",
    # created_timestamp_column="created",
)

feature_view_y = FeatureView(
    name="trip_features_y",
    entities=[trip_id],
    ttl=None,
    schema=[
        Field(name="driver_pay", dtype= Float64),
        Field(name="event_timestamp", dtype= String)
    ],

    source=file_source_y,
)


