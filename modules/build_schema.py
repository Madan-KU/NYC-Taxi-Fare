import os
import json
import numpy as np
import logging
from modules.data_loader import read_data
from modules.read_config import read_config
from modules.logger_configurator import configure_logger

# class SchemaBuilder:
#     def __init__(self, config):
#         self.config = config
#         # self.schema_path = self.config['schema']
#         # self.schema_file_path = os.path.join(self.schema_path, "input_schema.json")
#         # self.schema_file_path = os.path.join(self.schema_path, "input_schema.json")

        
#     def _generate_schema(self, df):
#         schema = {}
#         df_describe = df.describe(include='all')  

#         for column in df.columns:
#             try:
#                 data_type = str(df[column].dtype)
#                 min_value = df_describe[column]['min'] if 'min' in df_describe[column] else None
#                 max_value = df_describe[column]['max'] if 'max' in df_describe[column] else None

#                 if 'datetime' in data_type.lower():
#                     schema[column] = {'data_type': data_type }

#                 elif data_type in ('object','category'):
#                     top_value = df_describe[column]['top']  
#                     freq_value = df_describe[column]['freq'] 
#                     unique_values_count = df[column].nunique()  

#                     schema[column] = {
#                         'data_type': data_type,
#                         'unique_values_count': unique_values_count,
#                         'most_frequent_value': top_value,
#                         'frequency_of_most_frequent': freq_value
#                     }

#                     # if unique_values_count <= 5: # IF unique values less than 5, then add to schema
#                     #     schema[column]['unique_values'] = df[column].unique().tolist()
#                 else:
#                     schema[column] = {
#                         'data_type': data_type,
#                         'min_value': min_value,
#                         'max_value': max_value
#                     }
#             except KeyError as ke:
#                 logging.error(f"Column not found: {ke}")
#         return schema


#     def _write_schema_to_file(self, schema):

#         if not os.path.exists(os.path.dirname(self.schema_file_path)):
#             os.makedirs(os.path.dirname(self.schema_file_path))
        
#         with open(self.schema_file_path, "w+") as file:
#             json.dump(schema,
#                         file,
#                         default=lambda o: int(o) if isinstance(o, (np.int64, np.int32)) else o,
#                         indent=4)

#     def generate_and_save_schema(self,df_path):
#         try:
#             df, file = read_data(df_path)
#             schema = self._generate_schema(df)
#             self._write_schema_to_file(schema)
            
#             logging.info(f"Schema written to '{self.schema_file_path}'")
#         except Exception as e:
#             logging.error(f"Unable to write input schema. Error: {e}")

import os
import json
import numpy as np
import logging
from modules.data_loader import read_data
from modules.logger_configurator import configure_logger

class SchemaBuilder:
    def __init__(self):
        pass
        
    def _generate_schema(self, df):
        schema = {}
        df_describe = df.describe(include='all')

        for column in df.columns:
            try:
                data_type = str(df[column].dtype)
                min_value = df_describe[column]['min'] if 'min' in df_describe[column] else None
                max_value = df_describe[column]['max'] if 'max' in df_describe[column] else None

                if 'datetime' in data_type.lower():
                    schema[column] = {'data_type': data_type }

                elif data_type in ('object','category'):
                    top_value = df_describe[column]['top']  
                    freq_value = df_describe[column]['freq'] 
                    unique_values_count = df[column].nunique()  

                    schema[column] = {
                        'data_type': data_type,
                        'unique_values_count': unique_values_count,
                        'most_frequent_value': top_value,
                        'frequency_of_most_frequent': freq_value
                    }
                else:
                    schema[column] = {
                        'data_type': data_type,
                        'min_value': min_value,
                        'max_value': max_value
                    }
            except KeyError as ke:
                logging.error(f"Column not found: {ke}")
        return schema

    def _write_schema_to_file(self, schema, schema_file_path):

        if not os.path.exists(os.path.dirname(schema_file_path)):
            os.makedirs(os.path.dirname(schema_file_path))
        
        with open(schema_file_path, "w+") as file:
            json.dump(schema,
                        file,
                        default=lambda o: int(o) if isinstance(o, (np.int64, np.int32)) else o,
                        indent=4)

    def generate_and_save_schema(self, df_path):
        try:
            df, filename = read_data(df_path)
            schema = self._generate_schema(df)
            schema_file_name= filename+"_schema.json"
            
            schema_file_path = os.path.join(df_path,schema_file_name)
            self._write_schema_to_file(schema, schema_file_path)
            
            logging.info(f"Schema written to '{schema_file_path}'")
        except Exception as e:
            logging.error(f"Unable to write input schema. Error: {e}")


if __name__ == "__main__":
    configure_logger()
    config = read_config('parameters.yaml')
    df_path= config['data']['feature_engineered']
    schema_instance = SchemaBuilder()
    schema_instance.generate_and_save_schema(df_path)
