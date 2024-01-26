import os
import json
import numpy as np
import logging
from modules.data_loader import read_data
from modules.read_config import read_config
from modules.logger_configurator import configure_logger

class ValidateDataSchema:
    def __init__(self, config):
        self.config = config
        self.schema_path = self.config['schema']
        self.schema_file_path = os.path.join(self.schema_path, "input_schema.json")
       
    def _get_schema(self):
        try:
            with open(self.schema_file_path,'r') as file:
                input_schema = json.load(file)
            return input_schema
        except Exception as e:
            logging.error(f"Error reading schema: {e}")
            raise
    
    def validate_data(self):
        try:
            data, _ = read_data(self.config['data']['remote'])
            input_schema = self._get_schema()
            
            for field in input_schema:
                if field not in data.columns:
                    logging.error(f'{field} not in remote data')
                    return False
            
                expected_dtype = input_schema[field]['data_type']
                if data[field].dtype != expected_dtype:
                    logging.error(f"{field} has incorrect data type, expected: {expected_dtype}, but got: {data[field].dtype}")
                    return False

            logging.info("Data validation passed!")
            return True
        
        except Exception as e:
            logging.error(f"Error during validation: {e}")
            return False

# if __name__=='__main__':
#     configure_logger()
#     config = read_config('parameters.yaml')
    
#     Validation_instance = ValidateDataSchema(config)
#     Validation_instance.validate_data()
