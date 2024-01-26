import logging
import yaml

def configure_logger(config_path='parameters.yaml'):
    # Read the YAML configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Configuring logging using YAML values
    logging.basicConfig(level=config['logging']['level'],
                        format=config['logging']['format'],
                        handlers=[logging.FileHandler(config['logging']['log_file']),
                                   logging.StreamHandler()])
