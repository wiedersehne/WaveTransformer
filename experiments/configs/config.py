import argparse
from pathlib import Path
import yaml
import os

# We only specify the yaml file from argparse and handle rest
parser = argparse.ArgumentParser(description=__doc__)
# Get default yaml file path
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
default_path = FILE_PATH + r'/default.yaml'
parser.add_argument("-f", "--config_file", default=default_path, help="Configuration file to load.")
ARGS = parser.parse_args()

# Load the yaml file here
with open(ARGS.config_file, 'r') as f:
    config = yaml.load(f)
print(f"Loaded configuration file {ARGS.config_file}")


def extern(func):
    """Wraps keyword arguments from configuration."""
    def wrapper(*args, **kwargs):
        """Injects configuration keywords."""
        # We get the file name in which the function is defined, e.g.: train.py
        fname = Path(func.__globals__['__file__']).name
        # Then we extract arguments corresponding to the function name, e.g.: train.py -> load_data
        conf = config[fname][func.__name__]
        # And update the keyword arguments with any specified arguments. If it isn't specified then the default holds
        conf.update(kwargs)
        return func(*args, **conf)
    return wrapper
