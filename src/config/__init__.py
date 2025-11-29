from box import Box
from pathlib import Path
import yaml

def load_config(path:Path=Path('config.yaml')):
    with open(path,'r') as f:
        return Box(yaml.safe_load(f))
    
CONFIG=load_config()