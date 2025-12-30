import pandas as pd
import os,sys
from zenml import step
from src.logger import logging
from src.exception import MyException
from src.config import CONFIG

@step
def load_data() -> pd.DataFrame:
    path = CONFIG["data"]["local_data_file"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if not os.path.exists(path):
        url = CONFIG["data"]["source_URL"]
        df = pd.read_csv(url)
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    
    return df

@step
def save_data(df: pd.DataFrame) -> str:
    path = CONFIG["data"]["local_data_file"]
    df.to_csv(path, index=False)
    return path


