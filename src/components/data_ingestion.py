import pandas as pd
import os,sys
from zenml import step
from src.logger import logging
from src.exception import MyException
from src.config import CONFIG

@step
def load_data()->pd.DataFrame:
    config=CONFIG['data']
    dataset_url=config['source_URL']
    df=pd.read_csv(dataset_url)
    return df

@step
def save_data(df:pd.DataFrame)->str:
    config=CONFIG['data']
    local_csv_path=config['local_data_file']
    os.makedirs(os.path.dirname(local_csv_path),exist_ok=True)
    df.to_csv(local_csv_path,index=False)
    return local_csv_path


