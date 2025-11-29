import os
import sys
import numpy as np
from abc import ABC, abstractmethod
from typing import Union
from src.config import CONFIG
import pandas as pd

from src.logger import logging
from src.exception import MyException

class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreprocessStrategy(DataStrategy):
    """
    Data preprocessing strategy which preprocesses the data.
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes columns which are not required, fills missing values with median average values,
        and converts the data type to float.
        """
        try:
            df = data
            df.drop('Unnamed: 0', axis=1, inplace=True)
            df.drop_duplicates(inplace=True)

            to_drop = df[df['average_rain_fall_mm_per_year'].apply(self.isStr)].index
            df = df.drop(to_drop)

            df['average_rain_fall_mm_per_year'] = df['average_rain_fall_mm_per_year'].astype(np.float64)

            save_path = CONFIG["processed_data_path"]
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            logging.info(f"Successfully saved processed data to {save_path}")

            return df

        except Exception as e:
            logging.error("Error occurred in Processing data", exc_info=True)
            raise MyException(e, sys)
    
    def isStr(self, obj):
        try:
            float(obj)
            return False
        except:
            return True
 
class DataPreProcessing(DataStrategy):
    """
    Data cleaning class which preprocesses the data
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        logging.info("Initializing DataPreProcessing with given strategy")
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        logging.info("Handling data using the provided strategy")
        return self.strategy.handle_data(self.df)