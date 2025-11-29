import sys
from typing import Any
import pandas as pd
from zenml import pipeline, step
from src.constants import *
from src.logger import logging
from src.exception import MyException
from src.components.data_ingestion import load_data,save_data
from src.components.data_preprocess import DataPreProcessing, DataPreprocessStrategy
from src.components.model_training import ModelTrainingConfig, ModelTraining
from src.config import CONFIG

@step
def ingest_data() -> str:
    try:
        logging.info(f">>>>>> stage {INGESTION_STAGE_NAME} started <<<<<<")
        df = load_data()

        # 2) Save locally
        local_path = save_data(df)
        logging.info(f">>>>>> stage {INGESTION_STAGE_NAME} completed <<<<<<\n\nx==========x")
        return CONFIG["data_path"]
    except Exception as e:
        logging.exception(f"Error during ingestion: {e}")
        raise MyException(e, sys)

@step
def preprocess_data(data_path: str) -> pd.DataFrame:
    try:
        raw_data = pd.read_csv(data_path)
        strategy = DataPreprocessStrategy()
        processor = DataPreProcessing(data=raw_data, strategy=strategy)
        processed_df = processor.handle_data()
        return processed_df
    except Exception as e:
        raise MyException(e, sys)

@step
def train_model(processed_df: pd.DataFrame) -> Any:
    try:
        logging.info(">>>>>Model Training Started...<<<<<")
        model_training_strategy = ModelTraining(data=processed_df, strategy=ModelTrainingConfig())
        trained_model = model_training_strategy.handle_training()
        logging.info(">>>>>Model Training Completed<<<<<\n")
        return trained_model
    except MyException as e:
        logging.exception(e, sys)
        raise e


@pipeline(enable_cache=False)
def training_pipeline():
    data_path = ingest_data()
    processed_df = preprocess_data(data_path)
    train_model(processed_df)


# ------------------------
# Entry Point
# ------------------------

if __name__ == "__main__":
    try:
        training_pipeline()
    except Exception as e:
        logging.error("Pipeline failed")
        raise e