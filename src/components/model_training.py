import os
import sys
import joblib
import pandas as pd
from src.logger import logging
from src.exception import MyException
from abc import ABC, abstractmethod
from typing import Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,r2_score
from src.config import CONFIG

class ModelTrainingStrategy(ABC):

    @abstractmethod
    def handle_training(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class ModelTrainingConfig(ModelTrainingStrategy):

    def handle_training(self, data: pd.DataFrame):

        try:
            df = data

            col = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 
                'avg_temp', 'Area', 'Item', 'hg/ha_yield']
            df = df[col]
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            logging.info("Data preparation and splitting completed successfully.")

            ohe = OneHotEncoder(drop='first')
            scale = StandardScaler()

            preprocesser = ColumnTransformer(
                    transformers = [
                        ('StandardScale', scale, [0, 1, 2, 3]), 
                        ('OHE', ohe, [4, 5]),
                    ],
                    remainder='passthrough'
            )

            X_train_dummy = preprocesser.fit_transform(X_train)
            X_test_dummy = preprocesser.transform(X_test)
            preprocesser.get_feature_names_out(col[:-1])

            models = {
                    'lr':LinearRegression(),
                    'lss':Lasso(),
                    'Rid':Ridge(),
                    'Dtr':DecisionTreeRegressor()
                }
            
            for name, md in models.items():
                md.fit(X_train_dummy,y_train)
                y_pred = md.predict(X_test_dummy)

                print(f"{name} : mae : {mean_absolute_error(y_test,y_pred)} score : {r2_score(y_test,y_pred)}")

            dtr = DecisionTreeRegressor()
            dtr.fit(X_train_dummy,y_train)
            dtr.predict(X_test_dummy)

            model_path = CONFIG["model"]
            processer = CONFIG["processor"]
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.makedirs(os.path.dirname(processer), exist_ok=True)
            joblib.dump(dtr,open(processer,'wb'))
            joblib.dump(preprocesser,open(model_path,'wb'))

        except Exception as e:
            logging.error("Error occurred while training", exc_info=True)
            raise MyException(e, sys)

class ModelTraining(ModelTrainingStrategy):
    def __init__(self, data: pd.DataFrame, strategy: ModelTrainingStrategy):
        self.strategy = strategy
        self.df = data

    def handle_training(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_training(self.df)