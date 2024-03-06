import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred 
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:

    def __init__(self,
                 EDUCATION:str,
                 MARRIAGE:str,
                 BILL_AMT2:float,
                 BILL_AMT4:float,
                 BILL_AMT5:float,
                 BILL_AMT6:float,
                 PAY_0:float,
                 PAY_2:float,
                 PAY_3:float,
                 PAY_4:float,
                 PAY_5:float,
                 PAY_6:float,
                ):
        self.EDUCATION=EDUCATION
        self.MARRIAGE=MARRIAGE
        self.BILL_AMT2=BILL_AMT2
        self.BILL_AMT4=BILL_AMT4
        self.BILL_AMT5=BILL_AMT5
        self.BILL_AMT6=BILL_AMT6
        self.PAY_0=PAY_0
        self.PAY_2=PAY_2
        self.PAY_3=PAY_3
        self.PAY_4=PAY_4
        self.PAY_5=PAY_5
        self.PAY_6=PAY_6
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
            'EDUCATION' :   [self.EDUCATION],
            'MARRIAGE'   : [self.MARRIAGE],
            'BILL_AMT2'  :  [self.BILL_AMT2],
            'BILL_AMT4'  : [self.BILL_AMT4],
            'BILL_AMT5'  : [self.BILL_AMT5],
            'BILL_AMT6' :   [self.BILL_AMT6],
            'PAY_0' :  [self.PAY_0],
            'PAY_2'  : [self.PAY_2],
            'PAY_3'  :[self.PAY_3],
            'PAY_4' : [self.PAY_4],
            'PAY_5' :  [self.PAY_5],
            'PAY_6'  :[self.PAY_6]

            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e, sys)
