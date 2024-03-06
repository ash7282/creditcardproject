import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler,RobustScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def get_data_transformation_object(self):

        try:
            logging.info("Data Transformation initiated...")
            categorical_cols=['EDUCATION','MARRIAGE']
            numerical_cols=['AGE','BILL_AMT2',
                            'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6','PAY_0','PAY_0','PAY_3', 'PAY_4',
                            'PAY_5','PAY_6']
            EDUCATION =['Graduate','University','High school','other']
            MARRIAGE =['Married', 'Single','other']
                        

            logging.info("Pipeline Initiated....")

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler()),
                    ('robustscaler',RobustScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder', OrdinalEncoder(categories=[EDUCATION, MARRIAGE])),
                ('scaler', StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer(transformers=[
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            logging.info("Pipeline Completed..")

            return preprocessor
        
        except Exception as e:
            logging.info("Error in data Transformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj=self.get_data_transformation_object()


            target_column_name='default.payment.next.month'
            drop_columns=[target_column_name,'default.payment.next.month']

            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df)

            logging.info(
                "Applying preprocessing object on training and testing datasets.")

                          
            train_arr = np.c_[input_feature_train_arr,
                              np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,
                             np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.info("Exception occured in initiate_data_transformation")
            raise CustomException(e, sys)








