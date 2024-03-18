import pandas as pd
import os
from CustomerChurnPrediction import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import type_of_target
import joblib #we can also use pickle to save the model here, but joblib is better than pickle.
from CustomerChurnPrediction.entity.config_entity import  ModelTrainerConfig



class ModelTrainer:
    def __init__(self,config: ModelTrainerConfig):
        self.config=config
    
    def train(self):
        train_data=pd.read_csv(self.config.train_data_path)
        test_data=pd.read_csv(self.config.test_data_path)

        train_x=train_data.drop([self.config.target_column],axis=1)
        test_x=test_data.drop([self.config.target_column],axis=1)
        train_y=train_data[[self.config.target_column]]
        test_y=test_data[[self.config.target_column]]


        train_x, train_y = check_X_y(train_x, train_y.values.ravel(), multi_output=True)
        test_x, test_y = check_X_y(test_x, test_y.values.ravel(), multi_output=True)

        # Check the type of target variable (classification or regression)
        target_type = type_of_target(train_y)
        if target_type not in ['binary', 'multiclass']:
            raise ValueError(f"Unsupported target variable type: {target_type}. Model supports binary or multiclass classification.")


        rfc=RandomForestClassifier(n_estimators=self.config.n_estimators,min_samples_split = 5,max_depth= 11, criterion = 'gini',  random_state=self.config.random_state)
        
        rfc.fit(train_x,train_y)


        joblib.dump(rfc,os.path.join(self.config.root_dir,self.config.model_name))

