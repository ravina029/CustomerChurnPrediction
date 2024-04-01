import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler,LabelEncoder

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    def preprocess_data(self, data):
        # Convert data to DataFrame if it's not already
        if not isinstance(data, pd.DataFrame):
            columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary','Geography', 'Gender', 
                     'NumOfProducts', 'IsActiveMember','sufficient_balance', 'is_CreditScore_low']
            data = pd.DataFrame(data, columns=columns)

        numr = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
        catg =['Geography', 'Gender', 'NumOfProducts', 'IsActiveMember', 'sufficient_balance', 'is_CreditScore_low']
        
        numerical_transformer = StandardScaler()
        categoricals= []

        # Apply LabelEncoder to each categorical column
        for column in catg:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            categoricals.append((column, le))

        
        data[numr] = numerical_transformer.fit_transform(data[numr])   # Apply StandardScaler to numerical columns
        processed_data = pd.concat([data[numr], data[catg]], axis=1) 
        #processed_data = pd.DataFrame(StandardScaler().fit_transform(data), columns=columns)

        #return processed_data.values
        return processed_data.values

    def predict(self, data):
        processed_data = self.preprocess_data(data)
        prediction = self.model.predict(processed_data)
        if prediction == 1:
            return "Customer is likely to churn"
        else:
            return "Customer will not churn"
