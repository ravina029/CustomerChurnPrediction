import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

        # Define the subcategories for one-hot encoding (drop_first=True makes this unnecessary)
        # self.cat_subcategories = {
        #     'Geography': 3,
        #     'Gender': 2,
        #     'Card_Type': 4
        # }

        # Define the expected subcategories for one-hot encoding (not strictly required)
        self.cat_values = {
            'Geography': ['France', 'Spain', 'Germany'],
            'Card_Type': ['DIAMOND', 'GOLD', 'SILVER', 'PLATINUM'],
            'Gender': ['Female', 'Male']
        }

    def preprocess_data(self, data):
        # Convert data to DataFrame if it's not already
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=[
                'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                'EstimatedSalary', 'Satisfaction_Score', 'Point_Earned', 'Geography', 'Gender', 'Card_Type'
            ])

        # Handle categorical variables using one-hot encoding (dropping first category)
        cat_encoded = pd.get_dummies(data[['Geography', 'Gender', 'Card_Type']], prefix_sep='__', drop_first=True)

        # Standard scaling on numerical columns
        num_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                           'EstimatedSalary', 'Satisfaction_Score', 'Point_Earned']
        num_scaled = pd.DataFrame(StandardScaler().fit_transform(data[num_columns]), columns=num_columns)

        # Combine numerical and categorical data
        processed_data = pd.concat([num_scaled, cat_encoded], axis=1)

        return processed_data

    def predict(self, data):
        processed_data = self.preprocess_data(data)
        prediction = self.model.predict(processed_data)
        if prediction == 1:
            return "Customer is likely to churn"
        else:
            return "Customer will not churn"
