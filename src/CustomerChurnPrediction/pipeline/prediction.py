import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    def preprocess_data(self, data):
        # Convert data to DataFrame if it's not already
        if not isinstance(data, pd.DataFrame):
            columns=[
                'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                'EstimatedSalary', 'Satisfaction_Score', 'Point_Earned']
            data = pd.DataFrame(data, columns=columns)


        processed_data = pd.DataFrame(StandardScaler().fit_transform(data), columns=columns)

        return processed_data.values

    def predict(self, data):
        processed_data = self.preprocess_data(data)
        prediction = self.model.predict(processed_data)
        if prediction == 1:
            return "Customer is likely to churn"
        else:
            return "Customer will not churn"
