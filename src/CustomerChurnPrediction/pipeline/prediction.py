import joblib
import numpy as np
import pandas as pd
from pathlib import Path



class PredictionPipeline:
    def __init__(self):
        self.model=joblib.load(Path('artifacts/model_trainer/model.joblib'))


    def predict(self,data):
        prediction=self.model.predict(data)
        if prediction==1:
            return "Customer is likely to churn"
        else:
            return "Customer will not churn" 


        

        #return prediction
    