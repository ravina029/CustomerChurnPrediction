import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, f1_score
from urllib.parse import urlparse
from sklearn.utils.validation import check_X_y
#from sklearn.utils.multiclass import type_of_target
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from CustomerChurnPrediction.entity.config_entity import ModelEvaluationConfig
from CustomerChurnPrediction.utils.common import save_json
from pathlib import Path




class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
        self.config=config
        

    def eval_metrics(self,actual,pred):
            accuracy = accuracy_score(actual, pred)
            precision = precision_score(actual, pred)
            recall = recall_score(actual, pred)
            roc_auc = roc_auc_score(actual, pred)
            f_Score= f1_score(actual,pred)
            return accuracy,precision,recall,roc_auc,f_Score
        

    def log_into_mlflow(self):
            test_data=pd.read_csv(self.config.test_data_path)
            model=joblib.load(self.config.model_path)

            test_x=test_data.drop([self.config.target_column],axis=1)
            test_y=test_data[[self.config.target_column]]
            
            #test_x, test_y = check_X_y(test_x, test_y.values.ravel(), multi_output=True)


            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_score=urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                predicted_qualities=model.predict(test_x)
                #y_prob = model.predict_proba(test_x)[:, 1]
                #roc_auc = roc_auc_score(test_y, y_prob) #as our data is highly imbalanced, therefore using this method using y_prob.
                (accuracy,precision,recall,roc_auc,f_Score)=self.eval_metrics(test_y,predicted_qualities)
                
                #saving metrics as local
                scores = {"accuracy": accuracy, "precision_score": precision, "Recall_score": recall, "Roc_Auc_score": roc_auc,"f1_score":f_Score}
                print(scores)
                save_json(path=Path(self.config.metric_file_name), data=scores)
                mlflow.log_params(self.config.all_params)

                
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision_score", precision)
                mlflow.log_metric("Recall_score", recall)
                mlflow.log_metric("Roc_Auc_score",roc_auc)
                

                #Model registry doesn't work with file store
                if tracking_url_type_score!='file':
                    #Regitster the model
                    #There are other ways to use the Model registry, which depends on the use case,
                    #please refer to the doc for more information:
                    #https://mlflow.org/docs/latest/models.html
                    mlflow.sklearn.log_model(model,"model", registered_model_name='RandomForestClassifier')
                else:
                    mlflow.sklearn.log_model(model,'model') 



