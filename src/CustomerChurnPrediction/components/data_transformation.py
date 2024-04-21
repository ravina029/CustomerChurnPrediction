import os
from CustomerChurnPrediction import logger
from sklearn.model_selection import train_test_split
import pandas as pd 
from CustomerChurnPrediction.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import LabelEncoder,StandardScaler
import imblearn
from imblearn.over_sampling import SMOTE
#from imblearn.over_sampling import RandomOverSampler

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_splitting(self):
        df = pd.read_csv(self.config.data_path)
        
        # Separate the target variable 'Exited'
        y = df['Exited']
        X=df.drop(['Exited'], axis=1)
        
        numr = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
        catg =['Geography', 'Gender', 'NumOfProducts', 'IsActiveMember', 'sufficient_balance', 'is_CreditScore_low']
        
        numerical_transformer = StandardScaler()
        categoricals= []

        # Apply LabelEncoder to each categorical column
        for column in catg:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            categoricals.append((column, le))

        
        X[numr] = numerical_transformer.fit_transform(X[numr])   # Apply StandardScaler to numerical columns
        transformed_df = pd.concat([X[numr], X[catg]], axis=1)   # Concatenate numerical and encoded categorical columns
        
        
        smote = SMOTE(sampling_strategy='minority')    # Apply SMOTE to training set
        X_balanced, y_balanced= smote.fit_resample(transformed_df, y)

        df_transformed = pd.concat([X_balanced, y_balanced], axis=1)   # Concatenate the encoded categorical, scaled numerical and target y.
        train_df, test_df = train_test_split(df_transformed, test_size=0.20, random_state=2)   # Split into training and testing sets
        
        
        #train_resampled = pd.concat([pd.DataFrame(X_train, columns=train.drop(columns=['Exited']).columns), pd.DataFrame(y_train, columns=['Exited'])], axis=1)
        
        # Save train and test sets to CSV
        train_df.to_csv(os.path.join(self.config.root_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(self.config.root_dir, 'test.csv'), index=False)

        logger.info("Data has been split into training and test sets.")
        logger.info(f"Training set shape: {train_df.shape}")
        logger.info(f"Test set shape: {test_df.shape}")
        print("columns of X:",X.columns)
        print("columns of df_transformed:",df_transformed.columns)