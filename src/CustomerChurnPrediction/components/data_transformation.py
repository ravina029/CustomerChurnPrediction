import os
from CustomerChurnPrediction import logger
from sklearn.model_selection import train_test_split
import pandas as pd 
from CustomerChurnPrediction.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import LabelEncoder,StandardScaler
import imblearn
from imblearn.over_sampling import SMOTE
#from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours


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
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(transformed_df, y, test_size=0.3, stratify=y,random_state=42)
        
        
        smote = SMOTE(sampling_strategy='minority')    # Apply SMOTE to training set
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        train_df = pd.concat([X_train_balanced, y_train_balanced], axis=1)   # Concatenate the encoded categorical, scaled numerical and target y.
        test_df = pd.concat([X_test, y_test], axis=1)   # Concatenate the test set with target y.

        # Save train and test sets to CSV
        train_df.to_csv(os.path.join(self.config.root_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(self.config.root_dir, 'test.csv'), index=False)

        logger.info("Data has been split into training and test sets.")
        logger.info(f"Training set shape: {train_df.shape}")
        logger.info(f"Test set shape: {test_df.shape}")
