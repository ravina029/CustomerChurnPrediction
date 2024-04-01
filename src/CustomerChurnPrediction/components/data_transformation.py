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
        df_transformed = pd.concat([transformed_df, y], axis=1)   # Concatenate the encoded categorical, scaled numerical and target y.
        
        train, test = train_test_split(df_transformed, test_size=0.25, random_state=42)   # Split into training and testing sets
        smote = SMOTE(sampling_strategy='minority')    # Apply SMOTE to training set
        X_train, y_train= smote.fit_resample(train.drop(columns=['Exited']), train['Exited'])
        
        train_resampled = pd.concat([pd.DataFrame(X_train, columns=train.drop(columns=['Exited']).columns), pd.DataFrame(y_train, columns=['Exited'])], axis=1)
        
        # Save train and test sets to CSV
        train_resampled.to_csv(os.path.join(self.config.root_dir, 'train.csv'), index=False)
        test.to_csv(os.path.join(self.config.root_dir, 'test.csv'), index=False)

        logger.info("Data has been split into training and test sets.")
        logger.info(f"Training set shape: {train_resampled.shape}")
        logger.info(f"Test set shape: {test.shape}")
        print("columns of X:",X.columns)
        print("columns of df_transformed:",df_transformed.columns)
        print("Shape of X_train and y_train after appliing smote", X_train.shape,y_train.shape)
        print("columns of train_resample:",train_resampled.columns)