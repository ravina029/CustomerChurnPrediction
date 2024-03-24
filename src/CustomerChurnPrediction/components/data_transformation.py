import os
from CustomerChurnPrediction import logger
from sklearn.model_selection import train_test_split
import pandas as pd 
from CustomerChurnPrediction.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import StandardScaler
import imblearn
from imblearn.over_sampling import RandomOverSampler

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_splitting(self):
        df = pd.read_csv(self.config.data_path)
        
        # Separate the target variable 'Exited'
        y = df['Exited']
        X=df.drop(['Geography', 'Gender', 'Card Type','Exited'], axis=1)
        print("columns of X:",X.columns)
        
        #cat_columns = ['Geography', 'Gender', 'Card Type']
        num_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',"HasCrCard",'IsActiveMember','EstimatedSalary', 'Satisfaction Score', 'Point Earned']
        
        # Handle categorical variables using one-hot encoding
        #df_cat_encoded = pd.get_dummies(df[cat_columns], dtype=int,drop_first=True)

        # Standard scaling on numerical columns
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(X[num_columns])
        df_scaled = pd.DataFrame(df_scaled, columns=num_columns)

        # Concatenate the encoded categorical, scaled numerical and target y.
        df_transformed = pd.concat([df_scaled, y], axis=1)

        # Split into training and testing sets
        train, test = train_test_split(df_transformed, test_size=0.25, random_state=42)
        
        # Apply oversampling to training set
        ros = RandomOverSampler(random_state=0)
        X_train, y_train = ros.fit_resample(train.drop(columns=['Exited']), train['Exited'])
        train_resampled = pd.concat([pd.DataFrame(X_train, columns=train.drop(columns=['Exited']).columns), pd.DataFrame(y_train, columns=['Exited'])], axis=1)
        print("columns of train_resample:",train_resampled.columns)
        # Save train and test sets to CSV
        train_resampled.to_csv(os.path.join(self.config.root_dir, 'train.csv'), index=False)
        test.to_csv(os.path.join(self.config.root_dir, 'test.csv'), index=False)

        logger.info("Data has been split into training and test sets.")
        logger.info(f"Training set shape: {train_resampled.shape}")
        logger.info(f"Test set shape: {test.shape}")


