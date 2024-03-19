# Bank Customer Churn Prediction Project

Bank customer churn prediction involves leveraging data analysis and machine learning techniques to forecast which customers are likely to terminate their relationship with a bank. By accurately predicting churn, banks can take proactive measures to retain valuable customers, improve customer satisfaction, and optimize their operations.

## Key Components

- **Data Collection**: Gathering data about bank customers, including demographic information, account balances, credit card usage, customer service interactions etc.
- **Feature Engineering**: Extracting relevant features from the collected data, such as customer age, transaction activity, average balance, No of products, activity and complain.
- **Model Training**: Developing classification model using machine learning algorithms such as random forest classifier to predict the likelihood of churn based on the extracted features.
- **Evaluation**: Assessing the performance of the churn prediction models using metrics like accuracy, precision, recall, and area under the ROC curve (AUC). Achieved accuracy over 83.2% and ROC AUC score=75.8%.
- **Deployment**: Deploying the model on cloud or Bank system.

## Benefits

- **Customer Retention**: Identifying customers at risk of churn enables banks to implement personalized retention strategies, such as offering tailored financial products, rewards, or incentives.
- **Risk Management**: Proactively managing customer churn helps mitigate revenue loss and minimize the impact on the bank's profitability.
- **Enhanced Customer Experience**: Anticipating and addressing customer needs and concerns before they churn can lead to improved satisfaction and loyalty.

## Example Use Cases

- Predicting which bank customers are likely to close their accounts or switch to another bank.
- Forecasting customer attrition for specific banking products, such as credit cards, loans, or investment accounts.
- Identifying high-value customers who are at risk of churn and prioritizing retention efforts accordingly.




## Workflows
1. Update config.yaml
2. Update Schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. update the components
7. Update the pipeline
8. Update main.py
9. Update the app.py


# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/ravina029/CustomerChurnPrediction
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n newvenv python=3.10 -y
```

```bash
conda activate newvenv
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```



## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/ravina029/CustomerChurnPrediction.mlflow \
MLFLOW_TRACKING_USERNAME=ravina029 \
MLFLOW_TRACKING_PASSWORD=221d4e3a527ff8b9aef06e059d7efc4e89963e11 \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/ravina029/CustomerChurnPrediction.mlflow 

export MLFLOW_TRACKING_USERNAME=ravina029

export MLFLOW_TRACKING_PASSWORD=221d4e3a527ff8b9aef06e059d7efc4e89963e11

```



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 161774582158.dkr.ecr.us-east-1.amazonaws.com/churnapp

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = eu-north-1

    AWS_ECR_LOGIN_URI = 161774582158.dkr.ecr.us-east-1.amazonaws.com/churnapp

    ECR_REPOSITORY_NAME = churnpredict




## About MLflow 
MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & tagging your model


# cloud Deployement url: 
https://18.233.62.142:5000/

# Link for Demo video of the WebApp: 
https://www.youtube.com/watch?v=2Vjy56Mvm-o

