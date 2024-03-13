from CustomerChurnPrediction import logger

from CustomerChurnPrediction.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from CustomerChurnPrediction.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from CustomerChurnPrediction.pipeline.stage_03_data_transformation import  DataTransformationTrainingPipeline
from CustomerChurnPrediction.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
#from CustomerChurnPrediction.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline


STAGE_NAME="Data ingestion stage"

try:
    logger.info(f">>>>>>> stage{STAGE_NAME} started <<<<<<<<")
    data_ingestion=DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx=============x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME='Data validation stage'

try:
        logger.info(f">>>>>>> stage{STAGE_NAME} started <<<<<<<<")
        obj=DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\n x=============x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME='Data Transformation stage'
try:
        logger.info(f">>>>>>> stage{STAGE_NAME} started <<<<<<<<")
        obj=DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\n x=============x")
except Exception as e:
        logger.exception(e)
        raise e    



STAGE_NAME='Data Transformation stage'
try:
        logger.info(f">>>>>>> stage{STAGE_NAME} started <<<<<<<<")
        obj=ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\n x=============x")
except Exception as e:
         logger.exception(e)
         raise e       