
from dataclasses import dataclass 
from pathlib import Path

@dataclass(frozen=True)  #this is not python class but dataclass, here you can define the veriables without using self keyword.
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)  #this is not python class but dataclass, here you can define the veriables without using self keyword.
class DataValidationConfig:
    root_dir:Path
    STATUS_FILE:str
    unzip_data_dir:Path
    all_schema:dict

@dataclass(frozen=True)  #this is not python class but dataclass, here you can define the veriables without using self keyword.
class DataTransformationConfig:
    root_dir:Path
    data_path: Path


@dataclass(frozen=True)  #this is not python class but dataclass, here you can define the veriables without using self keyword.
class ModelTrainerConfig:
    root_dir:Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    n_estimators: int
    min_samples_split: int
    max_depth: int
    criterion: str
    random_state: int
    target_column: str

@dataclass(frozen=True)  #this is not python class but dataclass, here you can define the veriables without using self keyword.
class ModelEvaluationConfig:
    root_dir:Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name:Path
    target_column: str
    mlflow_uri: str