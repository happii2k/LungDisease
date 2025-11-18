from dataclasses import dataclass
from torch.utils.data.dataloader import DataLoader
from typing import Optional
import joblib
from torch.utils.data import DataLoader , Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os


@dataclass
class DataIngestionArtifact:
    train_file_path: str

    test_file_path: str


@dataclass
class DataTransformationArtifact:
    
    
    train_transform_file_path =  r"artifacts\\data_transformation\\train_transforms.pkl"

    test_transform_file_path  = r"artifacts\\data_transformation\\test_transforms.pkl"
    
    transformed_train_obj : DataLoader =  joblib.load(train_transform_file_path)

    transformed_test_obj: DataLoader = joblib.load(test_transform_file_path)
    
    path = r"artifacts\data_ingestion\data\data"
   

    test_data : Dataset = ImageFolder(
                os.path.join(path ,"test") , transform= transformed_test_obj

            )
            
            
    transformed_test_object : DataLoader = DataLoader(
                test_data , batch_size=2 , shuffle=False , pin_memory=True
            )
    

    train_data : Dataset = ImageFolder(
                os.path.join(path , "train") , transform= transformed_train_obj
            )
            
            
    transformed_train_object : DataLoader = DataLoader(
                train_data , batch_size=2 , shuffle=False , pin_memory=True
            )


@dataclass
class ModelTrainerArtifact:
    trained_model_path: str = r"artifacts\model_training\model.pt" 


@dataclass
class ModelEvaluationArtifact:
    model_accuracy: float


@dataclass
class ModelPusherArtifact:
    bentoml_model_name: str

    bentoml_service_name: str
