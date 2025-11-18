import sys

from typing import Tuple

import torch

from torch.nn import CrossEntropyLoss , Module
from torch.optim import SGD , Optimizer 
from torch.utils.data import DataLoader

from src.entity.artifacts_config import ModelEvaluationArtifact , ModelTrainerArtifact , DataTransformationArtifact
from src.entity.config import ModelEvaluationConfig
from src.exception import XRayException
from src.logger import logging
from src.dl.CustomNN import Net

class ModelEvaluation:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact , model_evaluation_config : ModelEvaluationConfig , model_trainer_artifact : ModelTrainerArtifact ):
        self.model_evalutation_config = model_evaluation_config
        self.model_trainer_artifact = model_trainer_artifact
        self.data_transformation_artifact = data_transformation_artifact
  
   
    def configuration(self )->tuple[DataLoader , Module , float , Optimizer]:
        logging.info("Entered the configuration method of Model evaluation class")

        try:
            test_dataloader : DataLoader = (self.data_transformation_artifact.transformed_test_object)

            model : Module = Net()

            model : Module = torch.load(self.model_trainer_artifact.trained_model_path , weights_only=False)

            model.to(self.model_evalutation_config.device)

            cost : Module = CrossEntropyLoss()

            optimizer : Optimizer = SGD(
                model.parameters() , **self.model_evalutation_config.optimizer_params
            )

            model.eval()

            logging.info("Exited the configuration method of Model evaluation class")

            return test_dataloader, model, cost, optimizer
        except Exception as e:
            raise XRayException(e, sys)
    def test_net(self) -> float :
        logging.info("Entered the test_net method of Model evaluation class")

        try:
            test_dataloader, net, cost, _ = self.configuration()

            with torch.no_grad():
                holder = []

                for _ , data in enumerate(test_dataloader):
                    images = data[0].to(self.model_evalutation_config.device)
                    labels = data[1].to(self.model_evalutation_config.device)

                    output = net(images)

                    loss = cost(output , labels)

                    predictions = torch.argmax(output , 1)

                    for i in zip(images , labels , predictions):
                        h = list(i)
                        holder.append(h)

                    logging.info(
                        f"Actual_Labels : {labels}     Predictions : {predictions}     labels : {loss.item():.4f}"
                    )
                    self.model_evalutation_config.test_loss += loss.item()

                    self.model_evalutation_config.test_accuracy += ((predictions == labels).sum().item())

                    self.model_evalutation_config.total_batch +=1 

                    self.model_evalutation_config.total += labels.size(0)

                    
            
            accuracy = (
                self.model_evalutation_config.test_accuracy/self.model_evalutation_config.total
            )*100
            logging.info("Exited the test_net method of Model evaluation class")

            return accuracy

        except Exception as e:
            raise XRayException(e, sys)
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:

        logging.info(
            "Entered the initiate_model_evaluation method of Model evaluation class"
        )

        try:
            accuracy = self.test_net()
            model_evaluation_artifact: ModelEvaluationArtifact = (
                ModelEvaluationArtifact(model_accuracy=accuracy)
            )

            logging.info(
                "Exited the initiate_model_evaluation method of Model evaluation class"
            )

            return model_evaluation_artifact

        except Exception as e:
            raise XRayException(e, sys)



