import os
import sys

import bentoml
import joblib
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR , _LRScheduler
from tqdm import tqdm

from src.constant import *
from src.exception import XRayException
from src.logger import logging
from src.dl.CustomNN import Net
from src.entity.config import ModelTrainerConfig
from src.entity.artifacts_config import DataTransformationArtifact , ModelTrainerArtifact

class ModelTrainer:
    def __init__(self , model_trainer_config : ModelTrainerConfig , data_transformation_artifact :  DataTransformationArtifact):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact  = data_transformation_artifact
        self.model : Module = Net()

    
    def train(self , optimizer : Optimizer ) -> None :

        logging.info("Entered the train method of Model trainer class")

        try:
            self.model.train()
            pbar = tqdm(self.data_transformation_artifact.transformed_train_object)
            correct : int = 0
            processed = 0

            for batch_idx , (data  ,target) in enumerate(pbar):
                data , target = data.to(self.model_trainer_config.device) , target.to(self.model_trainer_config.device)
                optimizer.zero_grad()
                y_pred = self.model(data)

                loss = F.nll_loss(y_pred , target)

                loss.backward()

                optimizer.step()

                pred = y_pred.argmax(dim=1 , keepdim = True)

                correct += pred.eq(target.view_as(pred)).sum().item()

                processed += len(data)

                pbar.set_description(
                    desc=f"Loss={loss.item()} Batch_id= {batch_idx} Accuraacy = {100*correct/processed:0.2f}"
                )
            
            logging.info("Exited the train method of Model trainer class")

        except Exception as e:
            raise XRayException(e, sys)
    

    def test(self)-> None:
        try:
            logging.info("Entered the test method of Model trainer class")
            self.model.eval()

            test_loss : float= 0

            correct : int = 0

            with torch.no_grad():
                for (data , target) in self.data_transformation_artifact.transformed_test_object:
                    data , target = data.to(DEVICE) , target.to(DEVICE)
                    output = self.model(data)

                    test_loss += F.nll_loss(output , target , reduction="sum").item()

                    pred = output.argmax(dim = 1 , keepdim = True)

                    correct += pred.eq(target.view_as(pred)).sum().item()

                    test_loss /= len(self.data_transformation_artifact.transformed_test_object.dataset)

                print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                        test_loss,
                        correct,
                        len(
                            self.data_transformation_artifact.transformed_test_object.dataset
                        ),
                        100.0
                        * correct
                        / len(
                            self.data_transformation_artifact.transformed_test_object.dataset
                        ),
                    )
                )
            logging.info(
                "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                    test_loss,
                    correct,
                    len(
                        self.data_transformation_artifact.transformed_test_object.dataset
                    ),
                    100.0
                    * correct
                    / len(
                        self.data_transformation_artifact.transformed_test_object.dataset
                    ),
                )
            )

            logging.info("Exited the test method of Model trainer class")

        except Exception as e:
            raise XRayException(e , sys)
        
    
    def iniatiate_model_trainer(self):
        try:
            logging.info(
                "Entered the initiate_model_trainer method of Model trainer class"
            )
            
            # Check if model file already exists - skip training if it does
            if os.path.exists(self.model_trainer_config.trained_model_path):
                logging.info(f"Model file already exists at {self.model_trainer_config.trained_model_path}. Skipping training.")
                model = torch.load(
                    self.model_trainer_config.trained_model_path, 
                    weights_only=False
                )
            else:
                # Train the model since it doesn't exist
                logging.info("Model file not found. Starting training...")
                
                model: Module = self.model.to(self.model_trainer_config.device)
                
                optimizer: Optimizer = torch.optim.SGD(
                    model.parameters(), **self.model_trainer_config.optimizer_params
                )
                
                scheduler: _LRScheduler = StepLR(
                    optimizer=optimizer, **self.model_trainer_config.scheduler_params
                )
                
                # Fixed epoch range to include all epochs
                for epoch in range(1, self.model_trainer_config.epochs + 1):
                    print("Epoch:", epoch)
                    
                    # Train method should handle batch loop with optimizer steps
                    self.train(optimizer=optimizer)
                    
                    # Scheduler steps once per epoch AFTER all batches
                    scheduler.step()
                    
                    # Evaluate on test set
                    self.test()
                
                # Save the trained model
                os.makedirs(self.model_trainer_config.artifact_dir, exist_ok=True)
                torch.save(model, self.model_trainer_config.trained_model_path)
                logging.info(f"Model saved to {self.model_trainer_config.trained_model_path}")
            
            # Load transformation object
            train_transform_obj = joblib.load(
                self.data_transformation_artifact.train_transform_file_path
            )
            
            # Save to BentoML
            bentoml.pytorch.save_model(
                name="x_ray_model",
                model=model,
                custom_objects={"transform": train_transform_obj}
            )
            logging.info("Model saved to BentoML successfully")
            
            model_trainer_artifact: ModelTrainerArtifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_path
            )
            
            logging.info(
                "Exited the initiate_model_trainer method of Model trainer class"
            )
            
            return model_trainer_artifact
                
        except Exception as e:
            raise XRayException(e, sys)




