import os
import sys

import bentoml
import joblib
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from tqdm import tqdm
from torch import nn

from src.constant import *
from src.exception import XRayException
from src.logger import logging
from src.dl.CustomNN import Net
from src.entity.config import ModelTrainerConfig
from src.entity.artifacts_config import DataTransformationArtifact, ModelTrainerArtifact


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact
        self.model: Module = Net()
        self.criterion = nn.CrossEntropyLoss()  # Create once for the class
        
        # Track metrics across epochs
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

    def train(self, optimizer: Optimizer, epoch: int) -> tuple:
        """
        Train the model for one epoch
        Returns: (average_loss, accuracy)
        """
        logging.info(f"Entered the train method of Model trainer class - Epoch {epoch}")
    
        try:
            self.model.train()
            pbar = tqdm(self.data_transformation_artifact.transformed_train_object)
            correct: int = 0
            processed: int = 0
            running_loss: float = 0.0
            
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.model_trainer_config.device), target.to(self.model_trainer_config.device)
                optimizer.zero_grad()
                y_pred = self.model(data)
                
                loss = self.criterion(y_pred, target)
                running_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
                pred = y_pred.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                processed += len(data)
                
                pbar.set_description(
                    desc=f"Epoch {epoch} | Loss={running_loss/(batch_idx+1):.4f} | Accuracy={100*correct/processed:.2f}%"
                )
            
            # Calculate epoch metrics
            avg_loss = running_loss / len(self.data_transformation_artifact.transformed_train_object)
            accuracy = 100.0 * correct / processed
            
            # Store metrics
            self.train_losses.append(avg_loss)
            self.train_accuracies.append(accuracy)
            
            logging.info(f"Train - Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
            logging.info("Exited the train method of Model trainer class")
            
            return avg_loss, accuracy
        
        except Exception as e:
            raise XRayException(e, sys)

    def test(self, epoch: int) -> tuple:
        """
        Test the model on test dataset
        Returns: (average_loss, accuracy)
        """
        try:
            logging.info(f"Entered the test method of Model trainer class - Epoch {epoch}")
            self.model.eval()
            
            test_loss: float = 0.0
            correct: int = 0
            
            with torch.no_grad():
                for data, target in self.data_transformation_artifact.transformed_test_object:
                    # FIX: Use self.model_trainer_config.device consistently
                    data, target = data.to(self.model_trainer_config.device), target.to(self.model_trainer_config.device)
                    output = self.model(data)
                    
                    # Accumulate loss (multiply by batch size for correct averaging)
                    test_loss += self.criterion(output, target).item() * data.size(0)
                    
                    # Get predictions
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Calculate average loss and accuracy AFTER the loop
            total_samples = len(self.data_transformation_artifact.transformed_test_object.dataset)
            test_loss /= total_samples
            test_accuracy = 100.0 * correct / total_samples
            
            # Store metrics
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_accuracy)
            
            # Print results
            print("\nTest set - Epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                epoch,
                test_loss,
                correct,
                total_samples,
                test_accuracy
            ))
            
            # Log results
            logging.info(
                "Test set - Epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                    epoch,
                    test_loss,
                    correct,
                    total_samples,
                    test_accuracy
                )
            )
            
            logging.info("Exited the test method of Model trainer class")
            
            return test_loss, test_accuracy
            
        except Exception as e:
            raise XRayException(e, sys)

    def initiate_model_trainer(self):
        """
        Main method to initiate model training pipeline
        """
        try:
            logging.info(
                "Entered the initiate_model_trainer method of Model trainer class"
            )
            
            # Check if model file already exists - skip training if it does
            if os.path.exists(self.model_trainer_config.trained_model_path):
                logging.info(f"Model file already exists at {self.model_trainer_config.trained_model_path}. Loading existing model.")
                model = torch.load(
                    self.model_trainer_config.trained_model_path, 
                    weights_only=False
                )
            else:
                # Train the model since it doesn't exist
                logging.info("Model file not found. Starting training...")
                
                # Move model to device
                self.model = self.model.to(self.model_trainer_config.device)
                
                # Setup optimizer
                optimizer: Optimizer = torch.optim.SGD(
                    self.model.parameters(), **self.model_trainer_config.optimizer_params
                )
                
                # Setup scheduler
                scheduler: _LRScheduler = StepLR(
                    optimizer=optimizer, **self.model_trainer_config.scheduler_params
                )
                
                # Training loop
                for epoch in range(1, self.model_trainer_config.epochs + 1):
                    print(f"\n{'='*60}")
                    print(f"EPOCH {epoch}/{self.model_trainer_config.epochs}")
                    print(f"{'='*60}")
                    
                    # Train for one epoch
                    train_loss, train_acc = self.train(optimizer=optimizer, epoch=epoch)
                    
                    # Test after each epoch
                    test_loss, test_acc = self.test(epoch=epoch)
                    
                    # Step the scheduler
                    scheduler.step()
                    
                    # Log current learning rate
                    current_lr = optimizer.param_groups[0]['lr']
                    logging.info(f"Epoch {epoch} - Learning Rate: {current_lr}")
                    print(f"Current Learning Rate: {current_lr}\n")
                
                # Save metrics summary
                logging.info("\n" + "="*60)
                logging.info("TRAINING COMPLETED")
                logging.info("="*60)
                logging.info(f"Final Train Loss: {self.train_losses[-1]:.4f}, Train Accuracy: {self.train_accuracies[-1]:.2f}%")
                logging.info(f"Final Test Loss: {self.test_losses[-1]:.4f}, Test Accuracy: {self.test_accuracies[-1]:.2f}%")
                logging.info("="*60 + "\n")
                
                # Move model to CPU before saving
                model = self.model.to('cpu')
                
                # Save the trained model
                os.makedirs(self.model_trainer_config.artifact_dir, exist_ok=True)
                torch.save(model, self.model_trainer_config.trained_model_path)
                logging.info(f"Model saved to {self.model_trainer_config.trained_model_path}")
            
            # Load transformation object
            train_transform_obj = joblib.load(
                self.data_transformation_artifact.train_transform_file_path
            )
            
            # Save to BentoML (model should be on CPU)
            bentoml.pytorch.save_model(
                name="x_ray_model",
                model=model,
                custom_objects={"transform": train_transform_obj}
            )
            logging.info("Model saved to BentoML successfully")
            
            # Create artifact
            model_trainer_artifact: ModelTrainerArtifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_path
            )
            
            logging.info(
                "Exited the initiate_model_trainer method of Model trainer class"
            )
            
            return model_trainer_artifact
                
        except Exception as e:
            raise XRayException(e, sys)
