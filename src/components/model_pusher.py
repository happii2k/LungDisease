import os
import sys

from src.entity.artifacts_config import ModelPusherArtifact
from src.entity.config import ModelPusherConfig
from src.exception import XRayException
from src.logger import logging

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
        self.model_pusher_config = model_pusher_config

    def build_and_push_bento_image(self):
        logging.info("Entered build_and_push_bento_image method of ModelPusher class")

        try:
            logging.info("Building the bento from bentofile.yaml")
            os.system("bentoml build")
            logging.info("Built the bento from bentofile.yaml")

            # Set up Azure ACR details
            acr_name = self.model_pusher_config.azure_acr_name  # e.g. "myacr"
            acr_login_server = f"{acr_name}.azurecr.io"
            image_name = self.model_pusher_config.bentoml_ecr_image  # e.g. "mybentoimage"

            # Build container image for bento
            logging.info("Creating docker image for bento")
            os.system(
                f"bentoml containerize {self.model_pusher_config.bentoml_service_name}:latest "
                f"-t {acr_login_server}/{image_name}:latest"
            )
            logging.info("Created docker image for bento")

            # Login to Azure ACR
            logging.info("Logging into Azure Container Registry")
            os.system(f"az acr login --name {acr_name}")
            logging.info("Logged into Azure Container Registry")

            # Push image to ACR
            logging.info("Pushing bento image to Azure Container Registry")
            os.system(f"docker push {acr_login_server}/{image_name}:latest")
            logging.info("Pushed bento image to Azure Container Registry")

            logging.info(
                "Exited build_and_push_bento_image method of ModelPusher class"
            )

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   This method initiates model pusher.
        Output      :   Model pusher artifact
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            self.build_and_push_bento_image()
            model_pusher_artifact = ModelPusherArtifact(
                bentoml_model_name=self.model_pusher_config.bentoml_model_name,
                bentoml_service_name=self.model_pusher_config.bentoml_service_name,
            )

            logging.info("Exited the initiate_model_pusher method of ModelPusher class")
            return model_pusher_artifact
        except Exception as e:
            raise XRayException(e, sys)
