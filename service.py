
import io
import bentoml
from bentoml.models import BentoModel
import numpy as np
import torch
from PIL import Image as PILImage
from pathlib import Path
from typing import Annotated
from bentoml.validators import ContentType
from src.constant import *
from src.dl.CustomNN import Net
from src.logger import logging


# Add your custom model class to safe globals for PyTorch 2.6+
torch.serialization.add_safe_globals([Net])


@bentoml.service(
    name=BENTOML_SERVICE_NAME,
    resources={"cpu": "2"},
    traffic={"timeout": 60},
    logging={
        "access": {
            "enabled": True,
            "request_content_length": True,
            "request_content_type": True,
            "response_content_length": True,
            "response_content_type": True,
        }
    }
)
class LungDiseaseService:
    bento_model = BentoModel(BENTOML_MODEL_NAME)
    
    def __init__(self):
        # Use .path property to get the directory path as a string
        model_dir = self.bento_model.path
        model_file = model_dir + r"\saved_model.pt"  # or "model.pth"
        
        # Load PyTorch model using string path
        self.model = torch.load(str(model_file), weights_only=False)
        self.transforms = self.bento_model.custom_objects.get("transform")
        if self.transforms is None:
                logging.warning(f"Transforms not found with key '{TRAIN_TRANSFORMS_KEY}'. Available custom objects: {list(self.bento_model.custom_objects.keys()) if self.bento_model.custom_objects else 'None'}")
                logging.info("Creating default transforms for inference")
    
    @bentoml.api
    def predict(self, image: Annotated[Path, ContentType('image/jpeg')]) -> str:
        """
        Predict lung disease from X-ray image
        """
        
        try:
            logging.info(f"Received prediction request for image: {image}")
            
            # Open image from the file path provided by BentoML
            logging.debug("Opening and converting image to RGB")
            img = PILImage.open(image).convert("RGB")
            logging.info(f"Image loaded successfully. Size: {img.size}")
            
            # Transform image
            logging.debug("Applying transformations to image")
            transformed = self.transforms(img).unsqueeze(0)
            image_tensor = torch.from_numpy(np.array(transformed))
            image_tensor = image_tensor.reshape(1, 3, 224, 224)
            logging.info(f"Image tensor shape: {image_tensor.shape}")
            
            # Run inference
            logging.debug("Running model inference")
            with torch.no_grad():
                batch_ret = self.model(image_tensor)
            logging.info(f"Model output shape: {batch_ret.shape}")
            
            # Get prediction
            pred_idx = max(torch.argmax(batch_ret, dim=1).detach().cpu().tolist())
            pred = PREDICTION_LABEL[pred_idx]
            
            logging.info(f"Prediction complete. Result: {pred} (index: {pred_idx})")
            logging.info(f"Model confidence scores: {batch_ret.detach().cpu().numpy()}")
            
            return pred
            
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}", exc_info=True)
            raise
