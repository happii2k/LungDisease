import io
import bentoml
from bentoml.models import BentoModel
import numpy as np
import torch as torchlib
from PIL import Image as PILImage
from src.constant import *


@bentoml.service(
    name=BENTOML_SERVICE_NAME,
    resources={"cpu": "2"},
    traffic={"timeout": 60}
)
class LungDiseaseService:
    # Load the model as a class attribute
    bento_model = BentoModel(BENTOML_MODEL_NAME)
    
    def __init__(self):
        # Load the PyTorch model during initialization
        self.model = torchlib.load(self.bento_model)
        self.transforms = self.bento_model.custom_objects.get(TRAIN_TRANSFORMS_KEY)
    
    @bentoml.api
    def predict(self, img: PILImage.Image) -> str:
        """
        Predict lung disease from X-ray image
        """
        # Load and transform image
        image = img.convert("RGB")
        image = torch.from_numpy(np.array(self.transforms(image).unsqueeze(0)))
        image = image.reshape(1, 3, 224, 224)
        
        # Run inference
        with torch.no_grad():
            batch_ret = self.model(image)
        
        # Get prediction
        pred = PREDICTION_LABEL[max(torch.argmax(batch_ret, dim=1).detach().cpu().tolist())]
        
        return pred
