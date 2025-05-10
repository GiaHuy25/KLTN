import os
import torch
import logging
from torchvision.models import efficientnet_b0
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import Dict
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from dotenv import load_dotenv

from config.db import get_db
from models.models import Disease as DiseaseModel, Medicines as MedicineModel, DiseaseMedicine

router = APIRouter(prefix="/diseases", tags=["Diseases"])

# Load environment variables
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    'PHYTOPHTHORA_PALMIVORA', 'ALLOCARIDARA_ATTACK', 'LEAF_BLIGHT',
    'LEAF_RHIZOCTONIA', 'PHOMOPSIS_LEAF_SPOT', 'HEALTHY_LEAF',
    'PHYTOPHTHORA_LEAF_BLIGHT', 'LEAF_SPOT', 'LEAF_ALGAL',
    'LEAF_COLLETOTRICHUM', 'ALGAL_LEAF_SPOT'
]
CONFIDENCE_THRESHOLD = 0.05

# Global variable for the model
model = None

# Custom EfficientNet model class
class CustomEfficientNet(torch.nn.Module):
    def __init__(self, num_classes=11):
        super(CustomEfficientNet, self).__init__()
        self.model = efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Preprocess the image
def preprocess_image(img: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = Image.fromarray(img)  

    transform = transforms.Compose([
        transforms.Resize((224, 224)),       
        transforms.ToTensor(),              
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
    ])
    tensor = transform(img).unsqueeze(0)        
    return tensor


# Load model only once
def load_model():
    global model
    try:
        if not os.path.isfile(MODEL_PATH):
            raise RuntimeError(f"Model file not found at {MODEL_PATH}")

        # Tạo một mô hình mới từ CustomEfficientNet
        model_instance = CustomEfficientNet()

        # Load state_dict từ tệp đã lưu
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

        # Nếu checkpoint là OrderedDict, tải vào model
        if isinstance(checkpoint, dict):
            model_instance.load_state_dict(checkpoint, strict=False)
        else:
            raise RuntimeError("Loaded model is not a valid state_dict.")

        # Đặt mô hình ở chế độ eval
        model_instance.eval()
        model = model_instance
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Error loading model: {e}")

# Load model at startup
load_model()

@router.post("/predict", response_model=Dict, summary="Predict disease from an uploaded image")
async def predict_disease(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        global model
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Read the uploaded image
        contents = await file.read()
        nparray = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparray, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Unable to decode image")

        # Preprocess the image
        input_tensor = preprocess_image(img)

        # Make prediction
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output[0], dim=0).numpy()
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])

        # Check confidence
        if confidence < CONFIDENCE_THRESHOLD:
            raise HTTPException(status_code=400, detail="Disease cannot be identified from the image")

        disease_name = CLASS_NAMES[predicted_class]
        disease = db.query(DiseaseModel).filter(DiseaseModel.name == disease_name).first()
        if not disease:
            raise HTTPException(status_code=404, detail="Disease information not found")

        # Fetch medicines for the predicted disease
        medicines = (
            db.query(MedicineModel)
            .join(DiseaseMedicine, MedicineModel.medicine_id == DiseaseMedicine.medicine_id)
            .filter(DiseaseMedicine.disease_id == disease.disease_id)
            .all()
        )

        medicines_data = [
            {
                "id": med.medicine_id,
                "name": med.name,
                "description": med.description,
                "how_to_use": med.how_to_use,
                "price": float(med.price),
                "stock": med.stock,
                "image_url": med.image_url
            }
            for med in medicines
        ]

        return {
            "disease_name": disease.name,
            "description": disease.description,
            "symptoms": disease.symptoms,
            "cause": disease.cause,
            "confidence": confidence,
            "medicines": medicines_data
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in /predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Server error")
