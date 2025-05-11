import os
import torch
import torch.nn as nn
import logging
from torchvision import models, transforms
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import Dict
import numpy as np
import cv2
from PIL import Image
from dotenv import load_dotenv

from config.db import get_db
from models.models import Disease as DiseaseModel, Medicines as MedicineModel, DiseaseMedicine

router = APIRouter(prefix="/diseases", tags=["Diseases"])

# Load environment variables
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    'ALGAL_LEAF_SPOT',
    'ALLOCARIDARA_ATTACK',
    'HEALTHY_LEAF',
    'LEAF_ALGAL',
    'LEAF_BLIGHT',
    'LEAF_COLLETOTRICHUM',
    'LEAF_RHIZOCTONIA',
    'LEAF_SPOT',
    'PHOMOPSIS_LEAF_SPOT',
    'PHYTOPHTHORA_LEAF_BLIGHT',
    'PHYTOPHTHORA_PALMIVORA'
]
CONFIDENCE_THRESHOLD = 0.05

# Global variable for the model
model = None

# Custom EfficientNet model class
class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes=11):
        super(CustomEfficientNet, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)

        # Đóng băng các lớp đầu tiên
        for name, param in self.model.named_parameters():
            if 'features.0' in name or 'features.1' in name or 'features.2' in name:
                param.requires_grad = False
        
        # Thay thế phần classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=1280, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Preprocess the image
def preprocess_image(img: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = Image.fromarray(img)  

    transform = transforms.Compose([
        transforms.Resize((256, 256)),       # Resize theo file train
        transforms.ToTensor(),               # Chuyển đổi sang tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Chuẩn hóa
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
        model_instance = CustomEfficientNet(num_classes=len(CLASS_NAMES))

        # Load mô hình từ file
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

        # Nếu checkpoint là mô hình hoàn chỉnh
        if isinstance(checkpoint, CustomEfficientNet):
            model_instance = checkpoint
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            model_instance.load_state_dict(checkpoint["model"], strict=False)
        else:
            model_instance.load_state_dict(checkpoint, strict=False)

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
