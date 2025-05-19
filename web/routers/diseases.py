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
import base64
from config.db import get_db
from models.models import Disease as DiseaseModel, Medicines as MedicineModel, DiseaseMedicine

# Cấu hình router và logging
router = APIRouter(prefix="/diseases", tags=["Diseases"])
logger = logging.getLogger(__name__)

# Load biến môi trường
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")
if not MODEL_PATH:
    logger.error("MODEL_PATH environment variable not set")
    raise RuntimeError("MODEL_PATH environment variable not set")

# Định nghĩa danh sách lớp bệnh và ngưỡng tin cậy
CLASS_NAMES = [
    'ALGAL_LEAF_SPOT',
    'ALLOCARIDARA_ATTACK',
    'HEALTHY_LEAF',
    'LEAF_ALGAL',
    'LEAF_BLIGHT',
    'LEAF_COLLETOTRICHUM',
    'LEAF_RHIZOCTONIA',
    'LEAF_SPOT',
    'NO_OBJECT',
    'PHOMOPSIS_LEAF_SPOT',
    'PHYTOPHTHORA_LEAF_BLIGHT',
    'PHYTOPHTHORA_PALMIVORA'
]
CONFIDENCE_THRESHOLD = 0.7

# Biến toàn cục cho mô hình
model = None

# Định nghĩa class mô hình tùy chỉnh
class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes: int = len(CLASS_NAMES)):
        super(CustomEfficientNet, self).__init__()
        # Sử dụng EfficientNet-B0 làm backbone
        self.model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')

        # Đóng băng các lớp đầu tiên (features.0, features.1, features.2)
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in ['features.0', 'features.1', 'features.2']):
                param.requires_grad = False

        # Thay thế classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=1280, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=512, out_features=num_classes, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# Hàm tiền xử lý ảnh
def preprocess_image(image: np.ndarray) -> torch.Tensor:
    try:
        # Kiểm tra xem ảnh có hợp lệ không
        if image is None or len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Invalid image: Expected a 3-channel color image")

        # Chuyển đổi từ BGR (OpenCV) sang RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Chuyển đổi từ numpy array sang PIL Image
        pil_image = Image.fromarray(image_rgb)

        # Định nghĩa pipeline transform (đồng bộ với huấn luyện)
        transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Áp dụng transform và thêm chiều batch
        tensor = transform(pil_image).unsqueeze(0)
        return tensor
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail=f"Error preprocessing image: {str(e)}")

# Hàm load mô hình
def load_model():
    global model
    try:
        # Kiểm tra file mô hình
        if not os.path.isfile(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            raise RuntimeError(f"Model file not found at {MODEL_PATH}")

        # Khởi tạo mô hình
        model_instance = CustomEfficientNet(num_classes=len(CLASS_NAMES))

        # Load trọng số mô hình
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

        # Xử lý checkpoint
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model_instance.load_state_dict(checkpoint["model"], strict=False)
        else:
            model_instance.load_state_dict(checkpoint, strict=False)

        # Chuyển mô hình sang chế độ đánh giá
        model_instance.eval()
        model = model_instance
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Error loading model: {e}")

# Load mô hình khi khởi động
try:
    load_model()
except Exception as e:
    logger.error(f"Failed to load model at startup: {e}")
    raise RuntimeError(f"Failed to load model at startup: {e}")

# API dự đoán bệnh
@router.post("/predict", response_model=Dict, summary="Dự đoán bệnh từ ảnh được tải lên")
async def predict_disease(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        global model
        if model is None:
            logger.error("Model not loaded")
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Đọc ảnh từ file tải lên
        contents = await file.read()
        nparray = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparray, cv2.IMREAD_COLOR)

        if img is None:
            logger.error("Unable to decode image")
            raise HTTPException(status_code=400, detail="Unable to decode image")

        # Tiền xử lý ảnh
        input_tensor = preprocess_image(img)

        # Dự đoán
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output[0], dim=0).cpu().numpy()
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])

        # Kiểm tra ngưỡng tin cậy
        if confidence < CONFIDENCE_THRESHOLD:
            logger.warning(f"Confidence {confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}")
            raise HTTPException(status_code=400, detail="Confidence too low to identify disease")

        # Lấy tên bệnh
        disease_name = CLASS_NAMES[predicted_class]
        logger.info(f"Predicted disease: {disease_name} with confidence {confidence:.2f}")

        # Truy vấn thông tin bệnh từ cơ sở dữ liệu
        disease = db.query(DiseaseModel).filter(DiseaseModel.name == disease_name).first()
        if not disease:
            logger.error(f"Disease {disease_name} not found in database")
            raise HTTPException(status_code=404, detail="Disease information not found")

        # Lấy danh sách thuốc liên quan
        medicines = (
            db.query(MedicineModel)
            .join(DiseaseMedicine, MedicineModel.medicine_id == DiseaseMedicine.medicine_id)
            .filter(DiseaseMedicine.disease_id == disease.disease_id)
            .all()
        )

        # Xử lý dữ liệu thuốc và mã hóa ảnh Base64
        medicines_data = []
        BASE_MEDIA_PATH = "media/medicines/"
        disease_folder = disease.name.replace(" ", "_").upper()

        for med in medicines:
            med_dict = {
                "id": med.medicine_id,
                "name": med.name,
                "description": med.description,
                "how_to_use": med.how_to_use,
                "price": float(med.price),
                "stock": med.stock,
                "image_url": med.image_url,
                "image_base64": None
            }

            # Mã hóa ảnh Base64 nếu tồn tại
            if os.path.isdir(os.path.join(BASE_MEDIA_PATH, disease_folder)):
                for ext in ['jpg', 'png']:
                    image_path = os.path.join(BASE_MEDIA_PATH, disease_folder, f"{med.image_url}.{ext}")
                    if os.path.isfile(image_path):
                        with open(image_path, "rb") as image_file:
                            med_dict["image_base64"] = base64.b64encode(image_file.read()).decode("utf-8")
                        break

            medicines_data.append(med_dict)

        # Trả về kết quả
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
        logger.error(f"Unexpected error in /predict: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")