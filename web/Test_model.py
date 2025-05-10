import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.models import efficientnet_b0
import torch.nn as nn
from PIL import Image

# Tạo lớp mô hình tùy chỉnh giống với mô hình huấn luyện
class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes=11):
        super(CustomEfficientNet, self).__init__()
        self.model = efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Tên các lớp bệnh
CLASS_NAMES = [
    'PHYTOPHTHORA_PALMIVORA', 'ALLOCARIDARA_ATTACK', 'LEAF_BLIGHT',
    'LEAF_RHIZOCTONIA', 'PHOMOPSIS_LEAF_SPOT', 'HEALTHY_LEAF',
    'PHYTOPHTHORA_LEAF_BLIGHT', 'LEAF_SPOT', 'LEAF_ALGAL',
    'LEAF_COLLETOTRICHUM', 'ALGAL_LEAF_SPOT'
]

# Đường dẫn mô hình
MODEL_PATH = "E:\\Kysu\\Doan\\New folder\\web\\EfficientNet-B0.pth"

# Hàm tiền xử lý ảnh
from PIL import Image

def preprocess_image(img: np.ndarray) -> torch.Tensor:
    # Chuyển đổi từ BGR (OpenCV) sang RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    # Chuyển đổi từ numpy array sang PIL Image
    img = Image.fromarray(img)  

    transform = transforms.Compose([
        transforms.Resize((224, 224)),       # Resize về đúng kích thước 224x224
        transforms.ToTensor(),               # Chuyển đổi sang tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Chuẩn hóa như ImageNet
    ])
    tensor = transform(img).unsqueeze(0)        # Thêm chiều batch
    return tensor


# Hàm load model
def load_model():
    try:
        # Tạo mô hình tùy chỉnh
        model = CustomEfficientNet()
        # Load trọng số
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        # Xử lý trường hợp mô hình là dictionary với key "model"
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Hàm dự đoán bệnh từ ảnh
def predict_disease(image_path: str):
    try:
        # Load model
        model = load_model()
        if model is None:
            print("Model could not be loaded.")
            return None

        # Đọc ảnh từ đường dẫn sử dụng OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print("Unable to read image from the provided path.")
            return None

        # Tiền xử lý ảnh
        input_tensor = preprocess_image(img)

        # Dự đoán
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output[0], dim=0).numpy()
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])

        disease_name = CLASS_NAMES[predicted_class]
        print(f"Predicted disease: {disease_name} (Confidence: {confidence:.2f})")
        return disease_name
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None

# Ví dụ sử dụng
image_path = "C:\\Users\\Admini\\OneDrive\\Desktop\\4.jpg"
predicted_disease = predict_disease(image_path)
print(f"Disease detected: {predicted_disease}")
