import torch
import cv2
import numpy as np
from torchvision import transforms, models
from torchvision.models import efficientnet_b0
from torchvision.models import resnet50
import torch.nn as nn
from PIL import Image

# Tạo lớp mô hình tùy chỉnh giống với mô hình huấn luyện
class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()
        # Sử dụng EfficientNet-B0 làm mô hình cơ sở
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

# Tên các lớp bệnh
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

# Đường dẫn mô hình
MODEL_PATH = "E:\\Kysu\\Doan\\KLTN\\KLTN\\web\\efficientnet.pth"

# Hàm tiền xử lý ảnh
from PIL import Image

def preprocess_image(img: np.ndarray) -> torch.Tensor:
    # Chuyển đổi từ BGR (OpenCV) sang RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    # Chuyển đổi từ numpy array sang PIL Image
    img = Image.fromarray(img)  

    transform = transforms.Compose([
        transforms.Resize((256, 256)),       # Resize về đúng kích thước 224x224
        transforms.ToTensor(),               # Chuyển đổi sang tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0)        # Thêm chiều batch
    return tensor


# Hàm load model
def load_model():
    try:
        # Tạo mô hình tùy chỉnh
        model = CustomEfficientNet(num_classes=len(CLASS_NAMES))

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
image_path = "C:\\Users\\Admini\\OneDrive\\Desktop\\7.jpg"
predicted_disease = predict_disease(image_path)
print(f"Disease detected: {predicted_disease}")
