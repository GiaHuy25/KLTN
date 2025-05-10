from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Kết nối với cơ sở dữ liệu MySQL
engine = create_engine("mysql+pymysql://root:1234@localhost:3306/klks_schema")
meta = MetaData()
conn = engine.connect()

# Tạo session và base cho SQLAlchemy
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Đường dẫn gốc của dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cấu hình thư mục media
MEDIA_ROOT = os.path.join(BASE_DIR, "media")
MEDICINE_IMAGE_DIR_RELATIVE = "medicines"  # Đường dẫn tương đối cho ảnh thuốc
MEDICINE_IMAGE_DIR_ABSOLUTE = os.path.join(MEDIA_ROOT, MEDICINE_IMAGE_DIR_RELATIVE)

# Các loại bệnh lá sầu riêng
DISEASE_TYPES = [
    "PHYTOPHTHORA_PALMIVORA",
    "ALLOCARIDARA_ATTACK",
    "LEAF_BLIGHT",
    "LEAF_RHIZOCTONIA",
    "PHOMOPSIS_LEAF_SPOT",
    "HEALTHY_LEAF",
    "PHYTOPHTHORA_LEAF_BLIGHT",
    "LEAF_SPOT",
    "LEAF_ALGAL",
    "LEAF_COLLETOTRICHUM",
    "ALGAL_LEAF_SPOT"
]

# Tạo các thư mục lưu ảnh thuốc theo loại bệnh
DISEASE_IMAGE_DIRS = {
    disease: os.path.join(MEDICINE_IMAGE_DIR_ABSOLUTE, disease) for disease in DISEASE_TYPES
}

# Tạo các thư mục nếu chưa tồn tại
for dir_path in DISEASE_IMAGE_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

print(f"Đã cấu hình các thư mục lưu trữ ảnh thuốc theo loại bệnh tại: {MEDICINE_IMAGE_DIR_ABSOLUTE}")
