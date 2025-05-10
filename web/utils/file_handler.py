import os
import shutil
from uuid import uuid4
from fastapi import UploadFile, HTTPException
import logging
from config.db import DISEASE_IMAGE_DIRS, MEDIA_ROOT

logger = logging.getLogger(__name__)

def ensure_dir(directory: str):
    """
    Tạo thư mục nếu chưa tồn tại.
    """
    os.makedirs(directory, exist_ok=True)

def save_image(upload_file: UploadFile, disease_type: str) -> str:
    """
    Lưu file ảnh vào thư mục tương ứng với loại bệnh.
    Trả về đường dẫn tương đối của file đã lưu.
    """
    if disease_type not in DISEASE_IMAGE_DIRS:
        raise HTTPException(status_code=400, detail=f"Invalid disease type: {disease_type}")

    disease_dir = DISEASE_IMAGE_DIRS[disease_type]

    ext = os.path.splitext(upload_file.filename)[1]
    unique_filename = f"{uuid4()}{ext}"
    relative_path = os.path.join("medicines", disease_type, unique_filename)
    absolute_path = os.path.join(disease_dir, unique_filename)

    try:
        # Lưu file
        with open(absolute_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return relative_path
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        raise HTTPException(status_code=500, detail="Could not save image file")
    finally:
        upload_file.file.close()

def delete_image(relative_path: str):
    """
    Xóa file ảnh dựa vào đường dẫn tương đối.
    """
    if not relative_path:
        return

    absolute_path = os.path.join(MEDIA_ROOT, relative_path)
    if os.path.isfile(absolute_path):
        try:
            os.remove(absolute_path)
            logger.info(f"Deleted image file: {absolute_path}")
        except Exception as e:
            logger.error(f"Error deleting image file {absolute_path}: {e}")
            raise HTTPException(status_code=500, detail="Could not delete image file")
    else:
        logger.warning(f"Image file not found for deletion: {absolute_path}")
