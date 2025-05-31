from sqlalchemy.orm import Session
from typing import List, Optional,Dict, Any
from fastapi import HTTPException, UploadFile
from schemas.medicines import MedicineCreate, MedicineUpdate, PaginatedMedicineResponse
from models.models import Medicines, Disease, DiseaseMedicine
import os
import base64
import logging

logger = logging.getLogger(__name__)
# Hàm xử lý logic lấy danh sách thuốc với phân trang và ảnh Base64
def get_paginated_medicines(db: Session, page: int = 1, per_page: int = 9) -> Dict[str, Any]:
    try:
        # Tính vị trí bắt đầu (skip)
        skip = (page - 1) * per_page
        
        # Lấy danh sách thuốc từ cơ sở dữ liệu
        medicines = db.query(Medicines).offset(skip).limit(per_page).all()
        
        # Tính tổng số bản ghi
        total_records = db.query(Medicines).count()
        
        # Xử lý dữ liệu thuốc, thêm ảnh Base64
        medicines_data = []
        for med in medicines:
            med_dict = med.__dict__.copy()
            
            # Lấy tên bệnh (folder_name) từ mối quan hệ Disease qua DiseaseMedicine
            disease_name = None
            if med.disease_medicines:
                disease = db.query(Disease).filter(Disease.disease_id == med.disease_medicines[0].disease_id).first()
                if disease:
                    disease_name = disease.name.replace(" ", "_").upper()  # Chuẩn hóa tên bệnh thành dạng ALGAL_LEAF_SPOT
            BASE_MEDIA_PATH = "media/medicines/"
            if disease_name and os.path.isdir(os.path.join(BASE_MEDIA_PATH, disease_name)):
                image_path_jpg = os.path.join(BASE_MEDIA_PATH, disease_name, f"{med.image_url}.jpg")
                image_path_png = os.path.join(BASE_MEDIA_PATH, disease_name, f"{med.image_url}.png")

                # Kiểm tra file ảnh .jpg hoặc .png
                if os.path.isfile(image_path_jpg):
                    with open(image_path_jpg, "rb") as image_file:
                        med_dict["image_base64"] = base64.b64encode(image_file.read()).decode("utf-8")
                elif os.path.isfile(image_path_png):
                    with open(image_path_png, "rb") as image_file:
                        med_dict["image_base64"] = base64.b64encode(image_file.read()).decode("utf-8")
                else:
                    med_dict["image_base64"] = None
            else:
                med_dict["image_base64"] = None
            
            medicines_data.append(med_dict)
        
        # Trả về kết quả phân trang
        return {
            "data": medicines_data,
            "total_records": total_records,
            "page": page,
            "per_page": per_page
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy danh sách thuốc: {str(e)}")

# Hàm tạo thuốc mới
def create_medicine(db: Session, medicine_data: MedicineCreate, file: UploadFile) -> Medicines:
    try:
        BASE_MEDIA_PATH = "media/medicines/"
        disease_map = {
            1: "PHYTOPHTHORA_PALMIVORA",
            2: "ALLOCARIDARA_ATTACK",
            3: "LEAF_BLIGHT",
            4: "LEAF_RHIZOCTONIA",
            5: "PHOMOPSIS_LEAF_SPOT",
            6: "HEALTHY_LEAF",
            7: "PHYTOPHTHORA_LEAF_BLIGHT",
            8: "LEAF_SPOT",
            9: "LEAF_ALGAL",
            10: "LEAF_COLLETOTRICHUM",
            11: "ALGAL_LEAF_SPOT"
        }
        folder_name = disease_map.get(medicine_data.disease_id, "Unknown")
        folder_path = os.path.join(BASE_MEDIA_PATH, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        db_medicine = Medicines(
            name=medicine_data.name,
            description=medicine_data.description,
            price=medicine_data.price,
            discounted_price=medicine_data.discounted_price,
            stock=medicine_data.stock,
            disease_id=medicine_data.disease_id,
            image_url="",
            is_freeship=medicine_data.is_freeship,
            how_to_use=medicine_data.how_to_use  # Thêm trường how_to_use
        )
        db.add(db_medicine)
        db.commit()
        db.refresh(db_medicine)

        file_extension = os.path.splitext(file.filename)[1]
        file_name = f"{db_medicine.id}{file_extension}"
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        db_medicine.image_url = file_path
        db.commit()
        db.refresh(db_medicine)

        return db_medicine
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating medicine: {str(e)}")

# Hàm lấy danh sách thuốc từ cơ sở dữ liệu
def get_medicines(db: Session, skip: int = 0, limit: int = 100) -> List[Medicines]:
    return db.query(Medicines).offset(skip).limit(limit).all()

# Hàm lấy thuốc theo ID
def get_medicine_by_id(db: Session, medicine_id: int) -> Optional[Dict[str, Any]]:
    try:
        # Lấy thông tin thuốc từ cơ sở dữ liệu
        medicine = db.query(Medicines).filter(Medicines.medicine_id == medicine_id).first()
        if not medicine:
            logger.info(f"Medicine with ID {medicine_id} not found")
            return None

        # Chuyển thông tin thuốc thành dictionary
        med_dict = {
            "medicine_id": medicine.medicine_id,
            "name": medicine.name,
            "description": medicine.description,
            "price": float(medicine.price) if medicine.price is not None else None,
            "stock": medicine.stock,
            "image_url": medicine.image_url,
            "how_to_use": medicine.how_to_use,
            "created_at": medicine.created_at
        }

        # Lấy tên bệnh từ mối quan hệ Disease qua DiseaseMedicine
        disease_name = None
        disease_medicine = db.query(DiseaseMedicine).filter(DiseaseMedicine.medicine_id == medicine_id).first()
        if disease_medicine:
            disease = db.query(Disease).filter(Disease.disease_id == disease_medicine.disease_id).first()
            if disease:
                disease_name = disease.name.replace(" ", "_").upper()  # Chuẩn hóa: ALGAL_LEAF_SPOT
                med_dict["disease_name"] = disease_name

        # Xử lý hình ảnh Base64
        BASE_MEDIA_PATH = "media/medicines/"
        med_dict["image_base64"] = None
        if disease_name and os.path.isdir(os.path.join(BASE_MEDIA_PATH, disease_name)):
            for ext in ["jpg", "png"]:
                image_path = os.path.join(BASE_MEDIA_PATH, disease_name, f"{medicine.image_url}.{ext}")
                if os.path.isfile(image_path):
                    try:
                        with open(image_path, "rb") as image_file:
                            med_dict["image_base64"] = base64.b64encode(image_file.read()).decode("utf-8")
                        break
                    except Exception as e:
                        logger.error(f"Error encoding image for medicine {medicine_id}: {str(e)}")
        else:
            logger.warning(f"No image directory found for disease {disease_name} or disease_name is None")

        return med_dict
    except Exception as e:
        logger.error(f"Error retrieving medicine {medicine_id}: {str(e)}", exc_info=True)
        raise Exception(f"Error retrieving medicine: {str(e)}")

# Hàm cập nhật thuốc
def update_medicine(db: Session, medicine_id: int, medicine_data: MedicineUpdate) -> Optional[Medicines]:
    try:
        # Kiểm tra sự tồn tại của thuốc bằng get_medicine_by_id
        existing_medicine = get_medicine_by_id(db, medicine_id)
        if not existing_medicine:
            logger.info(f"Medicine with ID {medicine_id} not found")
            return None

        # Lấy đối tượng Medicines để cập nhật
        db_medicine = db.query(Medicines).filter(Medicines.medicine_id == medicine_id).first()
        if not db_medicine:
            logger.warning(f"Medicine with ID {medicine_id} not found in database")
            return None

        # Chuyển medicine_data thành dictionary
        medicine_dict = medicine_data.dict(exclude_unset=True)

        # Cập nhật các trường nếu chúng tồn tại trong dictionary
        if "name" in medicine_dict:
            db_medicine.name = medicine_dict["name"]
        if "description" in medicine_dict:
            db_medicine.description = medicine_dict["description"]
        if "price" in medicine_dict:
            db_medicine.price = medicine_dict["price"]
        if "stock" in medicine_dict:
            db_medicine.stock = medicine_dict["stock"]
        if "image_url" in medicine_dict:
            db_medicine.image_url = medicine_dict["image_url"]
        if "how_to_use" in medicine_dict:
            db_medicine.how_to_use = medicine_dict["how_to_use"]

        db.commit()
        db.refresh(db_medicine)
        logger.info(f"Medicine with ID {medicine_id} updated successfully")
        return db_medicine
    except Exception as e:
        logger.error(f"Error updating medicine {medicine_id}: {str(e)}", exc_info=True)
        raise Exception(f"Error updating medicine: {str(e)}")

# Hàm xóa thuốc
def delete_medicine(db: Session, medicine_id: int) -> Optional[Medicines]:
    db_medicine = get_medicine_by_id(db, medicine_id)
    if db_medicine:
        db.delete(db_medicine)
        db.commit()
        return db_medicine
    return None