from sqlalchemy.orm import Session
from typing import List, Optional
from fastapi import HTTPException, UploadFile
from schemas.medicines import MedicineCreate, MedicineUpdate, PaginatedMedicineResponse
from models.models import Medicines
import os
import base64

# Hàm xử lý logic lấy danh sách thuốc với phân trang và ảnh Base64
def get_paginated_medicines(db: Session, page: int = 1, per_page: int = 10):
    try:
        # Tính vị trí bắt đầu (skip)
        skip = (page - 1) * per_page
        
        # Lấy danh sách thuốc từ cơ sở dữ liệu
        medicines = get_medicines(db, skip=skip, limit=per_page)
        
        # Tính tổng số bản ghi
        total_records = db.query(Medicines).count()
        
        # Xử lý dữ liệu thuốc, thêm ảnh Base64
        medicines_data = []
        for med in medicines:
            med_dict = med.__dict__.copy()
            image_path = med.image_url if med.image_url else None
            if image_path and os.path.isfile(image_path):
                with open(image_path, "rb") as image_file:
                    med_dict["image_base64"] = base64.b64encode(image_file.read()).decode("utf-8")
            else:
                med_dict["image_base64"] = None
            medicines_data.append(med_dict)
        
        # Trả về kết quả phân trang
        return {
            "data": medicines_data,
            "total_records": total_records,
            "page": page,
            "per_page": per_page  # Sửa từ "_per_page" thành "per_page"
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
def get_medicine_by_id(db: Session, medicine_id: int) -> Optional[Medicines]:
    return db.query(Medicines).filter(Medicines.medicine_id == medicine_id).first()

# Hàm cập nhật thuốc
def update_medicine(db: Session, medicine_id: int, medicine_data: MedicineUpdate) -> Optional[Medicines]:
    db_medicine = get_medicine_by_id(db, medicine_id)
    if db_medicine:
        if medicine_data.name is not None:
            db_medicine.name = medicine_data.name
        if medicine_data.description is not None:
            db_medicine.description = medicine_data.description
        if medicine_data.price is not None:
            db_medicine.price = medicine_data.price
        if medicine_data.discounted_price is not None:
            db_medicine.discounted_price = medicine_data.discounted_price
        if medicine_data.stock is not None:
            db_medicine.stock = medicine_data.stock
        if medicine_data.disease_id is not None:
            db_medicine.disease_id = medicine_data.disease_id
        if medicine_data.image_url is not None:
            db_medicine.image_url = medicine_data.image_url
        if medicine_data.is_freeship is not None:
            db_medicine.is_freeship = medicine_data.is_freeship
        if medicine_data.how_to_use is not None:
            db_medicine.how_to_use = medicine_data.how_to_use  # Thêm trường how_to_use
        db.commit()
        db.refresh(db_medicine)
    return db_medicine

# Hàm xóa thuốc
def delete_medicine(db: Session, medicine_id: int) -> Optional[Medicines]:
    db_medicine = get_medicine_by_id(db, medicine_id)
    if db_medicine:
        db.delete(db_medicine)
        db.commit()
        return db_medicine
    return None